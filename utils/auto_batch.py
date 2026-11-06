"""Calibrate physical batch size with the actual training and validation paths."""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import random

import numpy as np
import soundfile as sf
import torch


def _validation_duration(path):
    if os.path.isfile(path):
        return sf.info(path).duration
    # Composed mixtures are represented by a directory. Its audio files also
    # include the sources used to build the mixture.
    durations = [sf.info(entry.path).duration for entry in os.scandir(path)
                 if entry.is_file() and entry.name.lower().endswith(('.wav', '.flac'))]
    if not durations:
        raise ValueError(f'No audio found in validation track {path}')
    return max(durations)


def _chunk_candidates(original, reduction):
    """Longest first, at 1% intervals and including the exact requested bound."""
    fractions = [i / 100 for i in range(1, math.floor(reduction * 100) + 1)]
    if reduction and (not fractions or fractions[-1] < reduction - 1e-9):
        fractions.append(reduction)
    return list(dict.fromkeys(max(1, round(original * (1 - value))) for value in fractions))


def select_max_batch(model, ema_model, config, args, device, device_ids,
                     use_amp, resume_optimizer=None):
    """Choose a batch size; model, EMA, RNG and any resumed optimizer are untouched."""
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    from train import compute_training_loss
    from valid import get_mixture_paths, process_audio_files
    from utils.dataset import prepare_data
    from utils.losses import choice_loss
    from utils.model_utils import get_optimizer

    if device.type != 'cuda' or len(device_ids) != 1:
        raise ValueError('--do_max_batch requires exactly one CUDA device')
    if args.max_batch_limit < 1 or args.max_batch_probe_steps < 1:
        raise ValueError('Batch limit and probe steps must be positive')
    if not 0 < args.max_batch_memory_fraction < 1:
        raise ValueError('--max_batch_memory_fraction must be between 0 and 1')
    if not 0 <= args.max_batch_chunk_reduction <= 0.10:
        raise ValueError('--max_batch_chunk_reduction must be between 0 and 0.10')
    if args.max_batch_chunk_reduction and args.model_type not in ('bs_roformer', 'mel_band_roformer'):
        raise ValueError('Chunk reduction is supported only for bs_roformer and mel_band_roformer')
    augment = getattr(config, 'augmentations', {})
    if args.max_batch_chunk_reduction and getattr(augment, 'enable', False) and getattr(augment, 'chunk_size_augm', False):
        raise ValueError('Disable chunk_size_augm before using automatic chunk reduction')
    accumulation = int(getattr(config.training, 'gradient_accumulation_steps', 1))
    if accumulation < 1:
        raise ValueError('gradient_accumulation_steps must be positive')

    extension = getattr(config.inference, 'extension', 'wav')
    paths = get_mixture_paths(args, False, config, extension)
    if not paths:
        raise ValueError('--do_max_batch requires a validation track')
    validation_path = max(paths, key=_validation_duration)
    print(f'Auto batch: validation probe: {validation_path}', flush=True)
    original_chunk, original_batch = int(config.audio.chunk_size), int(config.training.batch_size)
    model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ema_state = ({k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
                 if ema_model is not None else None)
    model_mode = model.training
    ema_mode = ema_model.training if ema_model is not None else None
    selected = None
    attempts = []

    def restore():
        model.zero_grad(set_to_none=True)
        model.load_state_dict(model_state)
        model.train(model_mode)
        if ema_model is not None:
            ema_model.load_state_dict(ema_state)
            ema_model.train(ema_mode)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])

    def probe(batch, chunk):
        config.audio.chunk_size = chunk
        config.training.batch_size = batch
        restore()
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        external = max(0, total - free - torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        optimizer = get_optimizer(config, model)
        if resume_optimizer is not None:
            optimizer.load_state_dict(copy.deepcopy(resume_optimizer))
        try:
            scaler = torch.amp.GradScaler('cuda', enabled=bool(use_amp))
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp))
        loss_fn = choice_loss(args, config)
        probe_args = copy.copy(args)
        probe_args.num_workers = 0
        probe_args.persistent_workers = False
        probe_args.prefetch_factor = None
        loader = prepare_data(config, probe_args, batch)
        iterator = iter(loader)
        steps = max(args.max_batch_probe_steps, accumulation * 2)
        model.train()
        for i in range(steps):
            try:
                data = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                data = next(iterator)
            loss, _ = compute_training_loss(model, config, args, data, device, device_ids, loss_fn, use_amp)
            scaler.scale(loss / accumulation).backward()
            if (i + 1) % accumulation == 0 or i == steps - 1:
                scaler.unscale_(optimizer)
                if config.training.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if ema_model is not None:
                    __import__("train").update_ema(ema_model, model)
                optimizer.zero_grad(set_to_none=True)
            del data, loss
        validation_model = ema_model if ema_model is not None else model
        validation_model.eval()
        with torch.no_grad():
            process_audio_files([validation_path], validation_model, args, config, device, is_tqdm=False)
        torch.cuda.synchronize(device)
        free, _ = torch.cuda.mem_get_info(device)
        external = max(external, total - free - torch.cuda.memory_reserved(device))
        return (external + torch.cuda.max_memory_reserved(device)) / total

    def attempt(batch, chunk):
        try:
            used = probe(batch, chunk)
            ok = used <= args.max_batch_memory_fraction
            attempts.append({'batch_size': batch, 'chunk_size': chunk,
                             'estimated_peak_fraction': used, 'accepted': ok})
            print(f'Auto batch: batch={batch}, chunk={chunk}, estimated peak={used:.1%}: '
                  f'{"OK" if ok else "above limit"}', flush=True)
            return ok
        except torch.cuda.OutOfMemoryError:
            attempts.append({'batch_size': batch, 'chunk_size': chunk,
                             'estimated_peak_fraction': None, 'accepted': False,
                             'reason': 'cuda_oom'})
            print(f'Auto batch: batch={batch}, chunk={chunk}: CUDA out of memory', flush=True)
            return False
        finally:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

    try:
        chunks = _chunk_candidates(original_chunk, args.max_batch_chunk_reduction)
        if args.model_type == 'mel_band_roformer' and not getattr(config.model, 'match_input_audio_length', False):
            # Without an explicit ISTFT length, output is rounded to the STFT hop.
            # Round candidates up so the requested reduction bound is respected.
            hop = int(getattr(config.model, 'stft_hop_length', 512))
            chunks = list(dict.fromkeys(math.ceil(chunk / hop) * hop for chunk in chunks))
            chunks = [chunk for chunk in chunks if chunk < original_chunk]
        for chunk in [original_chunk] + chunks:
            candidate = 1 if selected is None else selected[0] + 1
            while candidate <= args.max_batch_limit:
                if not attempt(candidate, chunk):
                    break
                selected = (candidate, chunk)
                candidate += 1
            if selected and selected[0] == args.max_batch_limit:
                print('Auto batch reached --max_batch_limit; maximum may be higher.', flush=True)
                break
        if selected is None:
            raise RuntimeError('No batch fits the requested GPU memory limit, including validation')
    finally:
        restore()
        config.training.batch_size, config.audio.chunk_size = original_batch, original_chunk
        gc.collect()
        torch.cuda.empty_cache()
    config.training.batch_size, config.audio.chunk_size = selected
    report = {'batch_size': selected[0], 'chunk_size': selected[1],
              'original_chunk_size': original_chunk,
              'memory_fraction_limit': args.max_batch_memory_fraction,
              'validation_path': os.fspath(validation_path), 'attempts': attempts}
    with open(os.path.join(args.results_path, 'auto_batch.json'), 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    print(f'Auto batch selected: batch={selected[0]}, chunk={selected[1]} samples '
          '(config file unchanged)', flush=True)
