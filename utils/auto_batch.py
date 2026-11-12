"""Calibrate physical batch size with the actual training and validation paths."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import gc
import json
import math
import os
import random
import tempfile
import uuid

from filelock import FileLock

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


def _write_report(path, report):
    """Replace the report atomically, so readers never see a partial JSON file."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=os.path.dirname(path),
                                         delete=False) as file:
            temporary = file.name
            json.dump(report, file, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


class _BatchAttempt:
    def __init__(self, results_path):
        self.path = os.path.join(results_path, 'auto_batch.json')
        self.lock = FileLock(self.path + '.lock')
        self.report = {'status': 'running', 'attempt_id': uuid.uuid4().hex}
        self.finished = False

    def start(self):
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        with self.lock:
            try:
                with open(self.path, encoding='utf-8') as file:
                    old = json.load(file)
                if isinstance(old, dict):
                    previous = old if old.get('status', 'completed') == 'completed' else old.get('previous_success')
                    if previous is not None:
                        self.report['previous_success'] = previous
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            _write_report(self.path, self.report)

    def finish(self, report):
        with self.lock:
            try:
                with open(self.path, encoding='utf-8') as file:
                    current = json.load(file)
            except (FileNotFoundError, json.JSONDecodeError) as error:
                raise RuntimeError('Auto batch report disappeared or became invalid during the attempt') from error
            if not isinstance(current, dict) or not current.get('attempt_id'):
                raise RuntimeError('Auto batch report has no valid attempt identifier')
            if current['attempt_id'] == self.report['attempt_id']:
                _write_report(self.path, report)
        self.finished = True

    def complete(self, report):
        report.update(status='completed', attempt_id=self.report['attempt_id'])
        self.finish(report)

    def fail(self, error):
        report = dict(self.report, status='failed', error_type=type(error).__name__, error=str(error))
        self.finish(report)


@contextmanager
def batch_attempt(results_path):
    """Track a batch attempt from startup, without overwriting newer attempts."""
    attempt = _BatchAttempt(results_path)
    attempt.start()
    try:
        yield attempt
    except BaseException as error:
        if not attempt.finished:
            try:
                attempt.fail(error)
            except Exception as report_error:
                if hasattr(error, 'add_note'):
                    error.add_note(f'Could not update the auto batch report: {report_error}')
        raise


def select_max_batch(model, ema_model, config, args, device, device_ids,
                     use_amp, resume_optimizer=None, *, attempt=None):
    """Choose a batch and record its result for the current attempt."""
    def select(attempt):
        report = _select_max_batch(model, ema_model, config, args, device, device_ids,
                                   use_amp, resume_optimizer)
        attempt.complete(report)

    if attempt is not None:
        select(attempt)
    else:
        with batch_attempt(args.results_path) as attempt:
            select(attempt)


def _select_max_batch(model, ema_model, config, args, device, device_ids,
                      use_amp, resume_optimizer=None):
    """Choose a batch size; model, EMA, RNG and any resumed optimizer are untouched."""
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    from train import compute_training_loss, update_ema
    from valid import get_mixture_paths, process_audio_files
    from utils.dataset import prepare_data
    from utils.losses import choice_loss
    from utils.model_utils import get_optimizer, prefer_target_instrument

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
        probe_config = copy.deepcopy(config)
        probe_augment = getattr(probe_config, 'augmentations', {})
        if (getattr(probe_augment, 'enable', False)
                and getattr(probe_augment, 'chunk_size_augm', False)
                and 'chunk_size_min' in probe_augment and 'chunk_size_max' in probe_augment):
            max_chunk = min(chunk, int(probe_augment.chunk_size_max))
            if int(probe_augment.chunk_size_min) < 1 or int(probe_augment.chunk_size_min) > max_chunk:
                raise ValueError('Invalid chunk_size_augm range for the available chunk length')
            probe_augment.chunk_size_min = max_chunk
            probe_augment.chunk_size_max = max_chunk
        loader = prepare_data(probe_config, probe_args, batch)
        iterator = iter(loader)
        steps = max(args.max_batch_probe_steps, accumulation * 2)
        max_steps = steps + 16 * accumulation
        processed = 0
        successful_updates = 0
        model.train()
        while (processed < steps or successful_updates < 2) and processed < max_steps:
            window = min(accumulation, steps - processed) if processed < steps else accumulation
            for _ in range(window):
                try:
                    data = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    data = next(iterator)
                loss, _ = compute_training_loss(model, config, args, data, device, device_ids, loss_fn, use_amp)
                if not torch.isfinite(loss).all():
                    raise RuntimeError('Auto batch probe encountered non-finite training loss')
                scaler.scale(loss / accumulation).backward()
                processed += 1
                del data, loss
            scaler.unscale_(optimizer)
            if not scaler.is_enabled() and any(
                not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            ):
                raise RuntimeError('Auto batch probe encountered non-finite gradients')
            if config.training.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            old_scale = scaler.get_scale()
            if not math.isfinite(old_scale) or old_scale <= 0:
                raise RuntimeError('AMP gradient scale is non-finite or zero during auto batch')
            scaler.step(optimizer)
            scaler.update()
            current_scale = scaler.get_scale()
            if not math.isfinite(current_scale) or current_scale <= 0:
                raise RuntimeError('AMP gradient scale is non-finite or zero during auto batch')
            step_succeeded = current_scale >= old_scale
            if step_succeeded:
                successful_updates += 1
                if ema_model is not None:
                    update_ema(ema_model, model)
            optimizer.zero_grad(set_to_none=True)
        if successful_updates < 2:
            raise RuntimeError('Auto batch probe could not complete two optimizer updates due to AMP overflow')
        validation_model = ema_model if ema_model is not None else model
        validation_model.eval()
        required = {(metric, instr) for metric in args.metrics
                    for instr in prefer_target_instrument(config)}
        coverage = set()
        validation_paths = [validation_path] + [path for path in paths if path != validation_path]
        with torch.no_grad():
            for path in validation_paths:
                metrics = process_audio_files([path], validation_model, args, config, device, is_tqdm=False)
                for metric, per_instrument in metrics.items():
                    for instr, measurements in per_instrument.items():
                        if not all(math.isfinite(float(value)) for value in measurements):
                            raise RuntimeError('Auto batch validation produced no finite metrics')
                        if measurements:
                            coverage.add((metric, instr))
                if required.issubset(coverage):
                    break
        if not required.issubset(coverage):
            missing = ', '.join(f'{metric}/{instr}' for metric, instr in sorted(required - coverage))
            raise RuntimeError(f'Auto batch validation produced no finite metrics for: {missing}')
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
    print(f'Auto batch selected: batch={selected[0]}, chunk={selected[1]} samples '
          '(config file unchanged)', flush=True)
    return report
