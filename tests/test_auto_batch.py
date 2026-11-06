from pathlib import Path
import random

import numpy as np
import pytest
import soundfile as sf
import torch
from ml_collections import ConfigDict

class TinySeparator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = torch.nn.Conv1d(2, 2, 1)
    def forward(self, x):
        first = self.layer(x)
        return torch.stack((first, x-first), dim=1)

from utils.auto_batch import _chunk_candidates, select_max_batch
from utils.settings import parse_args_train


def test_chunk_candidates_include_exact_bound():
    assert _chunk_candidates(1200, 0.025) == [1188, 1176, 1170]
    assert _chunk_candidates(1200, 0) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('optimizer_name,use_amp', [('adam', False), ('adamw', True), ('adam', True)])
def test_real_gpu_probe_restores_model_ema_rng_and_runs_validation(tmp_path: Path, monkeypatch, optimizer_name, use_amp):
    from torch.optim.swa_utils import AveragedModel
    track = tmp_path / 'data' / 'song'
    track.mkdir(parents=True)
    rng = np.random.default_rng(7)
    vocals = (rng.normal(size=(1024, 2)) * 0.05).astype('float32')
    other = (rng.normal(size=(1024, 2)) * 0.05).astype('float32')
    for name, audio in [('vocals', vocals), ('other', other), ('mixture', vocals + other)]:
        sf.write(track / f'{name}.wav', audio, 8000, subtype='FLOAT')
    config = ConfigDict({
        'audio': {'num_channels': 2, 'chunk_size': 256, 'sample_rate': 8000, 'min_mean_abs': 0},
        'training': {'instruments': ['vocals', 'other'], 'target_instrument': None,
                     'grad_clip': 0, 'batch_size': 1, 'num_steps': 2, 'optimizer': optimizer_name,
                     'lr': 0.001, 'other_fix': False, 'read_metadata_procs': 1,
                     'gradient_accumulation_steps': 2},
        'inference': {'batch_size': 1, 'num_overlap': 2, 'normalize': False},
        'augmentations': {'enable': False},
    })
    args = parse_args_train({'model_type': 'tiny', 'data_path': [str(tmp_path / 'data')],
                             'valid_path': [str(tmp_path / 'data')], 'results_path': str(tmp_path),
                             'dataset_type': 6, 'loss': ['mse_loss'], 'max_batch_limit': 2,
                             'max_batch_memory_fraction': 0.99})
    model = TinySeparator(config).cuda()
    ema = AveragedModel(model)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    ema_state = {k: v.clone() for k, v in ema.state_dict().items()}
    torch_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    python_rng = random.getstate()
    select_max_batch(model, ema, config, args, torch.device('cuda:0'), [0], use_amp)
    assert config.training.batch_size == 2
    assert config.audio.chunk_size == 256
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert all(torch.equal(v, ema.state_dict()[k]) for k, v in ema_state.items())
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    assert python_rng == random.getstate()
    assert all(p.grad is None for p in model.parameters())
    import valid as validation
    validate = validation.process_audio_files
    def reject_larger_batch(*args, **kwargs):
        if config.training.batch_size > 1:
            raise torch.cuda.OutOfMemoryError('simulated validation OOM')
        return validate(*args, **kwargs)
    monkeypatch.setattr(validation, 'process_audio_files', reject_larger_batch)
    select_max_batch(model, ema, config, args, torch.device('cuda:0'), [0], use_amp)
    assert config.training.batch_size == 1
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert all(torch.equal(v, ema.state_dict()[k]) for k, v in ema_state.items())
    def always_fail(*args, **kwargs):
        raise RuntimeError('validation failure')
    monkeypatch.setattr(validation, 'process_audio_files', always_fail)
    with pytest.raises(RuntimeError, match='validation failure'):
        select_max_batch(model, ema, config, args, torch.device('cuda:0'), [0], use_amp)
    assert config.training.batch_size == 1 and config.audio.chunk_size == 256
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('model_type', ['bs_roformer', 'mel_band_roformer'])
def test_roformer_probe_can_trade_a_small_chunk_reduction_for_a_batch(tmp_path, monkeypatch, model_type):
    pytest.importorskip('beartype')
    pytest.importorskip('rotary_embedding_torch')
    from models.bs_roformer.bs_roformer import BSRoformer
    import valid as validation

    track = tmp_path / 'data' / 'song'
    track.mkdir(parents=True)
    rng = np.random.default_rng(9)
    vocals = (rng.normal(size=(1024, 2)) * 0.05).astype('float32')
    other = (rng.normal(size=(1024, 2)) * 0.05).astype('float32')
    for name, audio in [('vocals', vocals), ('other', other), ('mixture', vocals + other)]:
        sf.write(track / f'{name}.wav', audio, 8000, subtype='FLOAT')
    config = ConfigDict({
        'audio': {'num_channels': 2, 'chunk_size': 512, 'sample_rate': 8000, 'min_mean_abs': 0},
        'model': {'dim': 16, 'depth': 1, 'stereo': True, 'num_stems': 1,
                  'time_transformer_depth': 1, 'freq_transformer_depth': 1,
                  'freqs_per_bands': (4, 4, 9), 'dim_head': 8, 'heads': 2,
                  'flash_attn': False, 'dim_freqs_in': 17, 'stft_n_fft': 32,
                  'stft_hop_length': 8, 'stft_win_length': 32,
                  'multi_stft_resolutions_window_sizes': (32,), 'multi_stft_hop_size': 8},
        'training': {'instruments': ['vocals', 'other'], 'target_instrument': 'vocals',
                     'grad_clip': 0, 'batch_size': 1, 'num_steps': 1, 'optimizer': 'adam',
                     'lr': 0.001, 'other_fix': False, 'read_metadata_procs': 1},
        'inference': {'batch_size': 1, 'num_overlap': 2, 'normalize': False},
        'augmentations': {'enable': False},
    })
    args = parse_args_train({'model_type': model_type, 'data_path': [str(tmp_path / 'data')],
                             'valid_path': [str(tmp_path / 'data')], 'results_path': str(tmp_path),
                             'dataset_type': 6, 'max_batch_limit': 2, 'max_batch_probe_steps': 2,
                             'max_batch_memory_fraction': 0.99, 'max_batch_chunk_reduction': 0.02})
    if model_type == 'mel_band_roformer':
        del config.model.freqs_per_bands
        config.model.num_bands = 4
        config.model.sample_rate = 8000
    from models.bs_roformer.mel_band_roformer import MelBandRoformer
    model_cls = BSRoformer if model_type == 'bs_roformer' else MelBandRoformer
    model = model_cls(**dict(config.model)).cuda()
    real_validate = validation.process_audio_files
    def oom_for_large_full_chunk(*args, **kwargs):
        if config.training.batch_size == 2 and config.audio.chunk_size == 512:
            raise torch.cuda.OutOfMemoryError('simulated full-chunk OOM')
        return real_validate(*args, **kwargs)
    monkeypatch.setattr(validation, 'process_audio_files', oom_for_large_full_chunk)
    select_max_batch(model, None, config, args, torch.device('cuda:0'), [0], False)
    assert config.training.batch_size == 2
    assert config.audio.chunk_size == (504 if model_type == 'mel_band_roformer' else 507)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_training_entrypoint_runs_probe_then_epoch(tmp_path, monkeypatch):
    import wandb
    import yaml
    import train as training

    track = tmp_path / 'data' / 'song'
    track.mkdir(parents=True)
    audio = np.full((1024, 2), 0.05, dtype=np.float32)
    for name, samples in [('vocals', audio), ('other', audio), ('mixture', audio * 2)]:
        sf.write(track / f'{name}.wav', samples, 8000, subtype='FLOAT')
    config = {
        'audio': {'num_channels': 2, 'chunk_size': 256, 'sample_rate': 8000, 'min_mean_abs': 0},
        'training': {'instruments': ['vocals', 'other'], 'target_instrument': None,
                     'grad_clip': 0, 'batch_size': 1, 'num_steps': 1, 'num_epochs': 1,
                     'optimizer': 'adam', 'lr': 0.001, 'other_fix': False,
                     'read_metadata_procs': 1, 'patience': 2, 'reduce_factor': 0.5,
                     'use_amp': False, 'ema_momentum': 0.9},
        'inference': {'batch_size': 1, 'num_overlap': 2, 'normalize': False},
        'augmentations': {'enable': False},
    }
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    monkeypatch.setattr(training, 'wandb_init', lambda *args, **kwargs: None)
    monkeypatch.setattr(wandb, 'log', lambda *args, **kwargs: None)
    monkeypatch.setattr(training, 'get_model_from_config', lambda *args: (TinySeparator(ConfigDict(config)), ConfigDict(config)))
    training.train_model({'config_path': str(config_path), 'data_path': [str(tmp_path / 'data')],
        'valid_path': [str(tmp_path / 'data')], 'results_path': str(tmp_path / 'result'),
        'model_type': 'tiny', 'dataset_type': 6, 'loss': ['mse_loss'],
        'do_max_batch': True, 'max_batch_limit': 2, 'max_batch_memory_fraction': 0.99})
    assert (tmp_path / 'result' / 'auto_batch.json').is_file()
    assert list((tmp_path / 'result').glob('*.ckpt'))
