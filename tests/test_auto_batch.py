from pathlib import Path
import json
import random
import subprocess
import sys
import threading

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
    optimizer_class = torch.optim.AdamW if optimizer_name == 'adamw' else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=0.001)
    model(torch.randn(1, 2, 256, device='cuda')).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer_state = optimizer.state_dict()
    optimizer_state_copy = {key: {name: value.clone() if torch.is_tensor(value) else value
                                  for name, value in state.items()}
                            for key, state in optimizer_state['state'].items()}
    ema = AveragedModel(model)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    ema_state = {k: v.clone() for k, v in ema.state_dict().items()}
    torch_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    python_rng = random.getstate()
    select_max_batch(model, ema, config, args, torch.device('cuda:0'), [0], use_amp, optimizer_state)
    for key, saved in optimizer_state_copy.items():
        for name, value in saved.items():
            current = optimizer_state['state'][key][name]
            assert torch.equal(current, value) if torch.is_tensor(value) else current == value
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
            _, total = torch.cuda.mem_get_info()
            try:
                torch.cuda.set_per_process_memory_fraction(0.25)
                torch.empty(total // 4, dtype=torch.uint8, device='cuda')
            finally:
                torch.cuda.set_per_process_memory_fraction(1.0)
            pytest.fail('Expected a real CUDA out-of-memory error')
        return validate(*args, **kwargs)
    monkeypatch.setattr(validation, 'process_audio_files', reject_larger_batch)
    select_max_batch(model, ema, config, args, torch.device('cuda:0'), [0], use_amp)
    assert config.training.batch_size == 1
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    assert all(torch.equal(v, ema.state_dict()[k]) for k, v in ema_state.items())
    monkeypatch.setattr(validation, 'process_audio_files', validate)
    import train as training
    real_compute_loss = training.compute_training_loss

    def non_finite_loss(*args, **kwargs):
        loss, aux = real_compute_loss(*args, **kwargs)
        return loss * float('nan'), aux

    monkeypatch.setattr(training, 'compute_training_loss', non_finite_loss)
    with pytest.raises(RuntimeError, match='non-finite training loss'):
        select_max_batch(model, ema, config, args, torch.device('cuda:0'), [0], use_amp)
    assert config.training.batch_size == 1 and config.audio.chunk_size == 256
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())
    monkeypatch.setattr(training, 'compute_training_loss', real_compute_loss)

    for measurements in ([], [float('nan')]):
        monkeypatch.setattr(validation, 'process_audio_files',
                            lambda *args, **kwargs: {'sdr': {'vocals': measurements}})
        with pytest.raises(RuntimeError, match='no finite metrics'):
            select_max_batch(model, ema, config, args, torch.device('cuda:0'), [0], use_amp)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in state.items())

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
                     'use_amp': False, 'ema_momentum': 0.9, 'model_type': 'vit_mel_roformer'},
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
        'model_type': 'dbs_bsrnn', 'dataset_type': 6, 'loss': ['mse_loss'],
        'do_max_batch': True, 'max_batch_limit': 2, 'max_batch_memory_fraction': 0.99})
    report = json.loads((tmp_path / 'result' / 'auto_batch.json').read_text(encoding='utf-8'))
    assert report['status'] == 'completed' and report['batch_size'] >= 1
    assert list((tmp_path / 'result').glob('*.ckpt'))


@pytest.mark.parametrize('model_type', ['bs_roformer_official', 'mel_roformer_official', 'vit_mel_roformer'])
def test_standard_loss_is_preserved_for_non_internal_models(model_type):
    from utils.losses import choice_loss
    args = parse_args_train({'model_type': model_type, 'loss': ['mse_loss']})
    config = ConfigDict({'training': {'q': .95, 'coarse_loss_clip': True},
                         'audio': {'sample_rate': 8000, 'chunk_size': 256}, 'model': {}})
    loss = choice_loss(args, config)(torch.ones(2, 1, 2, 32), torch.zeros(2, 1, 2, 32), None)
    assert args.loss == ['mse_loss']
    assert torch.is_tensor(loss) and loss.item() > 0


def test_failed_attempt_replaces_stale_report_and_preserves_previous_success(tmp_path, monkeypatch):
    import utils.auto_batch as auto_batch

    args = parse_args_train({'results_path': str(tmp_path)})
    report_path = tmp_path / 'auto_batch.json'
    legacy = {'batch_size': 4, 'chunk_size': 1024}
    report_path.write_text(json.dumps(legacy), encoding='utf-8')
    seen = []

    def fail(*args, **kwargs):
        seen.append(json.loads(report_path.read_text(encoding='utf-8')))
        raise RuntimeError('missing validation source')

    monkeypatch.setattr(auto_batch, '_select_max_batch', fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='missing validation source'):
            select_max_batch(None, None, None, args, None, [], False)
        failed = json.loads(report_path.read_text(encoding='utf-8'))
        assert failed['status'] == 'failed'
        assert failed['error_type'] == 'RuntimeError'
        assert failed['previous_success'] == legacy
        assert 'batch_size' not in failed
    assert all(report['status'] == 'running' for report in seen)
    assert seen[0]['attempt_id'] != seen[1]['attempt_id']


@pytest.mark.parametrize('initial,effective,standard,expected', [
    ('dbs_bsrnn', 'vit_mel_roformer', False, ['mse_loss']),
    ('vit_mel_roformer', 'dbs_bsrnn', False, ['dbs_bsrnn_loss']),
    ('dbs_bsrnn', 'bs_roformer', True, ['mse_loss']),
    ('dbs_bsrnn', 'bs_roformer', False, ['bs_roformer_loss']),
])
def test_loss_tracks_effective_model(initial, effective, standard, expected):
    from utils.settings import resolve_training_loss
    args = parse_args_train({'model_type': initial, 'loss': ['mse_loss'], 'use_standard_loss': standard})
    args.model_type = effective
    resolve_training_loss(args)
    assert args.loss == expected
    resolve_training_loss(args)
    assert args.loss == expected


@pytest.mark.parametrize('failure_stage', ['config', 'checkpoint'])
def test_startup_failure_does_not_leave_previous_report_current(tmp_path, monkeypatch, failure_stage):
    import train as training

    report_path = tmp_path / 'auto_batch.json'
    previous = {'status': 'completed', 'attempt_id': 'previous', 'batch_size': 4}
    report_path.write_text(json.dumps(previous), encoding='utf-8')
    monkeypatch.setattr(training, 'initialize_environment', lambda *args: None)

    def fail(*args, **kwargs):
        raise RuntimeError(f'{failure_stage} unavailable')

    args = {'results_path': str(tmp_path), 'do_max_batch': True}
    if failure_stage == 'config':
        monkeypatch.setattr(training, 'get_model_from_config', fail)
    else:
        config = ConfigDict({'training': {'batch_size': 1}})
        monkeypatch.setattr(training, 'get_model_from_config', lambda *args: (object(), config))
        monkeypatch.setattr(torch, 'load', fail)
        args['start_check_point'] = str(tmp_path / 'checkpoint.ckpt')
    with pytest.raises(RuntimeError, match=f'{failure_stage} unavailable'):
        training.train_model(args)
    current = json.loads(report_path.read_text(encoding='utf-8'))
    assert current['status'] == 'failed'
    assert current['attempt_id'] != 'previous'
    assert current['previous_success'] == previous
    assert current['error_type'] == 'RuntimeError'


def test_older_attempt_cannot_replace_newer_failed_report(tmp_path, monkeypatch):
    import utils.auto_batch as auto_batch

    args = parse_args_train({'results_path': str(tmp_path)})
    path = tmp_path / 'auto_batch.json'
    started = threading.Event()
    release = threading.Event()
    errors = []

    def probe(model, *args, **kwargs):
        if model == 'older':
            started.set()
            if not release.wait(timeout=10):
                raise TimeoutError('The newer attempt did not finish')
            return {'batch_size': 2, 'chunk_size': 256}
        raise RuntimeError('newer attempt failed')

    def run_older():
        try:
            select_max_batch('older', None, None, args, None, [], False)
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(auto_batch, '_select_max_batch', probe)
    thread = threading.Thread(target=run_older)
    thread.start()
    try:
        assert started.wait(timeout=10), 'Older attempt did not start'
        with pytest.raises(RuntimeError, match='newer attempt failed'):
            select_max_batch('newer', None, None, args, None, [], False)
        newer = json.loads(path.read_text(encoding='utf-8'))
        assert newer['status'] == 'failed'
    finally:
        release.set()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert not errors
    final = json.loads(path.read_text(encoding='utf-8'))
    assert final['attempt_id'] == newer['attempt_id']
    assert final['status'] == 'failed'


def test_failure_after_successful_probe_keeps_completed_report(tmp_path):
    from utils.auto_batch import batch_attempt

    path = tmp_path / 'auto_batch.json'
    with pytest.raises(RuntimeError, match='later training failure'):
        with batch_attempt(str(tmp_path)) as attempt:
            attempt.complete({'batch_size': 2, 'chunk_size': 256})
            raise RuntimeError('later training failure')
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['status'] == 'completed' and report['batch_size'] == 2


def test_older_failed_attempt_cannot_replace_newer_success(tmp_path):
    from utils.auto_batch import batch_attempt

    path = tmp_path / 'auto_batch.json'
    with pytest.raises(RuntimeError, match='older failure'):
        with batch_attempt(str(tmp_path)):
            with batch_attempt(str(tmp_path)) as newer:
                newer.complete({'batch_size': 3, 'chunk_size': 256})
            raise RuntimeError('older failure')
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['status'] == 'completed' and report['batch_size'] == 3
    assert report['attempt_id'] == newer.report['attempt_id']


def test_another_process_can_supersede_running_attempt(tmp_path):
    from utils.auto_batch import batch_attempt

    path = tmp_path / 'auto_batch.json'
    script = ("import sys; from utils.auto_batch import batch_attempt\n"
              "with batch_attempt(sys.argv[1]) as attempt:\n"
              "    attempt.complete({'batch_size': 3, 'chunk_size': 256})\n")
    with batch_attempt(str(tmp_path)) as older:
        subprocess.run([sys.executable, '-c', script, str(tmp_path)], check=True, timeout=30)
        newer = json.loads(path.read_text(encoding='utf-8'))
        older.complete({'batch_size': 2, 'chunk_size': 256})
    final = json.loads(path.read_text(encoding='utf-8'))
    assert final == newer
    assert final['status'] == 'completed' and final['batch_size'] == 3


@pytest.mark.parametrize('batch_option,path_option', [('--do_max_batch', '--results_path'),
                                                      ('--do_max_b', '--results_p')])
def test_invalid_cli_option_marks_existing_report_failed(tmp_path, monkeypatch, batch_option, path_option):
    import train as training
    path = tmp_path / 'auto_batch.json'
    previous = {'status': 'completed', 'attempt_id': 'old', 'batch_size': 4}
    path.write_text(json.dumps(previous), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['train.py', batch_option, path_option, str(tmp_path),
                                     '--max_batch_limit', 'invalid'])
    with pytest.raises(SystemExit) as error:
        training.train_model(None)
    assert error.value.code == 2
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['status'] == 'failed'
    assert report['error_type'] == 'SystemExit'
    assert report['previous_success'] == previous


def test_help_does_not_replace_previous_report(tmp_path, monkeypatch, capsys):
    import train as training
    path = tmp_path / 'auto_batch.json'
    old = {'status': 'completed', 'attempt_id': 'old', 'batch_size': 4}
    path.write_text(json.dumps(old), encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['train.py', '--do_max_batch', '--results_path', str(tmp_path), '--help'])
    with pytest.raises(SystemExit) as error:
        training.train_model(None)
    assert error.value.code == 0
    assert json.loads(path.read_text(encoding='utf-8')) == old


@pytest.mark.parametrize('damage', ['missing', 'malformed', 'wrong_structure'])
def test_damaged_current_report_aborts_attempt(tmp_path, damage):
    from utils.auto_batch import batch_attempt
    path = tmp_path / 'auto_batch.json'
    with pytest.raises(RuntimeError, match='Auto batch report'):
        with batch_attempt(str(tmp_path)) as attempt:
            if damage == 'missing':
                path.unlink()
            elif damage == 'malformed':
                path.write_text('{', encoding='utf-8')
            else:
                path.write_text('{}', encoding='utf-8')
            attempt.complete({'batch_size': 2, 'chunk_size': 256})
    assert not attempt.finished


def test_programmatic_namespace_resolves_effective_model_loss():
    import argparse
    from utils.settings import resolve_training_loss
    args = parse_args_train(argparse.Namespace(model_type='dbs_bsrnn', loss=['mse_loss']))
    assert args.loss == ['dbs_bsrnn_loss']
    args.model_type = 'vit_mel_roformer'
    resolve_training_loss(args)
    assert args.loss == ['mse_loss']


def test_training_accepts_namespace_and_records_startup_failure(tmp_path, monkeypatch):
    import argparse
    import train as training

    monkeypatch.setattr(training, 'initialize_environment', lambda *args: None)
    def fail(*args, **kwargs):
        raise RuntimeError('config unavailable')
    monkeypatch.setattr(training, 'get_model_from_config', fail)
    args = argparse.Namespace(results_path=str(tmp_path), do_max_batch=True,
                              model_type='dbs_bsrnn', loss=['mse_loss'])
    with pytest.raises(RuntimeError, match='config unavailable'):
        training.train_model(args)
    report = json.loads((tmp_path / 'auto_batch.json').read_text(encoding='utf-8'))
    assert report['status'] == 'failed'
    assert report['error_type'] == 'RuntimeError'


@pytest.mark.parametrize('initial', ['dbs_bsrnn', 'vit_mel_roformer'])
def test_reparsed_namespace_respects_explicit_loss_changes(initial):
    from utils.settings import resolve_training_loss
    args = parse_args_train({'model_type': initial, 'loss': ['mse_loss']})
    args.model_type = 'vit_mel_roformer'
    args.loss = ['l1_loss']
    parsed = parse_args_train(args)
    assert parsed.loss == ['l1_loss']
    parsed.model_type = 'dbs_bsrnn'
    resolve_training_loss(parsed)
    assert parsed.loss == ['dbs_bsrnn_loss']
    parsed.use_standard_loss = True
    resolve_training_loss(parsed)
    assert parsed.loss == ['l1_loss']
