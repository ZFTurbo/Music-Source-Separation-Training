import copy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from ml_collections import ConfigDict

from utils.auto_batch import select_max_batch
from utils.settings import parse_args_train


class Separator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Conv1d(2, 2, 1)
        self.train_lengths = []

    def forward(self, x):
        if self.training:
            self.train_lengths.append(x.shape[-1])
        vocals = self.layer(x)
        return torch.stack((vocals, x - vocals), dim=1)


@pytest.fixture
def probe_fixture(tmp_path):
    train = tmp_path / "train" / "song"
    validation = tmp_path / "valid" / "song"
    train.mkdir(parents=True)
    validation.mkdir(parents=True)
    rng = np.random.default_rng(41)
    vocals = rng.normal(0, 0.05, (1024, 2)).astype("float32")
    other = rng.normal(0, 0.05, (1024, 2)).astype("float32")
    for directory in (train, validation):
        for name, samples in [
            ("vocals", vocals),
            ("other", other),
            ("mixture", vocals + other),
        ]:
            sf.write(directory / f"{name}.wav", samples, 8000, subtype="FLOAT")
    config = ConfigDict(
        {
            "audio": {
                "num_channels": 2,
                "chunk_size": 256,
                "sample_rate": 8000,
                "min_mean_abs": 0,
            },
            "training": {
                "instruments": ["vocals", "other"],
                "target_instrument": None,
                "grad_clip": 0,
                "batch_size": 1,
                "num_steps": 2,
                "optimizer": "adam",
                "lr": 0.001,
                "other_fix": False,
                "read_metadata_procs": 1,
            },
            "inference": {"batch_size": 1, "num_overlap": 2, "normalize": False},
            "augmentations": {"enable": False},
        }
    )
    args = parse_args_train(
        {
            "model_type": "tiny",
            "data_path": [str(train.parent)],
            "valid_path": [str(validation.parent)],
            "results_path": str(tmp_path),
            "dataset_type": 6,
            "loss": ["mse_loss"],
            "max_batch_limit": 1,
            "max_batch_memory_fraction": 0.99,
        }
    )
    return config, args, train, validation


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_probe_uses_longest_augmented_chunk_without_changing_config(
    probe_fixture, monkeypatch
):
    config, args, _, _ = probe_fixture
    config.augmentations.update(
        enable=True, chunk_size_augm=True, chunk_size_min=64, chunk_size_max=192
    )
    old_config = copy.deepcopy(config)
    monkeypatch.setattr("utils.dataset.random.randint", lambda low, high: low)
    model = Separator().cuda()
    select_max_batch(model, None, config, args, torch.device("cuda:0"), [0], False)
    assert model.train_lengths and set(model.train_lengths) == {192}
    assert config.augmentations == old_config.augmentations
    assert config.audio.chunk_size == 256


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_amp_overflow_retries_until_two_real_optimizer_updates(
    probe_fixture, monkeypatch
):
    config, args, _, _ = probe_fixture
    from torch.optim.swa_utils import AveragedModel
    import train

    model = Separator().cuda()
    ema = AveragedModel(model)
    original_ema_update = train.update_ema
    ema_updates = []

    def counted_ema_update(*args):
        ema_updates.append(1)
        return original_ema_update(*args)

    monkeypatch.setattr(train, "update_ema", counted_ema_update)
    gradients = []

    def overflow_once(gradient):
        gradients.append(1)
        return (
            torch.full_like(gradient, float("nan")) if len(gradients) == 1 else gradient
        )

    model.layer.weight.register_hook(overflow_once)
    steps = []
    original_step = torch.optim.Adam.step

    def counted_step(optimizer, *args, **kwargs):
        steps.append(1)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", counted_step)
    select_max_batch(model, ema, config, args, torch.device("cuda:0"), [0], True)
    assert len(gradients) >= 3
    assert len(steps) >= 2
    assert len(ema_updates) == len(steps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("use_amp", [False, True])
def test_nonfinite_gradients_never_accept_batch(probe_fixture, use_amp):
    config, args, _, _ = probe_fixture
    model = Separator().cuda()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    model.layer.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    with pytest.raises(RuntimeError, match="non-finite gradients|optimizer updates"):
        select_max_batch(
            model, None, config, args, torch.device("cuda:0"), [0], use_amp
        )
    assert config.training.batch_size == 1 and config.audio.chunk_size == 256
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in initial.items()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_missing_validation_stem_is_rejected(probe_fixture):
    config, args, _, validation = probe_fixture
    (validation / "other.wav").unlink()
    with pytest.raises(RuntimeError, match="sdr/other"):
        select_max_batch(
            Separator().cuda(), None, config, args, torch.device("cuda:0"), [0], False
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_validation_can_cover_stems_across_tracks(probe_fixture):
    config, args, _, first = probe_fixture
    second = first.parent / "second"
    second.mkdir()
    for filename in ("other.wav", "mixture.wav"):
        (second / filename).write_bytes((first / filename).read_bytes())
    (first / "other.wav").unlink()
    model = Separator().cuda()
    select_max_batch(model, None, config, args, torch.device("cuda:0"), [0], False)
    assert config.training.batch_size == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalid_augmented_chunk_range_is_not_hidden(probe_fixture):
    config, args, _, _ = probe_fixture
    config.augmentations.update(
        enable=True, chunk_size_augm=True, chunk_size_min=257, chunk_size_max=512
    )
    with pytest.raises(ValueError, match="Invalid chunk_size_augm range"):
        select_max_batch(
            Separator().cuda(), None, config, args, torch.device("cuda:0"), [0], False
        )
    assert config.audio.chunk_size == 256


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_probe_rejects_amp_scale_underflow(probe_fixture, monkeypatch):
    config, args, _, _ = probe_fixture
    model = Separator().cuda()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    model.layer.weight.register_hook(lambda grad: torch.full_like(grad, float('nan')))
    original_scaler = torch.cuda.amp.GradScaler
    factory_owner = torch.amp if hasattr(torch.amp, 'GradScaler') else torch.cuda.amp
    monkeypatch.setattr(factory_owner, 'GradScaler',
                        lambda *args, **kwargs: original_scaler(init_scale=1.401298464324817e-45))
    with pytest.raises(RuntimeError, match='AMP gradient scale is non-finite or zero'):
        select_max_batch(model, None, config, args, torch.device('cuda:0'), [0], True)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert config.training.batch_size == 1 and config.audio.chunk_size == 256
