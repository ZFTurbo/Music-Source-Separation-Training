"""End-to-end regression for Accelerate with single-target official models."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("model_type", ["bs_roformer_official", "mel_roformer_official"])
@pytest.mark.parametrize("loss_flag", ["--use_mse_loss", "--use_multistft_loss"])
def test_accelerate_single_target_batch_two(tmp_path, model_type, loss_flag):
    pytest.importorskip("accelerate")
    pytest.importorskip("auraloss")
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / f"config_vocals_{model_type}.yaml").read_text())
    config["model"].update(depth=1, num_feature=32)
    config["audio"].update(chunk_size=8192, min_mean_abs=0)
    config["training"].update(batch_size=2, num_epochs=1, num_steps=1, read_metadata_procs=1, use_amp=False)
    config["inference"].update(batch_size=2, num_overlap=2)
    config["augmentations"] = {"enable": False}
    config["loss_multistft"] = {"fft_sizes": [256], "hop_sizes": [64], "win_lengths": [256]}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    song = tmp_path / "data" / "song"
    song.mkdir(parents=True)
    rng = np.random.default_rng(19)
    vocals = rng.normal(0, .05, (16384, 2)).astype("float32")
    other = rng.normal(0, .05, (16384, 2)).astype("float32")
    for name, value in [("vocals", vocals), ("other", other), ("mixture", vocals + other)]:
        sf.write(song / f"{name}.wav", value, 44100, subtype="FLOAT")
    runner = tmp_path / "run_accelerate.py"
    runner.write_text("""
import sys
import torch
import auraloss
import train_accelerate

original_mse = torch.nn.MSELoss.forward
original_stft = auraloss.freq.MultiResolutionSTFTLoss.forward

def checked_mse(self, x, y):
    assert x.shape == y.shape, (x.shape, y.shape)
    assert x.shape[0] == 2
    print('MSE_SHAPES_MATCH', flush=True)
    return original_mse(self, x, y)

def checked_stft(self, x, y):
    assert x.shape == y.shape, (x.shape, y.shape)
    assert x.shape[0] == 2
    print('STFT_SHAPES_MATCH', flush=True)
    return original_stft(self, x, y)

torch.nn.MSELoss.forward = checked_mse
auraloss.freq.MultiResolutionSTFTLoss.forward = checked_stft
train_accelerate.train_model(sys.argv[1:])
""")
    env = dict(os.environ, WANDB_MODE="disabled", PYTHONPATH=str(root))
    result = subprocess.run([
        sys.executable, str(runner), "--model_type", model_type,
        "--config_path", str(config_path), "--data_path", str(song.parent),
        "--valid_path", str(song.parent), "--results_path", str(tmp_path / "results"),
        "--dataset_type", "1", "--num_workers", "0", loss_flag,
    ], cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert ("MSE_SHAPES_MATCH" if loss_flag == "--use_mse_loss" else "STFT_SHAPES_MATCH") in result.stdout
    assert list((tmp_path / "results").glob("*.ckpt"))
