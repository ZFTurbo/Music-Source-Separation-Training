"""Exercise both training and validation for new separators on a real CUDA device."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

from utils.settings import parse_args_train, get_model_from_config
from utils.auto_batch import select_max_batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "name",
    ["bs_roformer_official", "mel_roformer_official", "vit_mel_roformer"],
)
def test_new_model_batch_probe(name, tmp_path):
    root = Path(__file__).resolve().parents[1]
    d = tmp_path
    track = d / "data" / "song"
    track.mkdir(parents=True)
    file = root / "configs" / ("config_musdb18_" + name + ".yaml")
    if name.endswith("_official"):
        file = root / "configs" / ("config_vocals_" + name + ".yaml")
    c = yaml.load(file.read_text(), Loader=yaml.FullLoader)
    m = c["model"]
    for key, v in [
        ("depth", 1),
        ("dim", 64),
        ("n_layers", 1),
        ("n_heads", 4),
        ("heads", 4),
        ("dim_head", 16),
    ]:
        if key in m:
            m[key] = v
    length = 8192
    c["audio"]["chunk_size"] = length
    c["training"].update(
        read_metadata_procs=1, num_steps=2, gradient_accumulation_steps=1
    )
    c["inference"].update(batch_size=1, num_overlap=2)
    c["augmentations"] = {"enable": False}
    rng = np.random.default_rng(3)
    mixture = np.zeros((length * 2, 2), dtype=np.float32)
    for stem in c["training"]["instruments"]:
        audio = rng.normal(0, 0.05, mixture.shape).astype(np.float32)
        mixture += audio
        sf.write(track / (stem + ".wav"), audio, 44100, subtype="FLOAT")
    sf.write(track / "mixture.wav", mixture, 44100, subtype="FLOAT")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(c))
    model, cfg = get_model_from_config(name, str(config_path))
    model.cuda()
    args = parse_args_train(
        dict(
            model_type=name,
            data_path=[str(d / "data")],
            valid_path=[str(d / "data")],
            dataset_type=1,
            results_path=str(d),
            max_batch_limit=2,
            max_batch_probe_steps=2,
            max_batch_memory_fraction=0.98,
            loss=["mse_loss"],
        )
    )
    select_max_batch(model, None, cfg, args, torch.device("cuda:0"), [0], False)
    assert json.loads((d / "auto_batch.json").read_text())["batch_size"] == 2
