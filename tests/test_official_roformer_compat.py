import numpy as np
import pytest
import torch
import yaml
from ml_collections import ConfigDict

from train import forward_step
from utils.model_utils import demix
from utils.settings import get_model_from_config, parse_args_train


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("model_type", ["bs_roformer_official", "mel_roformer_official"])
def test_single_stem_preserves_model_api_and_works_in_training_and_demix(tmp_path, model_type):
    config_path = f"configs/config_vocals_{model_type}.yaml"
    with open(config_path, encoding="utf8") as file:
        config = yaml.safe_load(file)
    config["model"]["depth"] = 1
    config["model"]["num_feature"] = 32
    config["audio"]["chunk_size"] = 8192
    config["inference"]["batch_size"] = 2
    model_config = tmp_path / "model.yaml"
    model_config.write_text(yaml.safe_dump(config), encoding="utf8")
    model, parsed_config = get_model_from_config(model_type, str(model_config))
    model.cuda()
    x = torch.randn(2, 2, 8193, device="cuda")
    target = torch.randn(2, 1, 2, 8193, device="cuda")
    prediction = model(x)
    assert prediction.shape == x.shape
    loss = forward_step(x, target, None, False, model,
                        lambda prediction, target, mixture: (prediction - target).square().mean(), [0])
    assert loss.isfinite()
    loss.backward()
    separated = demix(parsed_config, model.eval(), x[0].detach().cpu().numpy(), torch.device("cuda"), model_type)
    assert separated["vocals"].shape == (2, 8193)
    assert np.isfinite(separated["vocals"]).all()
    args = parse_args_train({"model_type": model_type})
    assert not any(name.endswith("_loss") and "roformer" in name for name in args.loss)
