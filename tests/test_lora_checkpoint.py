import argparse

import pytest
import torch

from utils.model_utils import bind_lora_to_model, load_start_checkpoint

lora = pytest.importorskip('loralib')

CONFIG = {'lora': {'r': 4, 'lora_alpha': 8, 'lora_dropout': 0.0, 'merge_weights': False, 'fan_in_fan_out': False}}


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))


def test_loralib_checkpoint_is_applied_at_inference(tmp_path):
    base_path = tmp_path / 'base.ckpt'
    torch.save(_model().state_dict(), base_path)

    trained = bind_lora_to_model(CONFIG, _model())
    with torch.no_grad():
        for name, param in trained.named_parameters():
            if 'lora_B' in name:
                param.normal_()
    # Same layout as save_weights after a DataParallel loralib run
    lora_path = tmp_path / 'lora.ckpt'
    torch.save({'model_state_dict': lora.lora_state_dict(torch.nn.DataParallel(trained))}, lora_path)

    # any model_type other than htdemucs/apollo takes the plain state-dict path
    args = argparse.Namespace(
        start_check_point=str(base_path), model_type='bs_roformer', lora_checkpoint_loralib=str(lora_path)
    )
    model = _model()
    load_start_checkpoint(args, model, torch.load(base_path), type_='inference', config=CONFIG)

    x = torch.randn(3, 4)
    torch.testing.assert_close(model(x), trained(x))
    assert not torch.allclose(model(x), _model()(x))
