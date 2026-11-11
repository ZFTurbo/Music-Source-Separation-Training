import torch

from utils.model_utils import load_not_compatible_weights, strip_compile_prefix


def test_nested_compiled_checkpoint_loads_all_weights():
    source = torch.nn.Sequential(torch.nn.Linear(2, 3))
    # torch.compile wraps a nested module with _orig_mod in its state dict.
    source[0] = torch.compile(source[0], backend="eager")
    source(torch.randn(1, 2)).sum().backward()
    compiled = source.state_dict()
    assert "0._orig_mod.weight" in compiled
    target = torch.nn.Sequential(torch.nn.Linear(2, 3))
    load_not_compatible_weights(target, {"model_state_dict": compiled})
    torch.testing.assert_close(target[0].weight, source[0].weight)
    torch.testing.assert_close(target[0].bias, source[0].bias)
    target.load_state_dict(strip_compile_prefix(compiled))


def test_incompatible_rank_is_reported_and_preserves_weights(capsys):
    target = torch.nn.Linear(2, 3)
    before = target.weight.detach().clone()
    load_not_compatible_weights(target, {"weight": torch.ones(6)})
    torch.testing.assert_close(target.weight, before)
    assert "checkpoint loaded partially" in capsys.readouterr().out
