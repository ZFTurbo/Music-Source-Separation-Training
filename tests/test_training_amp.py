"""Exercise AMP optimizer skips through the real training epoch."""
from types import SimpleNamespace

import pytest
import torch
from torch.optim.swa_utils import AveragedModel

import train


class CountingScheduler:
    def __init__(self, name):
        self.name = name
        self.steps = 0

    def step(self):
        self.steps += 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('scheduler_name', ['linear_scheduler', 'cosine_scheduler'])
@pytest.mark.parametrize('accumulation', [1, 2])
def test_amp_overflow_skips_ema_and_scheduler_then_recovers(monkeypatch, scheduler_name, accumulation):
    model = torch.nn.Linear(1, 1, bias=False).cuda()
    with torch.no_grad():
        model.weight.fill_(0.5)
    ema = AveragedModel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    scheduler = CountingScheduler(scheduler_name)
    calls = []

    def overflow_once(gradient):
        calls.append(1)
        return torch.full_like(gradient, float('nan')) if len(calls) == 1 else gradient

    model.weight.register_hook(overflow_once)
    monkeypatch.setattr(train, 'compute_training_loss',
                        lambda model, *args, **kwargs: (model.weight.square().sum(), None))
    monkeypatch.setattr(train.wandb, 'log', lambda *args, **kwargs: None)
    start_scale = scaler.get_scale()
    train.train_one_epoch(model, SimpleNamespace(training=SimpleNamespace(grad_clip=0)),
                          SimpleNamespace(), optimizer, torch.device('cuda:0'), [0], 0,
                          True, scaler, scheduler, accumulation, [None] * (2 * accumulation),
                          None, {}, ema_model=ema)

    assert len(calls) == 2 * accumulation
    assert scaler.get_scale() == start_scale / 2
    assert scheduler.steps == 1
    assert ema.n_averaged.item() == 1
    assert torch.allclose(model.weight.detach(), torch.full_like(model.weight, 0.4))
    assert torch.allclose(ema.module.weight, model.weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('enabled', [False, True])
def test_successful_training_step_updates_ema_and_scheduler(monkeypatch, enabled):
    model = torch.nn.Linear(1, 1, bias=False).cuda()
    with torch.no_grad():
        model.weight.fill_(0.5)
    ema = AveragedModel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=enabled)
    scheduler = CountingScheduler('linear_scheduler')
    monkeypatch.setattr(train, 'compute_training_loss',
                        lambda model, *args, **kwargs: (model.weight.square().sum(), None))
    monkeypatch.setattr(train.wandb, 'log', lambda *args, **kwargs: None)
    train.train_one_epoch(model, SimpleNamespace(training=SimpleNamespace(grad_clip=0)),
                          SimpleNamespace(), optimizer, torch.device('cuda:0'), [0], 0,
                          enabled, scaler, scheduler, 1, [None], None, {}, ema_model=ema)
    assert scheduler.steps == 1
    assert ema.n_averaged.item() == 1
    assert torch.allclose(model.weight.detach(), torch.full_like(model.weight, 0.4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_amp_scale_underflow_aborts_without_advancing_training(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False).cuda()
    model.weight.register_hook(lambda gradient: torch.full_like(gradient, float('nan')))
    ema = AveragedModel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler('cuda', init_scale=1.401298464324817e-45)
    scheduler = CountingScheduler('linear_scheduler')
    before = model.weight.detach().clone()
    monkeypatch.setattr(train, 'compute_training_loss',
                        lambda model, *args, **kwargs: (model.weight.square().sum(), None))
    monkeypatch.setattr(train.wandb, 'log', lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match='AMP gradient scale is non-finite or zero'):
        train.train_one_epoch(model, SimpleNamespace(training=SimpleNamespace(grad_clip=0)),
                              SimpleNamespace(), optimizer, torch.device('cuda:0'), [0], 0,
                              True, scaler, scheduler, 1, [None, None], None, {}, ema_model=ema)
    assert scaler.get_scale() == 0
    assert torch.equal(before, model.weight.detach())
    assert ema.n_averaged.item() == 0
    assert scheduler.steps == 0
    assert model.weight.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_initial_zero_scale_does_not_update_optimizer_or_ema(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False).cuda()
    ema = AveragedModel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda', init_scale=0.0)
    scheduler = CountingScheduler('linear_scheduler')
    before = model.weight.detach().clone()
    monkeypatch.setattr(train, 'compute_training_loss',
                        lambda model, *args, **kwargs: (model.weight.square().sum(), None))
    monkeypatch.setattr(train.wandb, 'log', lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match='AMP gradient scale is non-finite or zero'):
        train.train_one_epoch(model, SimpleNamespace(training=SimpleNamespace(grad_clip=0)),
                              SimpleNamespace(), optimizer, torch.device('cuda:0'), [0], 0,
                              True, scaler, scheduler, 1, [None], None, {}, ema_model=ema)
    assert torch.equal(before, model.weight.detach())
    assert ema.n_averaged.item() == 0 and scheduler.steps == 0
