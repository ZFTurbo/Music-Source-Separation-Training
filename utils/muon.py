import math
from typing import List, Tuple

import torch
from torch import nn
from torch.distributed import all_gather, get_rank, get_world_size
from torch.optim import Optimizer

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Loss, Parameters, ParamGroup
from pytorch_optimizer.optimizer.shampoo_utils import zero_power_via_newton_schulz_5


def get_adjusted_lr(lr: float, param_shape: Tuple[float, ...], use_adjusted_lr: bool = False) -> float:
    r"""Get the adjust learning rate."""
    output_shape, *input_shape = param_shape
    input_shape = math.prod(input_shape)

    ratio: float = (
        math.pow(max(1.0, output_shape / input_shape), 0.5)
        if use_adjusted_lr
        else 0.2 * math.sqrt(max(output_shape, input_shape))
    )

    return lr * ratio


class Muon(BaseOptimizer):
    """Momentum Orthogonalized by Newton-schulz.

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-processing step, in which
    each 2D parameter's update is replaced with the nearest orthogonal matrix. To efficiently orthogonalize each
    update, we use a Newton-Schulz iteration, which has the advantage that it can be stably run in bfloat16 on the GPU.

    Muon is intended to optimize only the internal ≥2D parameters of a network. Embeddings, classifier heads, and
    scalar or vector parameters should be optimized using AdamW.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for fine-tuning pretrained models, but we haven't tested this.

    Args:
        params (Parameters): The parameters to be optimized by Muon.
        lr (float): Learning rate.
        momentum (float): The momentum used by the internal SGD.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        nesterov (bool): Whether to use nesterov momentum.
        ns_steps (int): The number of Newton-Schulz iterations to run. (5 is probably always enough)
        use_adjusted_lr (bool): Whether to use adjusted learning rate, which is from the Moonlight.
            Reference: https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
        adamw_lr (float): The learning rate for the internal AdamW.
        adamw_betas (tuple): The betas for the internal AdamW.
        adamw_wd (float): The weight decay for the internal AdamW.
        adamw_eps (float): The epsilon for the internal AdamW.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.

    Example:
        from pytorch_optimizer import Muon

        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        non_hidden_params = [*model.head.parameters(), *model.embed.parameters()]

        param_groups = [
            dict(params=hidden_weights, lr=0.02, weight_decay=0.01, use_muon=True),
            dict(
                params=hidden_gains_biases + non_hidden_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                use_muon=False,
            ),
        ]

        optimizer = Muon(param_groups)
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 2e-2,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        nesterov: bool = True,
        ns_steps: int = 5,
        use_adjusted_lr: bool = False,
        adamw_lr: float = 3e-4,
        adamw_betas: Betas = (0.9, 0.95),
        adamw_wd: float = 0.0,
        adamw_eps: float = 1e-10,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(adamw_lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_positive(ns_steps, 'ns_steps')
        self.validate_betas(adamw_betas)
        self.validate_non_negative(adamw_wd, 'adamw_wd')
        self.validate_non_negative(adamw_eps, 'adamw_eps')

        self.maximize = maximize

        for group in params:
            if 'use_muon' not in group:
                raise ValueError('`use_muon` must be set.')

            if group['use_muon']:
                group['lr'] = group.get('lr', lr)
                group['momentum'] = group.get('momentum', momentum)
                group['nesterov'] = group.get('nesterov', nesterov)
                group['weight_decay'] = group.get('weight_decay', weight_decay)
                group['ns_steps'] = group.get('ns_steps', ns_steps)
                group['use_adjusted_lr'] = group.get('use_adjusted_lr', use_adjusted_lr)
            else:
                group['lr'] = group.get('lr', adamw_lr)
                group['betas'] = group.get('betas', adamw_betas)
                group['eps'] = group.get('eps', adamw_eps)
                group['weight_decay'] = group.get('weight_decay', adamw_wd)

            group['weight_decouple'] = group.get('weight_decouple', weight_decouple)

        super().__init__(params, kwargs)

    def __str__(self) -> str:
        return 'Muon'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0:
                if group['use_muon']:
                    state['momentum_buffer'] = torch.zeros_like(p)
                else:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=False,
                )

                if group['use_muon']:
                    buf = state['momentum_buffer']
                    buf.lerp_(grad, weight=1.0 - group['momentum'])

                    update = grad.lerp_(buf, weight=group['momentum']) if group['nesterov'] else buf
                    if update.ndim > 2:
                        update = update.view(len(update), -1)

                    update = zero_power_via_newton_schulz_5(update, num_steps=group['ns_steps'])

                    if group.get('cautious'):
                        self.apply_cautious(update, grad)

                    lr: float = get_adjusted_lr(group['lr'], p.size(), use_adjusted_lr=group['use_adjusted_lr'])

                    p.add_(update.reshape(p.shape), alpha=-lr)
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    beta1, beta2 = group['betas']

                    bias_correction1: float = self.debias(beta1, group['step'])
                    bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().add_(group['eps']).div_(bias_correction2_sq)

                    p.addcdiv_(exp_avg / bias_correction1, de_nom, value=-group['lr'])

        return loss


class DistributedMuon(BaseOptimizer):  # pragma: no cover
    """Momentum Orthogonalized by Newton-schulz.

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-processing step, in which
    each 2D parameter's update is replaced with the nearest orthogonal matrix. To efficiently orthogonalize each
    update, we use a Newton-Schulz iteration, which has the advantage that it can be stably run in bfloat16 on the GPU.

    Muon is intended to optimize only the internal ≥2D parameters of a network. Embeddings, classifier heads, and
    scalar or vector parameters should be optimized using AdamW.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for fine-tuning pretrained models, but we haven't tested this.

    Args:
        params (Parameters): The parameters to be optimized by Muon.
        lr (float): Learning rate.
        momentum (float): The momentum used by the internal SGD.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        nesterov (bool): Whether to use nesterov momentum.
        ns_steps (int): The number of Newton-Schulz iterations to run. (5 is probably always enough)
        use_adjusted_lr (bool): Whether to use adjusted learning rate, which is from the Moonlight.
            Reference: https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
        adamw_lr (float): The learning rate for the internal AdamW.
        adamw_betas (tuple): The betas for the internal AdamW.
        adamw_wd (float): The weight decay for the internal AdamW.
        adamw_eps (float): The epsilon for the internal AdamW.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.

    Example:
        from pytorch_optimizer import DistributedMuon

        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        non_hidden_params = [*model.head.parameters(), *model.embed.parameters()]

        param_groups = [
            dict(params=hidden_weights, lr=0.02, weight_decay=0.01, use_muon=True),
            dict(
                params=hidden_gains_biases + non_hidden_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                use_muon=False,
            ),
        ]

        optimizer = DistributedMuon(param_groups)
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 2e-2,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        nesterov: bool = True,
        ns_steps: int = 5,
        use_adjusted_lr: bool = False,
        adamw_lr: float = 3e-4,
        adamw_betas: Betas = (0.9, 0.95),
        adamw_wd: float = 0.0,
        adamw_eps: float = 1e-10,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(adamw_lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_positive(ns_steps, 'ns_steps')
        self.validate_betas(adamw_betas)
        self.validate_non_negative(adamw_wd, 'adamw_wd')
        self.validate_non_negative(adamw_eps, 'adamw_eps')

        self.maximize = maximize

        self.world_size: int = get_world_size()
        self.rank: int = get_rank()

        for group in params:
            if 'use_muon' not in group:
                raise ValueError('`use_muon` must be set.')

            if group['use_muon']:
                group['lr'] = group.get('lr', lr)
                group['momentum'] = group.get('momentum', momentum)
                group['nesterov'] = group.get('nesterov', nesterov)
                group['weight_decay'] = group.get('weight_decay', weight_decay)
                group['ns_steps'] = group.get('ns_steps', ns_steps)
                group['use_adjusted_lr'] = group.get('use_adjusted_lr', use_adjusted_lr)
            else:
                group['lr'] = group.get('lr', adamw_lr)
                group['betas'] = group.get('betas', adamw_betas)
                group['eps'] = group.get('eps', adamw_eps)
                group['weight_decay'] = group.get('weight_decay', adamw_wd)

            group['weight_decouple'] = group.get('weight_decouple', weight_decouple)

        super().__init__(params, kwargs)

    def __str__(self) -> str:
        return 'DistributedMuon'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                p.grad = torch.zeros_like(p)

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0 and not group['use_muon']:
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            if group['use_muon']:
                params = group['params']
                padded_params = params + [torch.empty_like(params[-1])] * (
                    self.world_size - len(params) % self.world_size
                )

                for i in range(len(params))[:: self.world_size]:
                    if i + self.rank < len(params):
                        p = params[i + self.rank]

                        grad = p.grad

                        self.maximize_gradient(grad, maximize=self.maximize)

                        state = self.state[p]
                        if len(state) == 0:
                            state['momentum_buffer'] = torch.zeros_like(p)

                        self.apply_weight_decay(
                            p,
                            grad=grad,
                            lr=group['lr'],
                            weight_decay=group['weight_decay'],
                            weight_decouple=group['weight_decouple'],
                            fixed_decay=False,
                        )

                        buf = state['momentum_buffer']
                        buf.lerp_(grad, weight=1.0 - group['momentum'])

                        update = grad.lerp_(buf, weight=group['momentum']) if group['nesterov'] else buf
                        if update.ndim > 2:
                            update = update.view(len(update), -1)

                        update = zero_power_via_newton_schulz_5(update, num_steps=group['ns_steps'])

                        if group.get('cautious'):
                            self.apply_cautious(update, grad)

                        lr: float = get_adjusted_lr(group['lr'], p.size(), use_adjusted_lr=group['use_adjusted_lr'])

                        p.add_(update.reshape(p.shape), alpha=-lr)

                    all_gather(padded_params[i:i + self.world_size], padded_params[i:i + self.rank])  # fmt: skip
            else:
                for p in group['params']:
                    grad = p.grad

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    beta1, beta2 = group['betas']

                    bias_correction1: float = self.debias(beta1, group['step'])
                    bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().add_(group['eps']).div_(bias_correction2_sq)

                    p.addcdiv_(exp_avg / bias_correction1, de_nom, value=-group['lr'])

        return loss


class AdaMuon(BaseOptimizer):
    """Adaptive Muon optimizer.

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-processing step, in which
    each 2D parameter's update is replaced with the nearest orthogonal matrix. To efficiently orthogonalize each
    update, we use a Newton-Schulz iteration, which has the advantage that it can be stably run in bfloat16 on the GPU.

    Muon is intended to optimize only the internal ≥2D parameters of a network. Embeddings, classifier heads, and
    scalar or vector parameters should be optimized using AdamW.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for fine-tuning pretrained models, but we haven't tested this.

    Args:
        params (Parameters): The parameters to be optimized by Muon.
        lr (float): Learning rate.
        betas (tuple): Coefficients used for computing running averages of gradient and the squared Hessian trace.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        ns_steps (int): The number of Newton-Schulz iterations to run. (5 is probably always enough)
        use_adjusted_lr (bool): Whether to use adjusted learning rate, which is from the Moonlight.
            Reference: https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
        adamw_lr (float): The learning rate for the internal AdamW.
        adamw_betas (tuple): The betas for the internal AdamW.
        adamw_wd (float): The weight decay for the internal AdamW.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.

    Example:
        from pytorch_optimizer import AdaMuon

        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        non_hidden_params = [*model.head.parameters(), *model.embed.parameters()]

        param_groups = [
            dict(params=hidden_weights, lr=0.02, weight_decay=0.01, use_muon=True),
            dict(
                params=hidden_gains_biases + non_hidden_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                use_muon=False,
            ),
        ]

        optimizer = AdaMuon(param_groups)
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 2e-2,
        betas: Betas = (0.9, 0.95),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        ns_steps: int = 5,
        use_adjusted_lr: bool = False,
        adamw_lr: float = 3e-4,
        adamw_betas: Betas = (0.9, 0.999),
        adamw_wd: float = 0.0,
        eps: float = 1e-10,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(adamw_lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_positive(ns_steps, 'ns_steps')
        self.validate_betas(betas)
        self.validate_betas(adamw_betas)
        self.validate_non_negative(adamw_wd, 'adamw_wd')
        self.validate_non_negative(eps, 'eps')

        self.maximize = maximize

        for group in params:
            if 'use_muon' not in group:
                raise ValueError('`use_muon` must be set.')

            if group['use_muon']:
                group['lr'] = group.get('lr', lr)
                group['betas'] = group.get('betas', betas)
                group['weight_decay'] = group.get('weight_decay', weight_decay)
                group['ns_steps'] = group.get('ns_steps', ns_steps)
                group['use_adjusted_lr'] = group.get('use_adjusted_lr', use_adjusted_lr)
            else:
                group['lr'] = group.get('lr', adamw_lr)
                group['betas'] = group.get('betas', adamw_betas)
                group['weight_decay'] = group.get('weight_decay', adamw_wd)

            group['weight_decouple'] = group.get('weight_decouple', weight_decouple)
            group['eps'] = group.get('eps', eps)

        super().__init__(params, kwargs)

    def __str__(self) -> str:
        return 'AdaMuon'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0:
                if group['use_muon']:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p.flatten())
                else:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            beta1, beta2 = group['betas']

            bias_correction1: float = self.debias(beta1, group['step'])
            bias_correction2: float = self.debias(beta2, group['step'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=False,
                )

                if group['use_muon']:
                    m = state['m']
                    m.lerp_(grad, weight=1.0 - beta1)

                    update = m.clone()

                    if update.ndim > 2:
                        update = update.view(len(update), -1)

                    update = zero_power_via_newton_schulz_5(update, num_steps=group['ns_steps']).flatten()

                    v = state['v']
                    v.mul_(beta2).addcmul_(update, update, value=1.0 - beta2)

                    update.div_((v / bias_correction2).sqrt_().add_(group['eps']))
                    update = update.reshape(p.size())

                    update.mul_(0.2 * math.sqrt(p.numel())).div_(update.norm().add_(group['eps']))

                    lr: float = get_adjusted_lr(group['lr'], p.size(), use_adjusted_lr=group['use_adjusted_lr'])

                    p.add_(update, alpha=-lr)
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().add_(group['eps']).div_(math.sqrt(bias_correction2))

                    p.addcdiv_(exp_avg / bias_correction1, de_nom, value=-group['lr'])

        return loss


class AdaGO(BaseOptimizer):
    """AdaGrad Meets Muon: Adaptive Stepsizes for Orthogonal Updates.

    Args:
        params (Parameters): The parameters to be optimized by Muon.
        lr (float): Learning rate.
        momentum (float): The momentum used by the internal SGD.
        weight_decay (float): Weight decay (L2 penalty).
        weight_decouple (bool): The optimizer uses decoupled weight decay as in AdamW.
        nesterov (bool): Whether to use nesterov momentum.
        gamma (float): Gamma factor. Empirically, AdaGO performs robustly across a wide range of gamma values.
        eps (float): Epsilon value. Lower bound eps > 0 on the stepsizes.
        ns_steps (int): The number of Newton-Schulz iterations to run. (5 is probably always enough)
        use_adjusted_lr (bool): Whether to use adjusted learning rate, which is from the Moonlight.
            Reference: https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
        adamw_lr (float): The learning rate for the internal AdamW.
        adamw_betas (tuple): The betas for the internal AdamW.
        adamw_wd (float): The weight decay for the internal AdamW.
        adamw_eps (float): The epsilon for the internal AdamW.
        maximize (bool): Maximize the objective with respect to the params, instead of minimizing.

    Example:
        from pytorch_optimizer import AdaGO

        hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
        non_hidden_params = [*model.head.parameters(), *model.embed.parameters()]

        param_groups = [
            dict(params=hidden_weights, lr=0.02, weight_decay=0.01, use_muon=True),
            dict(
                params=hidden_gains_biases + non_hidden_params,
                lr=3e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
                use_muon=False,
            ),
        ]

        optimizer = AdaGO(param_groups)
    """

    def __init__(
        self,
        params: Parameters,
        lr: float = 5e-2,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        gamma: float = 10.0,
        eps: float = 5e-4,
        v: float = 1e-6,
        nesterov: bool = False,
        ns_steps: int = 5,
        use_adjusted_lr: bool = False,
        adamw_lr: float = 3e-4,
        adamw_betas: Betas = (0.9, 0.95),
        adamw_wd: float = 0.0,
        adamw_eps: float = 1e-10,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_learning_rate(adamw_lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_positive(ns_steps, 'ns_steps')
        self.validate_positive(gamma, 'gamma')
        self.validate_positive(eps, 'eps')
        self.validate_positive(v, 'v')
        self.validate_betas(adamw_betas)
        self.validate_non_negative(adamw_wd, 'adamw_wd')
        self.validate_non_negative(adamw_eps, 'adamw_eps')

        self.maximize = maximize

        for group in params:
            if 'use_muon' not in group:
                raise ValueError('`use_muon` must be set.')

            if group['use_muon']:
                group['lr'] = group.get('lr', lr)
                group['momentum'] = group.get('momentum', momentum)
                group['nesterov'] = group.get('nesterov', nesterov)
                group['weight_decay'] = group.get('weight_decay', weight_decay)
                group['ns_steps'] = group.get('ns_steps', ns_steps)
                group['gamma'] = group.get('gamma', gamma)
                group['eps'] = group.get('eps', eps)
                group['v'] = group.get('v', v)
                group['use_adjusted_lr'] = group.get('use_adjusted_lr', use_adjusted_lr)
            else:
                group['lr'] = group.get('lr', adamw_lr)
                group['betas'] = group.get('betas', adamw_betas)
                group['eps'] = group.get('eps', adamw_eps)
                group['weight_decay'] = group.get('weight_decay', adamw_wd)

            group['weight_decouple'] = group.get('weight_decouple', weight_decouple)

        super().__init__(params, kwargs)

    def __str__(self) -> str:
        return 'AdaGO'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            if grad.is_sparse:
                raise NoSparseGradientError(str(self))

            if torch.is_complex(p):
                raise NoComplexParameterError(str(self))

            state = self.state[p]

            if len(state) == 0:
                if group['use_muon']:
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['v'] = torch.tensor(group['v'], dtype=p.dtype, device=p.device)
                else:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        loss: Loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' not in group:
                self.init_group(group)
                group['step'] = 1
            else:
                group['step'] += 1

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                self.maximize_gradient(grad, maximize=self.maximize)

                state = self.state[p]

                self.apply_weight_decay(
                    p,
                    grad=grad,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    weight_decouple=group['weight_decouple'],
                    fixed_decay=False,
                )

                if group['use_muon']:
                    buf, v = state['momentum_buffer'], state['v']
                    buf.lerp_(grad, weight=1.0 - group['momentum'])

                    v.add_(min(grad.norm(p=2.0).pow(2), group['gamma'] ** 2))

                    update = grad.lerp_(buf, weight=group['momentum']) if group['nesterov'] else buf
                    if update.ndim > 2:
                        update = update.view(len(update), -1)

                    update = zero_power_via_newton_schulz_5(update, num_steps=group['ns_steps'])

                    if group.get('cautious'):
                        self.apply_cautious(update, grad)

                    lr: float = get_adjusted_lr(group['lr'], p.size(), use_adjusted_lr=group['use_adjusted_lr'])

                    p.add_(
                        update.reshape(p.shape),
                        alpha=-max(group['eps'], lr * min(grad.norm(2), group['gamma']) / v).item(),
                    )
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                    beta1, beta2 = group['betas']

                    bias_correction1: float = self.debias(beta1, group['step'])
                    bias_correction2_sq: float = math.sqrt(self.debias(beta2, group['step']))

                    exp_avg.lerp_(grad, weight=1.0 - beta1)
                    exp_avg_sq.lerp_(grad.square(), weight=1.0 - beta2)

                    de_nom = exp_avg_sq.sqrt().add_(group['eps']).div_(bias_correction2_sq)

                    p.addcdiv_(exp_avg / bias_correction1, de_nom, value=-group['lr'])

        return loss


def prepare_muon_parameters(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    adamw_lr: float = 3e-4,
    adamw_wd: float = 0.0,
    **kwargs,
) -> Optimizer:
    """Prepare the parameters for Muon optimizer.

    Be careful at using this function to prepare the parameters for Muon optimizer. It's not likely acting perfectly
    for all cases. So, highly recommend you to create the Muon optimizer manually following by the given example in the
    docstring.
    """
    muon_parameters: List[str] = []
    non_muon_params: List[str] = []

    for _, module in model.named_modules():
        for name, param in module.named_parameters(recurse=False):
            if (
                isinstance(module, (nn.Linear, nn.Conv1d, nn.LSTM, nn.Conv2d))
                and param.ndim >= 2
                and 'head' not in name
            ):
                muon_parameters.append(param)
            else:
                non_muon_params.append(param)

    param_groups: Parameters = [
        {'params': muon_parameters, 'lr': lr, 'weight_decay': weight_decay, 'use_muon': True},
        {'params': non_muon_params, 'lr': adamw_lr, 'weight_decay': adamw_wd, 'use_muon': False},
    ]

    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'adamuon':
        return AdaMuon(param_groups, **kwargs)
    if optimizer_name == 'adago':
        return AdaGO(param_groups, **kwargs)

    return Muon(param_groups, **kwargs)
