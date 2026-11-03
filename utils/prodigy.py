import math
from collections import defaultdict

import torch


class Prodigy(torch.optim.Optimizer):
    r"""
    Prodigy (prodigyopt 1.1.2) with the same math and state, but multi-tensor (torch._foreach_*) updates.

    prodigyopt loops over parameters in Python and calls .item() twice per parameter, which costs from
    ~0.2 s to more than 1 s per step on models with hundreds or thousands of tensors. Here the global sums
    for the D estimate stay on the GPU and are read back with one sync per step.

    State keys, shapes and param_group fields are the same as in prodigyopt 1.1.2, so optimizer checkpoints
    can be loaded by either implementation. FSDP is not supported (use prodigyopt.Prodigy with fsdp_in_use).

    Arguments are the same as in prodigyopt.Prodigy:
        params, lr=1.0, betas=(0.9, 0.999), beta3=None (sqrt(beta2)), eps=1e-8, weight_decay=0, decouple=True,
        use_bias_correction=False, safeguard_warmup=False, d0=1e-6, d_coef=1.0, growth_rate=inf,
        fsdp_in_use=False, slice_p=1 (use every slice_p-th element of each tensor for the D estimate)
    """
    # Max elements per batch of foreach ops: bounds the temporary memory (p0 - p, sqrt(exp_avg_sq), ...)
    foreach_numel: int = 2 ** 26

    def __init__(self, params, lr=1.0,
                 betas=(0.9, 0.999), beta3=None,
                 eps=1e-8, weight_decay=0, decouple=True,
                 use_bias_correction=False, safeguard_warmup=False,
                 d0=1e-6, d_coef=1.0, growth_rate=float('inf'),
                 fsdp_in_use=False,
                 slice_p=1):
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if fsdp_in_use:
            raise ValueError("utils.prodigy.Prodigy doesn't support FSDP, use prodigyopt.Prodigy")

        if decouple and weight_decay > 0:
            print("Using decoupled weight decay")

        defaults = dict(lr=lr, betas=betas, beta3=beta3,
                        eps=eps, weight_decay=weight_decay,
                        d=d0, d0=d0, d_max=d0,
                        d_numerator=0.0, d_coef=d_coef,
                        k=0, growth_rate=growth_rate,
                        use_bias_correction=use_bias_correction,
                        decouple=decouple, safeguard_warmup=safeguard_warmup,
                        fsdp_in_use=fsdp_in_use,
                        slice_p=slice_p)
        self.d0 = d0
        super().__init__(params, defaults)

    def _chunks(self, group, beta1):
        """Params with grads of one param group, split by (device, dtype) and into chunks of bounded size"""
        slice_p = group['slice_p']
        buckets = defaultdict(list)
        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("Prodigy doesn't support sparse gradients")
            if hasattr(p, "_fsdp_flattened"):
                raise RuntimeError("utils.prodigy.Prodigy doesn't support FSDP, use prodigyopt.Prodigy")
            state = self.state[p]
            # State initialization (same layout as prodigyopt 1.1.2)
            if 'step' not in state:
                state['step'] = 0
                state['s'] = torch.zeros_like(p.flatten()[::slice_p]).detach()
                if p.any():
                    state['p0'] = p.flatten()[::slice_p].detach().clone()
                else:
                    # All values are zero, so save VRAM with a zero-tensor
                    state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)
                if beta1 > 0:
                    state['exp_avg'] = torch.zeros_like(p).detach()
                state['exp_avg_sq'] = torch.zeros_like(p).detach()
            buckets[(p.device, p.dtype)].append(p)

        chunks = []
        for params in buckets.values():
            start, numel = 0, 0
            for i, p in enumerate(params):
                numel += p.numel()
                if numel >= self.foreach_numel or i == len(params) - 1:
                    chunks.append(params[start:i + 1])
                    start, numel = i + 1, 0
        return chunks

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        use_bias_correction = group['use_bias_correction']
        beta1, beta2 = group['betas']
        beta3 = group['beta3']
        if beta3 is None:
            beta3 = math.sqrt(beta2)
        k = group['k']

        d = group['d']
        d_max = group['d_max']
        d_coef = group['d_coef']
        lr = max(group['lr'] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1

        dlr = d * lr * bias_correction
        growth_rate = group['growth_rate']
        decouple = group['decouple']
        d_numerator = group['d_numerator'] * beta3

        # Per-chunk partial sums of the D estimate, kept on device until a single sync below
        numerator_parts, denom_parts = [], []
        all_chunks = []
        for group in self.param_groups:
            decay = group['weight_decay']
            group_lr = group['lr']
            d0 = group['d0']
            slice_p = group['slice_p']

            if group_lr not in [lr, 0.0]:
                raise RuntimeError(f"Setting different lr values in different parameter groups is only supported for values of 0")

            chunks = self._chunks(group, beta1)
            all_chunks.append(chunks)
            for params in chunks:
                grads = [p.grad for p in params]
                states = [self.state[p] for p in params]

                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    torch._foreach_add_(grads, params, alpha=decay)

                if group_lr > 0.0:
                    sliced_grads = [g.flatten()[::slice_p] for g in grads]
                    sliced_params = [p.flatten()[::slice_p] for p in params]
                    # p0 - p: p0 is a 0-dim zero for parameters that were all zeros at the first step
                    diff = [st['p0'] - sp if st['p0'].ndim == 0 else None for st, sp in zip(states, sliced_params)]
                    full = [i for i, x in enumerate(diff) if x is None]
                    if full:
                        for i, x in zip(full, torch._foreach_sub([states[i]['p0'] for i in full], [sliced_params[i] for i in full])):
                            diff[i] = x
                    torch._foreach_mul_(diff, sliced_grads)
                    # dot(grad, p0 - p) of all tensors, accumulated in float64
                    # we use d / d0 instead of just d to avoid getting values that are too small
                    numerator_parts.append((d / d0) * dlr * torch.cat([x.reshape(-1) for x in diff]).sum(dtype=torch.float64))
                    del diff

                    # Adam EMA updates
                    if beta1 > 0:
                        exp_avgs = [st['exp_avg'] for st in states]
                        torch._foreach_mul_(exp_avgs, beta1)
                        torch._foreach_add_(exp_avgs, grads, alpha=d * (1 - beta1))
                    exp_avg_sqs = [st['exp_avg_sq'] for st in states]
                    torch._foreach_mul_(exp_avg_sqs, beta2)
                    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=d * d * (1 - beta2))

                    s = [st['s'] for st in states]
                    torch._foreach_mul_(s, beta3)
                    if group['safeguard_warmup']:
                        torch._foreach_add_(s, sliced_grads, alpha=((d / d0) * d))
                    else:
                        torch._foreach_add_(s, sliced_grads, alpha=((d / d0) * dlr))
                    denom_parts.append(torch.stack(torch._foreach_norm(s, 1)).sum(dtype=torch.float64))

        d_denom = 0.0
        delta_numerator = 0.0
        if denom_parts:
            # the only device -> host sync of the step
            device = denom_parts[0].device
            delta_numerator, d_denom = torch.stack([
                torch.stack([x.to(device) for x in numerator_parts]).sum(),
                torch.stack([x.to(device) for x in denom_parts]).sum(),
            ]).tolist()

        d_hat = d

        # if we have not done any progres, return
        # if we have any gradients available, will have d_denom > 0 (unless \|g\|=0)
        if d_denom == 0:
            return loss

        if lr > 0.0:
            global_d_numerator = d_numerator + delta_numerator
            global_d_denom = d_denom

            d_hat = d_coef * global_d_numerator / global_d_denom
            if d == group['d0']:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

        for group, chunks in zip(self.param_groups, all_chunks):
            group['d_numerator'] = global_d_numerator
            group['d_denom'] = global_d_denom
            group['d'] = d
            group['d_max'] = d_max
            group['d_hat'] = d_hat

            decay = group['weight_decay']
            eps = group['eps']

            for params in chunks:
                states = [self.state[p] for p in params]
                for st in states:
                    st['step'] += 1

                denom = torch._foreach_sqrt([st['exp_avg_sq'] for st in states])
                torch._foreach_add_(denom, d * eps)

                # Apply weight decay (decoupled variant)
                if decay != 0 and decouple:
                    torch._foreach_add_(params, params, alpha=-decay * dlr)

                # Take step
                if beta1 > 0:
                    torch._foreach_addcdiv_(params, [st['exp_avg'] for st in states], denom, value=-dlr)
                else:
                    torch._foreach_addcdiv_(params, [p.grad for p in params], denom, value=-dlr * d)
                del denom

            group['k'] = group['k'] + 1

        return loss
