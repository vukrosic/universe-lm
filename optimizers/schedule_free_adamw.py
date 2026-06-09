"""Schedule-Free AdamW (Defazio et al. 2024, arXiv:2405.15682).

Eliminates the LR schedule entirely. Uses a 3-point iterate:
  z — gradient-following (Adam step target)
  x — Polyak-Ruppert average of z
  y — interpolation used for the forward/backward pass (stored in p.data)

During training, p.data holds y = (1-β1)*z + β1*x.
During eval, call optimizer.eval() to swap p.data to x (the average);
call optimizer.train() to swap back to y.

No warmup or LR decay is needed — the averaging handles late-training
stabilization. Use a constant LR throughout.

When use_schedule_free_adamw=False (default), this class is never
instantiated — the trainer uses torch.optim.AdamW unchanged.
"""
import math
import torch
from torch.optim.optimizer import Optimizer


class ScheduleFreeAdamW(Optimizer):
    """Schedule-Free AdamW.

    Parameters
    ----------
    params : iterable
    lr : float — constant learning rate (no schedule needed)
    betas : (β1, β2) — β1 controls iterate averaging weight toward z
    eps : float — Adam denominator
    weight_decay : float — decoupled (AdamW style), applied to y
    warmup_steps : int — steps over which c ramps from 1→1/(k+1).
        During warmup, x tracks z exactly (c=1), which is equivalent
        to a standard Adam warm-start. After warmup, averaging kicks in.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, warmup_steps=0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        warmup_steps=warmup_steps,
                        k=0, train_mode=True)
        super().__init__(params, defaults)

    def eval(self):
        """Swap params from y (train) to x (average) for evaluation."""
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta1 = group["betas"][0]
            for p in group["params"]:
                state = self.state[p]
                if "z" not in state:
                    continue
                # Recover x from y and z: y = (1-β1)*z + β1*x → x = (y - (1-β1)*z) / β1
                # But we store x directly, so just swap.
                p.data.copy_(state["x"])
            group["train_mode"] = False

    def train(self):
        """Swap params back from x to y after evaluation."""
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta1 = group["betas"][0]
            for p in group["params"]:
                state = self.state[p]
                if "z" not in state:
                    continue
                # Reconstruct y = (1-β1)*z + β1*x
                y = state["z"].mul(1 - beta1).add_(state["x"], alpha=beta1)
                p.data.copy_(y)
            group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            warmup_steps = group["warmup_steps"]
            k = group["k"]

            # Averaging coefficient: ramp c from 1→1/(k+1) after warmup.
            # During warmup c=1 (x=z, standard Adam warm-start).
            if k < warmup_steps:
                c = 1.0
            else:
                c = 1.0 / (k - warmup_steps + 1)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "ScheduleFreeAdamW does not support sparse gradients"
                    )

                state = self.state[p]
                if len(state) == 0:
                    # p.data is y at step 0 (= z = x = initial params)
                    state["z"] = p.data.clone()
                    state["x"] = p.data.clone()
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] = 0

                state["step"] += 1
                step_t = state["step"]

                z = state["z"]
                x = state["x"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Decoupled weight decay on z (not y — y is reconstructed).
                if weight_decay != 0:
                    z.mul_(1 - lr * weight_decay)

                # Adam moment updates (gradient was computed at y = current p.data).
                bc1 = 1 - beta1 ** step_t
                bc2 = 1 - beta2 ** step_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                adam_update = exp_avg / denom / bc1

                # Update z: gradient-following point.
                z.add_(adam_update, alpha=-lr)

                # Update x: Polyak-Ruppert average of z.
                x.mul_(1 - c).add_(z, alpha=c)

                # Reconstruct y = (1-β1)*z + β1*x for next forward pass.
                p.data.copy_(z.mul(1 - beta1).add_(x, alpha=beta1))

            group["k"] = k + 1

        return loss
