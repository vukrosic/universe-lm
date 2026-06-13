"""Tiger (Chen, Xie, Xiong, Gu, 2024, arXiv:2401.16691) — sign-based
optimizer with per-parameter magnitude EMA.

Canonical Tiger update (per-parameter):

    m_t   = beta1 * m_{t-1} + (1 - beta1) * g_t           (momentum EMA)
    v_t   = beta2 * v_{t-1} + (1 - beta2) * |g_t|          (magnitude EMA)
    update = m_t / (sqrt(v_t) + eps)                        (Adam-like ratio)
    p_{t+1} = p_t - lr * (update + wd * p_t)

vs Lion (sign-only, unit magnitude): Tiger's denominator is the
EMA of |g| — not |g|^2. The ratio gives a per-parameter-adaptive
step magnitude (smaller for params with large recent gradients,
larger for params with small recent gradients). β1=0.9 / β2=0.999
are the paper's defaults.

Cold-start identity at step 0: with m_0=0 and v_0=0, the first
update is `0 / (0 + eps) = 0` ⇒ no parameter change at step 0 ⇒
bit-identical to baseline. (The paper's warm-start `v_0 = |g_0|`
is NOT used here; that path would shift the first step to a unit
sign step on g_0, deviating from the byte-identical contract.)
"""
import torch
from torch.optim.optimizer import Optimizer


class Tiger(Optimizer):
    """Tiger — sign-based optimizer with per-parameter magnitude EMA.

    Parameters
    ----------
    params : iterable
    lr : float — Tiger's step size (paper: ~AdamW LR / 5..10).
    betas : (β1, β2) — β1 weights the gradient vs momentum; β2 is
        the EMA on |g| (magnitude).
    eps : float — additive to √v in the denominator (sign-stability).
    weight_decay : float — decoupled (AdamW style), applied after
        the ratio step.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

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

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Tiger does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_mag"] = torch.zeros_like(p)

                m = state["exp_avg"]
                v = state["exp_avg_mag"]
                # Match the Muon path's `g.float()` so mixed-precision
                # params don't produce a bf16 EMA buffer.
                g = grad.float()
                m_fp = m.float() if m.dtype != torch.float32 else m
                v_fp = v.float() if v.dtype != torch.float32 else v

                # m ← β1·m + (1-β1)·g       (momentum EMA on g)
                m_fp.mul_(beta1).add_(g, alpha=1 - beta1)
                # v ← β2·v + (1-β2)·|g|     (magnitude EMA on |g|)
                v_fp.mul_(beta2).add_(g.abs(), alpha=1 - beta2)

                # update = m / (√v + eps)   (Adam-like ratio, sign-stable)
                update = m_fp / (v_fp.sqrt().add_(eps))

                # Decoupled weight decay (AdamW style): p ← (1 - lr·wd)·p - lr·u
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Cast back to param dtype before the in-place step
                update = update.to(p.dtype)
                p.add_(update, alpha=-lr)

                if m_fp is not m:
                    m.copy_(m_fp)
                if v_fp is not v:
                    v.copy_(v_fp)

        return loss
