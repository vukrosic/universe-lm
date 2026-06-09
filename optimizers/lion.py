"""Lion (Chen et al. 2023, arXiv:2302.06675) — sign-based optimizer.

Canonical Lion update (per-parameter):

    c_t   = beta1 * m_t + (1 - beta1) * g_t
    p_t+1 = p_t - lr * (sign(c_t) + wd * p_t)
    m_t+1 = beta2 * m_t + (1 - beta2) * g_t

with `beta1, beta2 = 0.9, 0.98` (paper) and `wd` decoupled (AdamW style).
The `sign` of an interpolation between momentum and grad makes the update
direction only ever +/- 1 — bounded magnitude regardless of gradient scale.

Cautious variant (Liang et al. 2024, arXiv:2411.16085) — the "Cautious"
sign-mask generalized from AdamW to *any* sign-based optimizer. After
computing `update = sign(c)`, zero out the components whose sign
disagrees with the current gradient (`update * g <= 0`), then rescale
the surviving update by `1 / mask.mean().clamp(min=0.1)` to keep the
effective LR constant. Skips the step where momentum and gradient
disagree on sign. `clamp(min=0.1)` is pinned in the idea's mechanism
section — not deferred to the implementer.

Default off (cautious=False) → bit-identical to vanilla Lion. The mask
computation runs only when the flag is on.
"""
import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """Lion — sign-based optimizer with optional Cautious sign-mask.

    Parameters
    ----------
    params : iterable
    lr : float — Lion's step size (paper: ~10× smaller than AdamW)
    betas : (β1, β2) — β1 weights the gradient vs momentum for the
        sign-update; β2 is the momentum EMA on the gradient.
    weight_decay : float — decoupled (AdamW style), applied after the
        sign step.
    cautious : bool — if True, apply the Liang et al. (2024) sign-mask
        and `1 / mask.mean().clamp(min=0.1)` rescale to the sign-update.
        Default False — bit-identical to vanilla Lion.
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.98),
                 weight_decay=0.0, cautious=False):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                        cautious=bool(cautious))
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
            weight_decay = group["weight_decay"]
            cautious = group["cautious"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                m = state["momentum_buffer"]
                # Lion's update uses the gradient float for the sign
                # interpolation. Match the Muon path's `g.float()` so
                # mixed-precision params don't produce a bf16 momentum
                # buffer (Lion's EMA is small but persistent).
                g = grad.float()
                m_fp = m.float() if m.dtype != torch.float32 else m

                # Interpolation c = β1·m + (1-β1)·g
                c = m_fp.mul(beta1).add_(g, alpha=1 - beta1)
                update = c.sign()

                if cautious:
                    # Cautious mask: keep components whose sign agrees
                    # with the current gradient. Rescale by the mean
                    # survival rate (clamped to 0.1) so the effective
                    # step magnitude stays close to lr.
                    mask = (update * g > 0).to(update.dtype)
                    mask_mean = mask.mean().clamp(min=0.1)
                    update = update * mask / mask_mean

                # Cast back to param dtype before the in-place step
                update = update.to(p.dtype)

                # Decoupled weight decay (AdamW style): scale param
                # toward 0 in the same step as the update. p ← p - lr·(u + wd·p)
                # is equivalent to p ← (1 - lr·wd)·p - lr·u when wd is
                # treated as decoupled — see schedule_free_adamw.py
                # for the same pattern.
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                p.add_(update, alpha=-lr)

                # m ← β2·m + (1-β2)·g (Lion updates m AFTER the weight step)
                m_fp.mul_(beta2).add_(g, alpha=1 - beta2)
                if m_fp is not m:
                    m.copy_(m_fp)

        return loss
