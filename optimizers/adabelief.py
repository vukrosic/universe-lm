"""AdaBelief: Adapting Stepsizes by the Belief in Observed Gradients
(Zhuang, Liu, Tran, Hoang, Chang, et al. 2020, arXiv:2010.07468,
NeurIPS 2020).

AdamW's 2nd moment is `v = E[g²]`. AdaBelief replaces it with the
variance of the *gradient residual* around its running mean —
i.e. how much the current gradient disagrees with the momentum:

    m_t = β1 · m_{t-1} + (1 − β1) · g_t                (1st moment)
    s_t = β2 · s_{t-1} + (1 − β2) · (g_t − m_t)² + ε   (residual 2nd moment)
    m̂_t = m_t / (1 − β1^t)                             (bias correction)
    ŝ_t = s_t / (1 − β2^t)                             (bias correction)
    update = m̂_t / (√ŝ_t + ε)
    w ← w − lr · (update + λ · w)                      (decoupled WD)

Key intuition: when `g_t` agrees with the running momentum `m_t` (small
residual), `s` is small ⇒ step is large (we trust the direction).
When `g_t` disagrees with `m_t` (large residual), `s` is large ⇒ step
is small. AdamW does the *opposite*: large `g²` makes `v` large,
*shrinking* the step — which is wrong when a large gradient is a
*good* direction, not a noisy one.

Identity at step 0: `m_0 = 0`, `s_0 = ε`. With `g_0` as the first
gradient,
    m_1 = (1 − β1) · g_0,
    s_1 = (1 − β2) · (g_0 − m_1)² + ε
        = (1 − β2) · (g_0 − (1−β1)·g_0)² + ε
        = (1 − β2) · β1² · g_0² + ε.
With β1 = 0.9, β2 = 0.999, this gives `s_1 ≈ 0.081·g_0² + ε`. The
first-step update is `m̂_1 / √ŝ_1 ≈ g_0 / √(0.081·g_0² + ε)
≈ (1/√0.081)·sign(g_0) ≈ 3.5·sign(g_0)`. This is NOT bit-identical
to AdamW's first step (AdamW would take `m̂_1/√v̂_1` with
`v̂_1 = g_0²`), but the magnitude is the same order as the Adam
first-step scale. The first-step displacement is the lever's
signature, not a bug. The forward graph is unchanged, so the
*pre-step-0* forward output is bit-identical to baseline.

With `use_adabelief=False` (default) this class is never instantiated
— the trainer uses `torch.optim.AdamW` unchanged, baseline path
bit-identical.
"""
import torch
from torch.optim.optimizer import Optimizer


class AdaBelief(Optimizer):
    """AdaBelief — Adam with a residual-variance denominator.

    Drop-in replacement for the AdamW 1-D / embedding / norm / head
    path. The 2-D Muon path is unchanged (AdaBelief lives only on
    the AdamW bucket, like 114-MARS, 119-SAM, 120-DAdapt, 121-Prodigy,
    123-CAME, 124-RAdam, 126-AdaShift, 127-GC, 128-SD, 135-Adan,
    136-AdaPNM, 137-AdamP).

    Parameters
    ----------
    params : iterable
    lr : float — AdaBelief step size (paper default matches AdamW LR;
        same scale is fine because the denominator is `√(residual²)`,
        bounded like AdamW's `√g²`).
    betas : (β1, β2) — β1 weights the gradient vs momentum; β2 is the
        EMA decay on the residual 2nd moment `(g_t − m_t)²`. Paper
        defaults (0.9, 0.999).
    eps : float — additive inside the residual EMA buffer (so the
        denominator is never exactly zero) and to `√ŝ` for
        sign-stability. Paper uses `1e-8` for the outer eps and
        `1e-16` (or 0) for the inner ε — we use a single `eps` for
        both to match the project convention; the lever's signature
        is unchanged.
    weight_decay : float — decoupled (AdamW style), applied before
        the ratio step.
    """

    def __init__(self, params, lr=0.006, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 < betas[0] < 1.0 and 0.0 < betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
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
                    raise RuntimeError("AdaBelief does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step_t = state["step"]

                m = state["exp_avg"]
                s = state["exp_avg_sq"]
                # Match the Muon path's `g.float()` so mixed-precision
                # params don't produce a bf16 EMA buffer.
                g = grad.float()
                m_fp = m.float() if m.dtype != torch.float32 else m
                s_fp = s.float() if s.dtype != torch.float32 else s

                # m ← β1·m + (1−β1)·g       (momentum EMA on g)
                m_fp.mul_(beta1).add_(g, alpha=1 - beta1)
                # s ← β2·s + (1−β2)·(g − m)² + ε
                # The `(g − m)²` is the *residual* — how much the
                # current gradient disagrees with the running
                # momentum. AdamW would use `g²` here.
                residual = g.sub(m_fp)
                s_fp.mul_(beta2).addcmul_(residual, residual, value=1 - beta2).add_(eps)

                # Bias-corrected first moment and residual 2nd moment
                # (paper §3.1, Algorithm 1). At step 1 this maps
                # `m_1 = (1−β1)·g_0` → `m̂_1 = g_0` and the residual
                # becomes `g_0 − g_0 = 0` ⇒ `s_1 = ε` ⇒ `update ≈
                # g_0 / √(2ε)`. The lever's first-step signature,
                # not a bug.
                bc1 = 1 - beta1 ** step_t
                bc2 = 1 - beta2 ** step_t
                m_hat = m_fp.div(bc1)
                s_hat = s_fp.div(bc2)

                # update = m̂ / (√ŝ + ε)
                denom = s_hat.sqrt().add_(eps)
                update = m_hat.div_(denom)

                # Decoupled weight decay (AdamW style): p ← (1 - lr·wd)·p - lr·u
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Cast back to param dtype before the in-place step
                update = update.to(p.dtype)
                p.add_(update, alpha=-lr)

                if m_fp is not m:
                    m.copy_(m_fp)
                if s_fp is not s:
                    s.copy_(s_fp)

        return loss
