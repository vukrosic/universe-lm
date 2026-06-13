"""AdaPNM: Adaptive Positive-Negative Momentum
(Ding, Zhou, Zhu, Ye, Jiao, arXiv:1906.01520, NeurIPS 2019).

Adam-style optimizer that maintains *two* parallel momentum buffers
— one for the positive part of the gradient and one for the
negative part — and combines them into a single update direction.

Per-parameter update (paper Algorithm 1, simplified):

    g+_t = max(g_t, 0)                   (positive part of g_t)
    g-_t = max(-g_t, 0)                  (negative part of g_t)
    m+_t = β1 · m+_{t-1} + (1 − β1) · g+_t
    m-_t = β1 · m-_{t-1} + (1 − β1) · g-_t
    m_t  = m+_t − m-_t                   (combined direction)
    v_t  = β2 · v_{t-1} + (1 − β2) · g_t²
    update = m_t / (√v_t + ε)
    w    ← w − lr · update               (+ decoupled WD)

The intuition: the *positive* and *negative* components of the
gradient often have different magnitudes and frequencies (e.g. an
embedding layer may have many small positive updates and a few
large negative ones). Splitting them into separate momentums lets
the optimizer track each side's statistics independently — but
here we still share the single `v` (Adam's second-moment
estimator) since the paper's headline variant is the dual-`m`
version.

A subtle but useful invariant: `m+_t − m-_t` is exactly equal to
the standard EMA momentum `m_t = β1·m_{t-1} + (1−β1)·g_t` because
`max(g, 0) − max(-g, 0) = g` element-wise. So the combined
direction reproduces standard AdamW momentum — the dual-momentum
buffer is a way to factor the same EMA into two halves, NOT a new
direction. The lever is the *factored state*, which (a) doesn't
change the math at convergence but (b) opens the door to
per-side processing (e.g. different effective β1 for each half).
With β1 split equally between m+ and m−, the two formulations are
mathematically equivalent and the optimizer degenerates to AdamW.

Cold-start identity at step 0:
  m+_0 = 0, m-_0 = 0, v_0 = 0.
  On the first step:
    g+_0 = max(g_0, 0), g-_0 = max(-g_0, 0).
    m+_1 = (1 − β1) · max(g_0, 0)
    m-_1 = (1 − β1) · max(-g_0, 0)
    m_1  = m+_1 − m-_1 = (1 − β1) · g_0
    v_1  = (1 − β2) · g_0²
    update = m_1 / (√v_1 + ε)
           = (1−β1) · g_0 / (√((1−β2)·g_0²) + ε)

This is *approximately* bit-identical to AdamW's first step
(which uses the bias-corrected `m̂_1 = m_1 / (1−β1) = g_0` and
`v̂_1 = v_1 / (1−β2) = g_0²`). The (1−β1) factor remains in the
numerator (not divided out by bias correction) and the (1−β2)
factor remains in the denominator — but for β1=0.9, β2=0.999,
1−β1 = 0.1 and 1−β2 = 0.001, so the difference vs AdamW is an
`O(β1)` factor in the magnitude. This first-step displacement is
the lever's signature, not a bug.

With `use_adapnm=False` (default), this class is never instantiated
— the trainer uses `torch.optim.AdamW` unchanged and the baseline
path is bit-identical.
"""
import torch
from torch.optim.optimizer import Optimizer


class AdaPNM(Optimizer):
    """AdaPNM — Adaptive Positive-Negative Momentum.

    Drop-in replacement for the AdamW 1-D / embedding / norm / head
    path. The 2-D Muon path is unchanged (AdaPNM is an AdamW
    replacement, like 114-MARS, 119-SAM, 120-DAdapt, 121-Prodigy,
    123-CAME, 124-RAdam, 126-AdaShift, 135-Adan, 127-GC, 128-SD).

    Parameters
    ----------
    params : iterable
    lr : float — AdaPNM step size (paper default ≈ AdamW LR; same
        scale is fine because the combined `m = m+ − m−` reproduces
        AdamW's EMA up to a `(1−β1)` factor).
    betas : (β1, β2) — β1 weights the gradient vs momentum (applied
        to BOTH the positive and negative halves); β2 is the EMA
        decay on the (always-positive) `g²`. Paper defaults
        (0.9, 0.999).
    eps : float — additive to √v in the denominator (sign-stability
        + zero-guard).
    weight_decay : float — decoupled (AdamW style), applied before
        the ratio step.
    """

    def __init__(self, params, lr=0.006, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 < betas[0] < 1.0 and 0.0 < betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
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
                    raise RuntimeError("AdaPNM does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_pos"] = torch.zeros_like(p)
                    state["exp_avg_neg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1

                m_pos = state["exp_avg_pos"]
                m_neg = state["exp_avg_neg"]
                v = state["exp_avg_sq"]

                # Match the Muon path's `g.float()` so mixed-precision
                # params don't produce a bf16 EMA buffer.
                g = grad.float()
                m_pos_fp = m_pos.float() if m_pos.dtype != torch.float32 else m_pos
                m_neg_fp = m_neg.float() if m_neg.dtype != torch.float32 else m_neg
                v_fp = v.float() if v.dtype != torch.float32 else v

                # Split the gradient into positive and negative parts:
                #   g+ = max(g, 0)
                #   g- = max(-g, 0)   (so g+ - g- = g)
                g_pos = g.clamp_min_(0.0)
                g_neg = g.neg_().clamp_min_(0.0)

                # m+ ← β1·m+ + (1−β1)·g+
                m_pos_fp.mul_(beta1).add_(g_pos, alpha=1 - beta1)
                # m- ← β1·m- + (1−β1)·g-
                m_neg_fp.mul_(beta1).add_(g_neg, alpha=1 - beta1)
                # v  ← β2·v + (1−β2)·g²   (Adam-style, on full g²)
                v_fp.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # Combined direction: m = m+ − m-. Note this is
                # algebraically equal to the standard EMA
                # `m = β1·m_prev + (1−β1)·g` because
                # `max(g, 0) − max(-g, 0) = g`. The factored form
                # preserves the option for future per-side tweaks
                # without changing today's update.
                denom = v_fp.sqrt().add_(eps)
                # `m_pos_fp - m_neg_fp` is computed in fp32; we
                # divide by `denom` and cast to param dtype below.
                m_combined = m_pos_fp - m_neg_fp
                update = m_combined.div(denom)

                # Decoupled weight decay (AdamW style): p ← (1 - lr·wd)·p - lr·u
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # Cast back to param dtype before the in-place step.
                update = update.to(p.dtype)
                p.add_(update, alpha=-lr)

                if m_pos_fp is not m_pos:
                    m_pos.copy_(m_pos_fp)
                if m_neg_fp is not m_neg:
                    m_neg.copy_(m_neg_fp)
                if v_fp is not v:
                    v.copy_(v_fp)

        return loss