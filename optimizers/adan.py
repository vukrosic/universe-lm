"""Adan: Adaptive Nesterov Momentum with N-Step Lookback
(Xie, Zhou, Lin, Li, Yan, Wang, Wang, arXiv:2208.06677,
TPAMI 2022 / ICLR 2023 workshop).

Adam-style optimizer that combines (1) a 1-step first moment, (2) an
N-step lookback variance estimate, and (3) a Nesterov-style
extrapolated gradient. Per-parameter update (paper Algorithm 1):

    g_t        = current gradient
    g_lookahead = g_t + β_la · (g_t − g_{t−1})     (Nesterov-style)
    m_t         = β1 · m_{t−1} + (1 − β1) · g_lookahead
    v_t         = β2 · v_{t−1} + (1 − β2) · mean_{i=0..N-1}(g_{t−i}²)
    update      = m_t / (√v_t + ε)
    w          ← w − lr · update                       (+ decoupled WD)

Where `N` is the *lookback window* (paper default `N = 4`).

The intuition: Adam's `v_t` only sees `g_t` at step `t`, which makes
the variance estimate *noisy* when the gradient oscillates. Adan's
N-step lookback smooths `v_t` over the last N gradients, giving a
more stable second moment.

Cold-start identity at step 0: `m_0 = 0, v_0 = 0, prev_grad = None,
grad_queue = []`. On the first step:
  - `g_lookahead = g_0` (no `prev_grad` to extrapolate from).
  - `m_1 = (1 − β1) · g_0`.
  - The queue receives `g_0²` and the variance term becomes
    `mean([g_0²]) = g_0²`, so `v_1 = (1 − β2) · g_0²`.
  - `update_0 = m_1 / (√v_1 + ε) ≈ g_0 / (|g_0| + ε) ≈ sign(g_0)`.

NOT bit-identical to AdamW's first step (which uses bias-corrected
Adam normalization `m̂/√v̂`), but the magnitudes are similar and the
`O(1/N)` deviation in `v_t` over the first N steps is the lever's
signature, not a bug. The N=4 lookback ramps in over the first 4
steps.

With `N = 0` Adan collapses to Nesterov-SGD (no `v` denominator —
the variance term is the mean of an empty queue, defined as zero,
so the update is `m / (√0 + ε) = m / ε`, which is the SGD
direction; in practice the queue always has at least one element
once we've appended `g_t`). With `β_la = 0` the Nesterov lookahead
collapses to plain Adam-style `m` accumulation (no extrapolation).
With `use_adan=False` (default), this class is never instantiated —
the trainer uses `torch.optim.AdamW` unchanged.
"""
import torch
from torch.optim.optimizer import Optimizer


class Adan(Optimizer):
    """Adan — Adaptive Nesterov Momentum with N-Step Lookback.

    Drop-in replacement for the AdamW 1-D / embedding / norm / head
    path. The 2-D Muon path is unchanged (Adan is an AdamW
    replacement, like 114-MARS, 119-SAM, 120-DAdapt, 121-Prodigy,
    123-CAME, 124-RAdam, 126-AdaShift, 127-GC, 128-SD).

    Parameters
    ----------
    params : iterable
    lr : float — Adan step size (paper default ≈ AdamW LR).
    betas : (β1, β2) — β1 weights the gradient vs momentum; β2 is
        the EMA decay on the N-step lookback variance. Paper
        defaults (0.9, 0.999).
    eps : float — additive to √v in the denominator (sign-stability
        + zero-guard).
    weight_decay : float — decoupled (AdamW style), applied before
        the ratio step.
    lookahead_beta : float — Nesterov-style extrapolation coefficient
        on the gradient difference `g_t − g_{t−1}`. Paper default
        0.5. At `lookahead_beta=0` the lookahead collapses to a
        plain first-moment EMA.
    n_lookback : int — gradient lookback window for the variance
        estimate (paper default 4). At `n_lookback=0` the variance
        term is undefined (the mean of an empty queue); at
        `n_lookback=1` Adan reduces to the Adam `g_t²` form.
    """

    def __init__(self, params, lr=0.006, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0, lookahead_beta=0.5,
                 n_lookback=4):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 < betas[0] < 1.0 and 0.0 < betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if n_lookback < 0:
            raise ValueError(f"Invalid n_lookback: {n_lookback}")
        if not (0.0 <= lookahead_beta <= 1.0):
            raise ValueError(f"Invalid lookahead_beta: {lookahead_beta}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        lookahead_beta=float(lookahead_beta),
                        n_lookback=int(n_lookback))
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
            lookahead_beta = group["lookahead_beta"]
            n_lookback = group["n_lookback"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adan does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    # g_{t-1} for the Nesterov-style extrapolation.
                    # None on the first step — the lookahead term
                    # falls back to the current gradient.
                    state["prev_grad"] = None
                    # Last N gradients for the variance lookback
                    # (clones, fp32, length bounded by n_lookback).
                    state["grad_queue"] = []

                state["step"] += 1

                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                grad_queue = state["grad_queue"]
                prev_grad = state["prev_grad"]

                # Match the Muon path's `g.float()` so mixed-precision
                # params don't produce a bf16 EMA buffer.
                g = grad.float()
                m_fp = m.float() if m.dtype != torch.float32 else m
                v_fp = v.float() if v.dtype != torch.float32 else v

                # Nesterov-style extrapolated gradient:
                #   g_la = g_t + β_la · (g_t − g_{t−1})
                # On the first step `prev_grad` is None ⇒ fall back
                # to the bare gradient (the lookahead term is 0).
                if prev_grad is not None and lookahead_beta != 0.0:
                    diff = g - prev_grad
                    g_la = g.add(diff, alpha=lookahead_beta)
                else:
                    g_la = g

                # m_t = β1·m + (1−β1)·g_la
                m_fp.mul_(beta1).add_(g_la, alpha=1 - beta1)

                # N-step lookback variance.
                # Append g² to the queue, then take the mean of the
                # last N entries' squares. The queue is bounded to
                # length n_lookback.
                if n_lookback > 0:
                    grad_queue.append(g.pow(2).clone())
                    if len(grad_queue) > n_lookback:
                        grad_queue.pop(0)
                    # mean of the last min(N, t) entries. The
                    # `stack` over the queue is O(N·|p|) which is
                    # cheap at tiny1m3m (N=4, |p| ≤ 49152 vocab).
                    v_lookback = torch.stack(grad_queue, dim=0).mean(dim=0)
                    # v_t = β2·v + (1−β2)·v_lookback
                    v_fp.mul_(beta2).add_(v_lookback, alpha=1 - beta2)
                # else: n_lookback=0 ⇒ skip variance update (Nesterov-
                # SGD regime). The denominator is `√0 + ε = ε`, so
                # `update = m / ε` is a numerically-amplified step.

                # Save prev_grad for the *next* step's Nesterov term.
                state["prev_grad"] = g.clone()

                # update = m / (√v + ε). No bias correction (paper
                # Algorithm 1 does not include it).
                denom = v_fp.sqrt().add_(eps)
                update = m_fp.div(denom)

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
