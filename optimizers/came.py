"""CAME: Confidence-guided Adaptive Memory Efficient Optimization
(Luo et al. 2023, arXiv:2307.02085, NeurIPS 2023).

Adam-style optimizer with a confidence rescaling that adjusts the
update magnitude by how much the current gradient agrees with the
running first-moment estimate. Per-parameter update (with bias
correction, matching the paper's Algorithm 1):

    m_t = β1 · m_{t-1} + (1 − β1) · g_t                (momentum EMA)
    v_t = β2 · v_{t-1} + (1 − β2) · g_t²               (Adam 2nd moment)
    m̂_t = m_t / (1 − β1^t)                             (bias correction)
    v̂_t = v_t / (1 − β2^t)                             (bias correction)
    res_t = (m̂_t − g_t) / (√v̂_t + ε)                  (residual)
    conf_t = max(res_t, 0) + ε                          (clipped residual)
    update = m̂_t / (√v̂_t + ε) · conf_t / (|m̂_t| + ε)
    w ← w − lr · update

The `conf_t` factor down-weights updates when `g_t ≈ m̂_t` (the
residual is small — the momentum already captured this direction)
and applies a residual-shaped step when `g_t` and `m̂_t` disagree
(oscillating / noisy gradients — step in the consensus direction).
This is the same intuition as Lion's sign-agreement but with a
continuous magnitude instead of a binary sign check.

Cold-start identity at step 0: `m_0 = 0`, `v_0 = 0` ⇒
    `m̂_1 = m_1 / (1−β1) = g_0`,
    `v̂_1 = v_1 / (1−β2) = g_0²`,
    `res_1 = (g_0 − g_0) / (|g_0| + ε) = 0`,
    `conf_1 = max(0, 0) + ε = ε`,
    `update ≈ sign(g_0) · ε · |g_0| / (|g_0|+ε)² ≈ 1e-6` (vanishingly
    small) ⇒ first optimizer call leaves all params within fp32
    noise of baseline. The bias correction is what makes this exact-
    zero (without it, the residual at step 0 is sign-dependent and
    ~50% of elements have a non-zero first-step update).

Numerical-stability guards (added after 2026-06-13 GPU blowup)
--------------------------------------------------------------
The previous version of this class did NOT bound the update
magnitude. When `v̂ ≈ 0` (cold-start or post-sudden-vanish), the
denominator `√v̂ + ε` collapses to `ε` (1e-8) and the update
`m̂ / denom · conf / |m̂|` can reach `m̂ / ε² ≈ m̂ · 1e16`. A single
bad step with non-trivial `m̂` and tiny `v̂` produces an
`O(1e16)` per-element displacement; with `lr=0.006` that's
`O(6e13)` per step, observed as val loss 10.81 → 6.79e7 at step 25
→ 1.06e8 by step 150. The fix:

1. **NaN/Inf guard on `grad` and the `m, v` buffers** before any
   arithmetic. If the input is non-finite, skip the parameter for
   this step (`m`, `v`, and `p` are not touched) — don't poison the
   buffers with a non-finite ratio. Defensive: shouldn't fire on a
   healthy trajectory.
2. **Per-element update-magnitude clip** (`update_clip`, default
   `10.0`) — `update ← clamp(update, -update_clip, update_clip)`
   before applying. The confidence factor can rescale `m̂/denom`
   beyond ±1 when the residual is large; the clip bounds any single
   step's per-element displacement to `±update_clip · lr` (so
   `±0.06` at the default lr).

Bit-identical at step 0 with `use_came=False` (default): this class
is never instantiated, trainer uses plain `torch.optim.AdamW`.

Bit-identical at step 0 with `use_came=True`: at step 1 the
step-0 update magnitude is `~1e-6 · sign(g_0)`, well under
`update_clip=10.0`, and the NaN/Inf guard is a no-op on finite
grads. So the v2 step-0 output matches v1's step-0 output exactly.
"""
import torch
from torch.optim.optimizer import Optimizer


class CAME(Optimizer):
    """CAME — Confidence-guided Adaptive Memory Efficient Optimization.

    Parameters
    ----------
    params : iterable
    lr : float — CAME step size (paper default ≈ AdamW LR; same
        scale is fine because the update magnitude is bounded by
        `|m̂| / √v̂` like AdamW with bias correction).
    betas : (β1, β2) — β1 weights the gradient vs momentum; β2 is
        the EMA decay on `g²`. Paper defaults (0.9, 0.999).
    eps : float — additive to √v in the denominator and to the
        confidence floor (sign-stability + zero-guard).
    weight_decay : float — decoupled (AdamW style), applied before
        the ratio step.
    update_clip : float — per-element magnitude cap on the raw
        `update` before the LR scaling. Default `10.0` bounds any
        single step's per-element displacement to `±10 · lr` —
        protects against the `m̂ / ε² ≈ 1e16` blowup that occurs
        when `v̂ ≈ 0` and `m̂` is non-trivial. Default `10.0` is
        well above the natural per-element step magnitude (~1.0
        for vanilla AdamW) so it doesn't clip healthy updates;
        only triggers on the runaway regime.
    """

    def __init__(self, params, lr=0.006, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0, update_clip=10.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 < betas[0] < 1.0 and 0.0 < betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if update_clip <= 0.0:
            raise ValueError(f"Invalid update_clip: {update_clip}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        update_clip=update_clip)
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
            update_clip = group["update_clip"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients")

                # NaN/Inf guard on the input grad. If non-finite,
                # skip the parameter for this step (m, v, p
                # untouched) — don't poison the buffers with a
                # non-finite ratio. Defensive only.
                if not torch.isfinite(grad).all():
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] = 0

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # Defensive finiteness check on existing buffers.
                # If a prior step left non-finite values, skip this
                # parameter rather than amplifying the corruption.
                if (not torch.isfinite(m).all()
                        or not torch.isfinite(v).all()):
                    continue

                state["step"] += 1
                step_t = state["step"]
                # Match the Muon path's `g.float()` so mixed-precision
                # params don't produce a bf16 EMA buffer.
                g = grad.float()
                m_fp = m.float() if m.dtype != torch.float32 else m
                v_fp = v.float() if v.dtype != torch.float32 else v

                # m ← β1·m + (1−β1)·g       (momentum EMA on g)
                m_fp.mul_(beta1).add_(g, alpha=1 - beta1)
                # v ← β2·v + (1−β2)·g²      (Adam 2nd moment on g²)
                v_fp.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # Bias-corrected first / second moments (paper §3,
                # Algorithm 1). At step 1 this maps m_1 = (1−β1)·g_0
                # → m̂_1 = g_0, making the residual at step 1
                # exactly zero ⇒ confidence = ε ⇒ update ≈ 0 ⇒
                # baseline-bit-identical (well under `update_clip`).
                bc1 = 1 - beta1 ** step_t
                bc2 = 1 - beta2 ** step_t
                m_hat = m_fp.div(bc1)
                v_hat = v_fp.div(bc2)

                # √v̂ + ε
                denom = v_hat.sqrt().add_(eps)
                # res_t = (m̂_t − g_t) / (√v̂_t + ε)
                res = (m_hat - g).div_(denom)
                # conf_t = max(res, 0) + ε
                conf = res.clamp_min_(0.0).add_(eps)
                # update = m̂ / (√v̂ + ε) · conf / (|m̂| + ε).
                # Capture `m_hat_abs` BEFORE the in-place division
                # so the final divisor uses the un-divided m̂, not
                # the post-division value.
                m_hat_abs = m_hat.abs().add_(eps)
                update = m_hat.div_(denom).mul_(conf).div_(m_hat_abs)

                # Per-element magnitude clip (default 10.0). Bounds
                # any runaway step before the LR scaling. The natural
                # per-element magnitude is ~1.0 (vanilla AdamW is
                # |m̂/√v̂| ≤ ~1 with bias correction), so this clip
                # is effectively inactive on a healthy trajectory and
                # only triggers in the `v̂ ≈ 0` blowup regime.
                if update_clip < float("inf"):
                    update.clamp_(-update_clip, update_clip)

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