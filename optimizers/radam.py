"""RAdam: Rectified Adam — Variance-Aware Adaptive Learning Rate
(Liu, Jiang, He, Chen, Liu, Gao, Han, "On the Variance of the
Adaptive Learning Rate and Beyond", arXiv:1908.03265, ICLR 2020).

Adam's update is `update = m̂_t / (√v̂_t + ε)` with the standard
bias-correction `1 / (1 − β2^t)` on the denominator. For early
`t` (e.g. `t = 10, β2 = 0.999` ⇒ `1 − β2^t = 0.01`), this correction
amplifies the update dramatically — the standard fix is a manual
warmup `lr ← lr · min(1, t / warmup_steps)`.

RAdam derives a *closed-form* correction that accounts for the
variance of `1 / (1 − β2^t)`. Let

    ρ_∞ = 2 / (1 − β2) − 1
    sma_length(t) = ρ_∞ − 4·(t − 2)·(t − 1)·(1 − β2)² /
                              (t · (1 + β2)² · (1 − β2^t))
    ρ_t          = max(0, sma_length(t)) / (1 − β2^t)

(Liu et al. 2019 §3.2 Eq (4)). Then:

    if ρ_t > 4: update = m̂_t · √(ρ_t) / (√v̂_t + ε)  (variance-bounded)
    else:        update = m̂_t                        (SGD-only fallback)

When `t` is small the variance of `1/(1−β2^t)` is high ⇒ RAdam
falls back to the SGD-style `m̂_t` step (no `v̂` denominator). When
`t` is large enough for the variance to settle (≈ `t > 4 / (1−β2)`),
RAdam switches to the full Adam-normalized update with a
variance-aware rescale. This *removes the need for a manual warmup*
— RAdam auto-detects when the effective LR is safe.

Identity at step 0: at `t = 1` the denominator is `1 − β2 ≈ 0.001`
so `ρ_1 ≪ 4` ⇒ RAdam takes the *SGD-fallback* path:
`update = m̂_1 = (1 − β1) · g_0`. This is NOT bit-identical to AdamW's
first step (which uses the full `m̂ / √v̂`), but the magnitude is
comparable. The first-step divergence is the lever, not a bug.
With `use_radam=False` (default) this class is never instantiated —
the trainer uses `torch.optim.AdamW` unchanged, baseline path
bit-identical.
"""
import math
import torch
from torch.optim.optimizer import Optimizer


class RAdam(Optimizer):
    """RAdam — Rectified Adam (Liu et al. 2019).

    Drop-in replacement for the AdamW 1-D / embedding / norm / head
    path. The 2-D Muon path is unchanged (RAdam lives only on the
    AdamW bucket, like 114-MARS, 119-SAM, 120-DAdapt, 121-Prodigy,
    123-CAME).

    Parameters
    ----------
    params : iterable
    lr : float — RAdam step size (paper default ≈ AdamW LR; same
        scale works because the early-step SGD-fallback is
        self-stabilizing).
    betas : (β1, β2) — β1 weights the gradient vs momentum; β2 is
        the EMA decay on `g²`. Paper defaults (0.9, 0.999).
    eps : float — additive to √v in the Adam path (sign-stability +
        zero-guard). Unused on the SGD-fallback branch.
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

            # RAdam precomputes ρ_∞ once per group (it's a function of β2
            # only) and reuses it across every param at every step.
            rho_inf = 2.0 / (1.0 - beta2) - 1.0
            one_minus_beta2_sq = (1.0 - beta2) * (1.0 - beta2)
            one_plus_beta2_sq = (1.0 + beta2) * (1.0 + beta2)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step_t = state["step"]

                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                # Match the Muon path's `g.float()` so mixed-precision
                # params don't produce a bf16 EMA buffer.
                g = grad.float()
                m_fp = m.float() if m.dtype != torch.float32 else m
                v_fp = v.float() if v.dtype != torch.float32 else v

                # m ← β1·m + (1−β1)·g       (momentum EMA on g)
                m_fp.mul_(beta1).add_(g, alpha=1 - beta1)
                # v ← β2·v + (1−β2)·g²      (Adam 2nd moment on g²)
                v_fp.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # m̂_t = m_t / (1 − β1^t)
                bias_correction1 = 1.0 - beta1 ** step_t
                m_hat = m_fp.div(bias_correction1)

                # ρ_t from Liu et al. 2019 §3.2.
                #   β2^t             = β2 ** step_t
                #   1 − β2^t        = 1 − β2_pow_t
                #   sma_length(t)    = ρ_∞
                #                    − 4·(t − 2)·(t − 1)·(1 − β2)²
                #                      / (t · (1 + β2)² · (1 − β2^t))
                #   ρ_t              = max(0, sma_length(t)) / (1 − β2^t)
                # When `ρ_t > 4` use the variance-bounded Adam step
                # `update = m̂_t · √(ρ_t) / (√v̂_t + ε)`. Otherwise the
                # SGD-only fallback `update = m̂_t` (no `v̂_t` denom).
                beta2_pow_t = beta2 ** step_t
                one_minus_beta2_pow_t = 1.0 - beta2_pow_t
                use_adam_path = False
                rho_t = 0.0
                if step_t > 2 and one_minus_beta2_pow_t > 0.0:
                    sma_length = (
                        rho_inf
                        - 4.0 * (step_t - 2.0) * (step_t - 1.0)
                          * one_minus_beta2_sq
                        / (step_t * one_plus_beta2_sq * one_minus_beta2_pow_t)
                    )
                    rho_t = max(0.0, sma_length / one_minus_beta2_pow_t)
                    use_adam_path = rho_t > 4.0

                if use_adam_path:
                    # Variance-bounded Adam: update = m̂_t · √(ρ_t) / (√v̂_t + ε)
                    bias_correction2 = 1.0 - beta2_pow_t
                    v_hat = v_fp.div(bias_correction2) if bias_correction2 > 0 else v_fp
                    denom = v_hat.sqrt().add_(eps)
                    update = m_hat.mul(math.sqrt(rho_t)).div_(denom)
                else:
                    # SGD-only fallback: update = m̂_t
                    # (no `v̂_t` denominator; this is the RAdam "early-step"
                    # behavior when the variance of 1/(1−β2^t) is high).
                    update = m_hat

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