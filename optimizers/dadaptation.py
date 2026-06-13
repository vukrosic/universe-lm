"""D-Adaptation for AdamW (Defazio 2023, arXiv:2301.11933 / arXiv:2201.11941).

Removes the learning-rate knob from training by maintaining a log-scale
running lower bound `D` on the distance from `w_init` to `w_optimal`.
The effective LR at step `t` is derived as `lr_t = D_t / ‖g_t‖` where
`g_t` is the AdamW-processed gradient. The 1st / 2nd moments of
AdamW are retained intact — only the *outer* LR scaling is replaced.

Mechanism
---------
For each param group:
  m_t      = exp_avg from the previous step (AdamW 1st moment)
  c_+      = ⟨sign(g_t),  sign(m_t)⟩ / N      # agreement ratio
  c_-      = ⟨sign(g_t), −sign(m_t)⟩ / N      # disagreement ratio
  D_t      ← D_t · exp(d0_lr · (c_+ − c_-))   # log-LR update
  lr_t     = D_t / ‖g_t‖                       # effective per-step LR
  (then standard AdamW m/v/bias-correction/decoupled-WD with lr_t)

Identity at step 0
------------------
`D_0 = 1e-6` warm-start. At step 0 the parent AdamW hasn't allocated
`exp_avg` yet ⇒ agreement signal undefined ⇒ `D` stays at `d_init`.
The first-step LR is `lr_0 = D_0 / ‖g_0‖ ≈ 1e-6 / ‖g_0‖` (essentially
zero). After ~10–20 steps `D` reaches a typical AdamW-equivalent
value (~0.001–0.01); the first ~10 steps see a *ramp-up* in effective
LR — this is the lever's signature, not a bug (the design sketch in
`autoresearch/ideas/120-dadaptation/idea.md` explicitly accepts this).

When `use_dadapt=False` (default), this class is never instantiated —
the trainer uses `torch.optim.AdamW` unchanged. See
`training/trainer.py:setup_muon_optimizer` for the gate.

Numerical-stability guards (added after 2026-06-13 GPU blowup)
--------------------------------------------------------------
The previous version of this class did NOT bound `D` from above. With
`d0_lr=1.0` and consistent agreement (`c_+ - c_- ≈ 1.0`) `D` grows
as `e^t` per step — after ~92 steps `D ≈ e^92 ≈ 1e40`, and then
`lr_t = D / ‖g_t‖` explodes whenever ‖g_t‖ is small (post-warmup
plateau). At 0.94M / 92-step this produced val loss 10.81 → 36.89
(step 50, NaN-like spike) → 7.04e15 final. The fix:

1. **`d_max` upper clamp on `D`** (paper §3.1, default `1.0`): even
   when the agreement signal says `D` should grow, `D` cannot exceed
   `d_max`. Bounds the discovery loop into a stable band.
2. **`lr_t` magnitude clamp to `d_max`**: when ‖g_t‖ is small-but-
   non-zero, `D / ‖g_t‖` can still exceed `d_max`; final clamp
   enforces the paper's invariant.
3. **Gradient-norm floor (`grad_norm < 1e-12 → base lr`)**: prevents
   division blowup on degenerate / zero-grad steps.
4. **NaN/Inf guard on `D`-update and `lr_t`**: if the input gradient
   or momentum goes non-finite, hold `D` at its current value and
   fall back to the base lr for this step (don't poison the
   discovery loop with a non-finite ratio).

Bit-identical at step 0 with `use_dadapt=False` (default): this class
is never instantiated, trainer uses plain `torch.optim.AdamW`.
"""
import math
import torch
from torch.optim import AdamW


class DAdaptAdamW(AdamW):
    """AdamW with the D-Adaptation LR-discovery loop (Defazio 2023).

    Subclass of `torch.optim.AdamW`. On each `.step()`, the parent's
    AdamW math runs unchanged (m, v, bias-correction, decoupled WD);
    only the per-group `lr` is replaced with `D / ‖g_t‖` for the
    duration of the parent's `step()`, then restored.

    Parameters
    ----------
    params : iterable
    lr : float — the *base* LR constant (paper default 1.0). With
        D-Adapt this is NOT the actual LR used for the step — it is
        the `η` scaling constant in `D ← D · exp(η · (c_+ - c_-))`.
    betas : (β1, β2) — Adam
    eps : float — Adam denominator
    weight_decay : float — decoupled (AdamW style)
    d0_lr : float — `η`, the log-LR update constant. Paper default 1.0.
    d_init : float — initial value of `D` (warm-start). Paper default
        1e-6 ⇒ first-step lr is essentially zero.
    min_lr : float — lower clamp on `D` (prevents `D` from collapsing
        to zero on a sign-disagreement spike). Paper default 0.0.
    d_max : float — upper clamp on `D` (paper §3.1). Paper default
        `1.0`. Also caps the derived `lr_t = D / ‖g_t‖` to `d_max` as
        a final safety net against division-by-tiny-gradient spikes.
    """

    def __init__(self, params, lr=1.0, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, d0_lr=1.0, d_init=1e-6, min_lr=0.0,
                 d_max=1.0):
        if d0_lr <= 0.0:
            raise ValueError(f"D-Adapt d0_lr must be > 0, got {d0_lr}")
        if d_init <= 0.0:
            raise ValueError(f"D-Adapt d_init must be > 0, got {d_init}")
        if min_lr < 0.0:
            raise ValueError(f"D-Adapt min_lr must be >= 0, got {min_lr}")
        if d_max <= 0.0:
            raise ValueError(f"D-Adapt d_max must be > 0, got {d_max}")
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay)
        for group in self.param_groups:
            group["d0_lr"] = float(d0_lr)
            group["d_init"] = float(d_init)
            group["min_lr"] = float(min_lr)
            group["d_max"] = float(d_max)
            # Per-group scalar D (log-LR lower bound). Lazy-init to
            # `d_init` so the first step's lr_0 ≈ d_init / ‖g_0‖.
            group["D"] = float(d_init)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Pass 1: per group, compute agreement against the previous
        # step's exp_avg (= m_{t-1}), update D, derive effective
        # lr_t = D / ‖g_t‖. At step 1 exp_avg hasn't been allocated
        # by the parent yet ⇒ D stays at d_init ⇒ lr_t ≈ eps / ‖g‖.
        for group in self.param_groups:
            D = group["D"]
            d0_lr = group["d0_lr"]
            min_lr = group["min_lr"]
            d_max = group["d_max"]

            agree = 0.0
            disagree = 0.0
            total_n = 0
            grad_norm_sq = 0.0
            grad_nonfinite = False
            for p in group["params"]:
                if p.grad is None:
                    continue
                g_det = p.grad.detach()
                if not torch.isfinite(g_det).all():
                    # NaN/Inf in the gradient ⇒ don't trust D-update
                    # or the effective LR for this step. Use a safe
                    # fallback (base lr) so the parent's update is
                    # no-worse than vanilla AdamW; we don't make it
                    # worse by poisoning D.
                    grad_nonfinite = True
                    break
                grad_norm_sq += g_det.pow(2).sum().item()
                # `exp_avg` is the parent's lazy-init 1st-moment
                # buffer — it does not exist until the parent's first
                # `.step()` has run. Use it as the "have we completed
                # at least one step?" signal.
                if "exp_avg" not in self.state[p]:
                    continue
                g = p.grad
                m = self.state[p]["exp_avg"]
                if not torch.isfinite(m).all():
                    # Momentum went non-finite in a previous step —
                    # bail to safe fallback rather than poison D.
                    grad_nonfinite = True
                    break
                agree += (g.sign() == m.sign()).sum().item()
                disagree += (g.sign() != m.sign()).sum().item()
                total_n += g.numel()

            if grad_nonfinite:
                # Safe fallback: hold D at its current value, use the
                # base lr for this step. The parent's own math may
                # produce NaN/Inf, but we don't make it worse.
                group["_dadapt_lr"] = group["lr"]
                continue

            # Update D only if we have an agreement signal.
            if total_n > 0:
                c_plus = agree / total_n
                c_minus = disagree / total_n
                # Simplified paper form:
                #   D ← D · exp(η · (c_+ - c_-))
                D_new = D * math.exp(d0_lr * (c_plus - c_minus))
                # NaN/Inf guard on D-update itself (defensive —
                # shouldn't trigger with finite inputs).
                if not math.isfinite(D_new):
                    D_new = D
                # Clamp D into [min_lr, d_max]. Without `d_max`, with
                # `d0_lr=1.0` and consistent agreement `D` grows
                # unboundedly (~e^t per step) — at 0.94M / 92-step
                # this caused a numerical blowup (val loss 10.81 →
                # 36.89 at step 50 → 7.04e15 final). `d_max=1.0` is
                # the paper default and bounds the discovery loop.
                if D_new < min_lr:
                    D_new = min_lr
                if D_new > d_max:
                    D_new = d_max
                D = D_new
                group["D"] = D

            # Compute the effective per-step lr.
            grad_norm = math.sqrt(grad_norm_sq)
            # Floor the gradient norm: `lr_t = D / ‖g_t‖` blows up
            # when ‖g_t‖ → 0 (post-warmup plateaus). Use a small
            # floor so `lr_t` is bounded by `D / floor`.
            if grad_norm < 1e-12:
                # Effectively zero grads — fall back to the base lr.
                lr_t = group["lr"]
            else:
                lr_t = D / grad_norm
            # Final NaN/Inf safety net on lr_t itself.
            if not math.isfinite(lr_t):
                lr_t = group["lr"]
            # Final magnitude clamp on lr_t (paper's `d_max` is also
            # the maximum effective LR — when ‖g_t‖ is very small
            # but non-zero, `D / ‖g_t‖` could exceed `d_max`).
            if lr_t > d_max:
                lr_t = d_max
            group["_dadapt_lr"] = lr_t

        # Pass 2: temporarily override group["lr"] for the parent's call.
        base_lrs = []
        for group in self.param_groups:
            base_lrs.append(group["lr"])
            group["lr"] = group["_dadapt_lr"]
        try:
            # Pass 3: delegate to parent AdamW — runs m, v, bias-
            # correction, decoupled WD, all unchanged.
            super().step(closure=None)
        finally:
            # Pass 4: restore base_lrs and clean up scratch.
            for group, base_lr in zip(self.param_groups, base_lrs):
                group["lr"] = base_lr
                if "_dadapt_lr" in group:
                    del group["_dadapt_lr"]
        return loss
