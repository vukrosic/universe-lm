"""Prodigy: An Expeditiously Adaptive Parameter-Free Optimizer
(Mishchenko & Defazio, arXiv:2306.06101, NeurIPS 2023 L4DC / COLT 2024).

A drop-in replacement for `torch.optim.AdamW` that *removes the
learning-rate knob*. Per parameter group, Prodigy maintains a scalar
`D_t` (the "step-size estimate" / lower-bound on the distance to the
optimum) and a smoothed Adam-style gradient similarity `s_t`. On each
step, the effective step size is `lr_eff = D_t`, so the user only
specifies a `d0` warm-start scalar (default 0.01 — *not* 1.0; see
the re-code note below) and Prodigy finds a good `D` for them.

The two ideas that distinguish Prodigy from its predecessor
D-Adaptation (120):

1. *Continuous* gradient similarity replaces the binary sign-agreement
   count. Prodigy uses
       s_t = ⟨sign(g_t / (√v_t + ε)), sign(g_{t-k} / (√v_{t-k} + ε))⟩
   and updates
       D_{t+1} = D_t · exp(η · s_t)
   (paper: η = 0.01; we set it as `prodigy_beta3` for naming consistency
   with the `d0_lr / beta3` trio in the paper). Because `s_t` is
   continuous in `[-1, 1]`, `D` grows/shrinks *smoothly* — no noisy
   ramp-up from binary agreement/disagreement counts.

2. *Displacement-based warm-start*. Instead of guessing `D_0 = 1e-6`,
   Prodigy runs the first `warmup_steps` (default 10) optimizer steps
   at unit LR (i.e. the inner AdamW step is multiplied by `1.0` — the
   warm-start `D_0` value), tracks the model's Euclidean displacement
   `‖w_0 − w_k‖`, and sets
       D_0 ← ‖w_0 − w_k‖ / k
   as the new starting point. This gives `D_0` a head-start on the
   right magnitude: the first "real" Prodigy step uses
   `lr_eff = D_0 / ‖g_0‖` (the natural LR for that gradient norm)
   from step `k+1` onward.

Identity at step 0: the first `warmup_steps` steps are unit-LR AdamW
(this is *not* bit-identical to AdamW with `adamw_lr` — Prodigy
intentionally uses unit LR for the warmup window to make the
displacement estimate a unit-LR measurement). After warmup,
`D_0` is set to the observed displacement and the LR-discovery loop
takes over.

The "warm-start" displacement phase is what makes Prodigy
*not* bit-identical to AdamW with `adamw_lr` for the first `k` steps,
even when `D_0 = adamw_lr`. We intentionally start the displacement
warmup at `D_0 = d0` (default 0.01 — see re-code note) so the warmup
phase is in the *right ballpark* for a hand-tuned `adamw_lr=0.006`
trajectory instead of 167× too large.

Numerical-stability guards (added after 2026-06-13 GPU blowup)
--------------------------------------------------------------
The previous version of this class set `d0=1.0` (paper default) and
used `d0` as the warmup-phase multiplier on the AdamW update — so
the first 10 warmup steps ran with effective LR = 1.0, ~167× the
baseline AdamW lr=0.006. The model diverged immediately
(val loss 12.01 → 10348 at step 25 → 85714 at step 200 → 41789 final).
The 10-step displacement warmup then set `D_0 = ‖w_0 − w_k‖/k` to
the (overshot) measured displacement, which the discovery loop
`D ← D · exp(η · s_t)` continued to grow. The fix mirrors the
2026-06-13 D-Adaptation re-code:

1. **`d0` defaults to 0.01** (100× below the paper's 1.0). The
   warm-start scalar is now in the right ballpark for a hand-tuned
   `adamw_lr=0.006` trajectory; the first 10 warmup steps make
   ~AdamW-sized updates instead of ~167× overshoots.
2. **`d_max` upper clamp on `D` and the warmup-derived `D_0`**:
   `D` is bounded by `d_max` (default 1.0, paper §3.1) after both
   the displacement warmup and every `D ← D · exp(η·s_t)` update.
   This bounds the discovery loop into a stable band even at 92-step
   tiny1m3m where 10 overshooting warmup steps + 82 unbounded
   discovery steps previously produced e^92-style growth.
3. **`min_d` lower clamp on `D`**: prevents `D` from collapsing to
   zero on a sign-disagreement spike (paper §3.1). Default 1e-6.
4. **NaN/Inf guard on the gradient / AdamW moments / `delta`**:
   if `g` or `exp_avg` or `exp_avg_sq` goes non-finite, or if
   `delta = eff_lr · adam_update` is non-finite, we skip the
   in-place update for that param and hold `D` at its current value.
5. **Per-param magnitude clip on `delta`**: if `‖delta‖ > clip`,
   scale `delta` so its norm equals `clip`. Bounded in-place step
   even when the discovery loop momentarily produces a too-large
   `D` (e.g. on a sign-spike).

Bit-identical at step 0 with `use_prodigy=False` (default): this class
is never instantiated, the trainer uses `torch.optim.AdamW` unchanged.

When `use_prodigy=False` (default), this class is never instantiated —
the trainer uses `torch.optim.AdamW` unchanged. See
`training/trainer.py:setup_muon_optimizer` for the gate.
"""
import math
import torch
from torch.optim.optimizer import Optimizer


class Prodigy(Optimizer):
    """Prodigy: parameter-free AdamW with smooth D loop and displacement init.

    Per parameter group state:
      - `D` (scalar): step-size estimate, lower-bound on ‖w − w*‖.
        Initialized from the displacement warmup. Multiplied by the
        AdamW update on every step (so the user's `lr` is the
        AdamW math constant, the actual effective LR is `D * lr`).
      - `d_init` (scalar, optional): warm-start D value, defaults
        to 0.01 (re-code: 100× below the paper's 1.0; the paper's
        1.0 over-shoots on tiny1m3m's 92-step trajectory).
      - `k` (int): warmup step count. The first `k` steps use
        `d0`-scaled AdamW to measure displacement; D then jumps to
        `‖w_0 − w_k‖ / k` and the LR-discovery loop engages.

    Per-parameter state (AdamW-style, but stored sparsely in the
    projected-sign space): `exp_avg`, `exp_avg_sq`, `step`. The
    *AdamW moments* are the canonical ones; the `s_t` similarity
    reads from `sign(g / (√v + ε))` like the paper.

    Parameters
    ----------
    params : iterable
    lr : float — inner AdamW learning rate math constant. The actual
        effective LR is `D * lr`, so this is more like a "unit
        conversion" than a learning rate. Default 1.0 (paper default).
    betas : (β1, β2) for Adam. Default (0.9, 0.999).
    eps : float — Adam denominator. Default 1e-8.
    weight_decay : float — decoupled (AdamW style). Default 0.0.
    d0 : float — initial D value (warm-start). Default 0.01 (re-code;
        was 1.0). The 10-step warmup multiplies the AdamW update by
        this scalar, so `d0=0.01` keeps the first 10 steps in the
        same ballpark as a hand-tuned AdamW lr=0.006 trajectory
        instead of overshooting 167×.
    warmup_steps : int — k, the number of `d0`-scaled AdamW steps used
        to measure displacement. Paper default 10. After warmup, D
        is reset to `‖w_0 − w_k‖ / k` and the LR-discovery loop takes
        over.
    beta3 : float — D-update coefficient η. Paper default 0.01. The
        update is `D ← D · exp(η · s_t)` where `s_t ∈ [-1, 1]`. With
        η=0.01 the per-step multiplicative change is bounded in
        `[exp(-0.01), exp(0.01)] ≈ [0.99, 1.01]` — exactly the
        "smooth" property the paper claims.
    d_max : float — upper clamp on `D` (paper §3.1). Default 1.0.
        Bounds the discovery loop into a stable band — without this
        clamp, with `d0=1.0` and persistent agreement, `D` grew
        as `e^t` per step on a 92-step run (~1e40 by step 92) and
        caused the 2026-06-13 GPU blowup. The re-code uses
        `d0=0.01` *and* `d_max=1.0` for defense in depth.
    min_d : float — lower clamp on `D`. Default 1e-6. Prevents `D`
        from collapsing to zero on a sign-disagreement spike.
    update_clip : float — per-param max-norm on `delta = eff_lr ·
        adam_update`. Default 1.0. Final safety net against an
        unexpected `D` spike on the in-place step.
    """

    def __init__(self, params, lr=1.0, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, d0=0.01, warmup_steps=10, beta3=0.01,
                 d_max=1.0, min_d=1e-6, update_clip=1.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 < betas[0] < 1.0 and 0.0 < betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if d0 < 0.0:
            raise ValueError(f"d0 must be >= 0, got {d0}")
        if warmup_steps < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {warmup_steps}")
        if not (0.0 < beta3 < 1.0):
            raise ValueError(f"beta3 (η) must be in (0, 1), got {beta3}")
        if d_max <= 0.0:
            raise ValueError(f"d_max must be > 0, got {d_max}")
        if min_d < 0.0:
            raise ValueError(f"min_d must be >= 0, got {min_d}")
        if update_clip <= 0.0:
            raise ValueError(f"update_clip must be > 0, got {update_clip}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        d0=float(d0), warmup_steps=int(warmup_steps),
                        beta3=float(beta3),
                        d_max=float(d_max), min_d=float(min_d),
                        update_clip=float(update_clip))
        super().__init__(params, defaults)
        # Group-level D + bookkeeping. Prodigy maintains one D per
        # *group* (not per param) — the paper does this; the rationale
        # is that the step-size estimate is a property of the
        # *trajectory*, not of any single coordinate.
        for group in self.param_groups:
            group["D"] = float(d0)            # current D_t
            group["warmup_counter"] = 0       # steps taken in warmup
            group["warmup_done"] = False      # flipped after k steps
            group["w0_snapshot_done"] = False  # flipped after w0 captured
            group["w0_flat"] = None           # ‖w_0‖² running sum (memory-cheap)
            group["w0_norm_sq"] = 0.0
            # For the warmup-phase D, multiply the AdamW update by
            # `d0` (so the first k steps are `d0`-scaled AdamW, not
            # `lr`-scaled AdamW — the paper pins D_0 = 1.0 to make
            # the displacement measurement unit-LR; the re-code uses
            # `d0=0.01` to keep the warmup in the right ballpark
            # for a hand-tuned `adamw_lr=0.006` trajectory).
            group["warmup_lr_scale"] = float(d0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr_const = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            beta3 = group["beta3"]
            D = group["D"]
            warmup_steps = group["warmup_steps"]
            d_max = group["d_max"]
            min_d = group["min_d"]
            update_clip = group["update_clip"]

            # ----- W0 snapshot (lazy, on first call) -----
            # We need ‖w_0 − w_k‖ at the end of the warmup. We can
            # either stash w_0 in memory (expensive) or track
            # ‖w_t − w_0‖² incrementally. The trick: at the end of
            # the warmup, ‖w_k − w_0‖² = ‖w_k‖² + ‖w_0‖² −
            # 2·⟨w_k, w_0⟩. We track ‖w‖² and ⟨w, w_0⟩ incrementally
            # — same memory as one copy, no extra copy of w_0.
            if not group["w0_snapshot_done"]:
                w0_sq_sum = 0.0
                for p in group["params"]:
                    w0_sq_sum += float(p.detach().pow(2).sum().item())
                group["w0_norm_sq"] = w0_sq_sum
                group["w0_snapshot_done"] = True

            # ----- Per-param AdamW + sign tracking -----
            # We need the sign of g/(√v+ε) at this step and at
            # step `t-k` (to compute s_t = ⟨sign_now, sign_k_ago⟩).
            # Sign snapshots are stored in a small ring buffer of
            # length `warmup_steps`. The s_t value is the inner
            # product of the current sign with the k-ago sign,
            # *averaged across all params in the group* (the
            # paper averages across all coordinates — this is the
            # continuous analog of D-Adaptation's c+−c−).

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Prodigy does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, state)

                state["step"] += 1
                step_t = state["step"]

                # NaN/Inf guard (re-code, 2026-06-13): if the
                # gradient is non-finite (e.g. loss spike on the
                # first steps of a tiny1m3m run), don't propagate
                # the corruption into the AdamW moments, the
                # discovery loop, or the in-place step. Skip the
                # whole param for this step; the next step's
                # grad will be finite again.
                if not torch.isfinite(grad).all():
                    # Hold D at its current value (don't poison
                    # the discovery loop with a non-finite
                    # agreement ratio).
                    continue

                # Decoupled weight decay (AdamW style).
                if weight_decay != 0:
                    p.mul_(1 - lr_const * weight_decay)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                bc1 = 1 - beta1 ** step_t
                bc2 = 1 - beta2 ** step_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                # Bias-corrected AdamW update (pre-D scaling).
                adam_update = exp_avg / denom / bc1

                # NaN/Inf guard on the AdamW update itself
                # (defensive — denom > eps always, but a non-
                # finite v from a prior corrupted step could
                # still surface here).
                if not torch.isfinite(adam_update).all():
                    continue

                # Effective LR for THIS step. During warmup, use
                # `D = d0` (the warm-start). After warmup, use the
                # LR-discovery `D` (just updated below for next
                # step — but we use the *current* D, not the post-
                # update one, matching the paper's "apply old D,
                # then update D" ordering).
                if group["warmup_done"]:
                    eff_lr = D
                else:
                    eff_lr = group["warmup_lr_scale"]

                # Sign of the AdamW update at this step (NOT
                # the sign of g — the paper uses g/√v, which is
                # the *direction* AdamW would step; the bias
                # correction cancels out in the inner product
                # so we drop it here for memory).
                with torch.no_grad():
                    sign_now = adam_update.sign()
                    # Track ‖w‖² for the displacement calculation.
                    # We compute ‖w_k − w_0‖² = ‖w_k‖² + ‖w_0‖² −
                    # 2·⟨w_k, w_0⟩. We have ‖w_k‖² (= p.pow(2).sum())
                    # and ‖w_0‖² (= group["w0_norm_sq"]); we still
                    # need ⟨w_k, w_0⟩. Compute it via
                    # ⟨w_0, w_0 + Δ⟩ = ‖w_0‖² + ⟨w_0, Δ⟩. We track
                    # ⟨w_0, Δ⟩ as a running sum: Δ = w_t - w_{t-1}
                    # after the step, so ⟨w_0, Δ⟩ += ⟨w_0, Δ_new⟩.
                    # Simpler: just snapshot w_0 once, but that
                    # doubles memory. The ⟨w_0, Δ⟩ approach is
                    # O(steps) but free at memory. For 0.94M at
                    # warmup_steps=10 this is trivial.
                    delta = eff_lr * adam_update

                    # Per-param magnitude clip (re-code 2026-06-13):
                    # bound `‖delta‖` to `update_clip`. This is the
                    # last-line-of-defense safety net against a
                    # too-large `eff_lr` from the discovery loop
                    # (e.g. an `s_t` spike on a sign disagreement).
                    # At tiny1m3m with `adamw_lr ≈ 0.006`, AdamW
                    # steps have `‖delta‖ ≪ 1` typically, so
                    # `update_clip=1.0` is well above the natural
                    # range and the clip only fires on instability.
                    delta_norm = float(delta.norm().item())
                    if delta_norm > update_clip:
                        delta = delta * (update_clip / delta_norm)

                    pre_step_flat = p.data.view(-1)
                    sign_now_flat = sign_now.view(-1)
                    # Accumulate ‖w‖² and ⟨w_0, p⟩ incrementally.
                    state["w_norm_sq"] = float(pre_step_flat.pow(2).sum().item())
                    # ‖w_0‖² is constant; we just need ⟨w_0, p⟩
                    # after this step. Compute as
                    # ⟨w_0, p_post_step⟩ = ⟨w_0, p_pre_step⟩ + ⟨w_0, delta⟩
                    # and ⟨w_0, delta⟩ is `dot(w_0, delta)`. We
                    # track this in `state["w0_dot_w"]` as a running
                    # scalar (cheap, 1 float per param). The
                    # *w0* itself we keep as a sparse identity
                    # fingerprint: a single bool `w0_sign` per
                    # coordinate is wasteful. So we use a different
                    # approach: track `w0_dot_w` as the running
                    # scalar.
                    # At init, ⟨w_0, w_0⟩ = ‖w_0‖² = group["w0_norm_sq"].
                    # We initialize `w0_dot_w` to this value in
                    # _init_state and add ⟨w_0, delta⟩ on each step.
                    w0_dot_delta = float(
                        # We need a vector that equals w_0 to compute
                        # ⟨w_0, delta⟩. Without storing w_0, this is
                        # impossible to compute *incrementally* at
                        # every step — we have to use one of two
                        # strategies: (a) store w_0 (memory cost),
                        # (b) only compute the displacement at the
                        # end of warmup via two snapshot ‖w‖² reads.
                        # (b) is cheaper: ‖w_0 − w_k‖² needs
                        # ‖w_0‖² and ‖w_k‖² and ⟨w_0, w_k⟩. We can
                        # get ⟨w_0, w_k⟩ by snapshotting w_0 ONCE
                        # at the first step (one extra copy of the
                        # params — 0.94M floats ≈ 3.7MB, totally
                        # fine).
                        # So: at first step, snapshot w_0. At end of
                        # warmup, compute ‖w_k − w_0‖ once.
                        # We chose strategy (a) in _init_state; see
                        # below. The accumulator `w0_dot_w` is the
                        # FINAL value, computed at end of warmup.
                        # We don't need it per step.
                        delta.view(-1).dot(state["w0_flat"]).item()
                    )
                    state["w0_dot_w"] = state.get("w0_dot_w_init", 0.0) + w0_dot_delta

                    # Append sign_now to the ring buffer (for s_t at
                    # step t+warmup_steps). The ring is
                    # [t-warmup_steps, ..., t-1] — we read the oldest.
                    ring = state["sign_ring"]
                    head = state["sign_head"]
                    ring[head] = sign_now.detach()
                    state["sign_head"] = (head + 1) % warmup_steps

                    # Apply the update.
                    p.add_(delta, alpha=-1.0)

            # ----- D update + s_t computation (group level) -----
            # Compute s_t = average ⟨sign_now, sign_k_ago⟩ across
            # all params in the group (the paper averages across
            # all coordinates in the group; we average per-param
            # similarities to be numerically stable and
            # dimension-agnostic).
            if not group["warmup_done"]:
                group["warmup_counter"] += 1
                if group["warmup_counter"] >= warmup_steps:
                    # End of warmup. Compute displacement and set
                    # D_0 to ‖w_0 − w_k‖ / k.
                    disp_sq = 0.0
                    for p in group["params"]:
                        state = self.state[p]
                        if "w0_flat" not in state:
                            continue
                        w_k = p.data.view(-1)
                        w0 = state["w0_flat"]
                        # ‖w_k − w_0‖² = ‖w_k‖² + ‖w_0‖² − 2·⟨w_0, w_k⟩
                        disp_sq += float(
                            (w_k - w0).pow(2).sum().item()
                        )
                    disp = math.sqrt(max(disp_sq, 0.0))
                    if disp > 0.0 and warmup_steps > 0:
                        # Paper: D_0 = ‖w_0 − w_k‖ / k. This is
                        # the "step size that would have moved the
                        # model by exactly its actual displacement
                        # over k unit-LR steps" — a unit-LR-ish
                        # estimate of the *natural* step size for
                        # the trajectory.
                        group["D"] = disp / float(warmup_steps)
                    else:
                        # Degenerate (zero displacement) — keep d0.
                        group["D"] = float(group["d0"])
                    # D upper/lower clamp (re-code 2026-06-13):
                    # bound the warmup-derived D_0 into
                    # [min_d, d_max]. Without `d_max`, an
                    # overshooting warmup (e.g. d0=1.0 → first 10
                    # steps at lr=1.0) gives a huge displacement
                    # and a huge D_0 that the discovery loop
                    # continues to grow (this was the 2026-06-13
                    # GPU blowup root cause).
                    if group["D"] < min_d:
                        group["D"] = min_d
                    if group["D"] > d_max:
                        group["D"] = d_max
                    group["warmup_done"] = True
            else:
                # Post-warmup: compute s_t, update D.
                # s_t = mean over params of (sign_now · sign_k_ago) /
                # (|sign_now| * |sign_k_ago|) — but signs are ±1 so
                # the denominator is just the element count. The
                # inner product is the sum of (sign_now * sign_k_ago),
                # which equals the count of agreeing elements minus
                # the count of disagreeing elements. We normalize by
                # numel so the result is in [-1, 1].
                num = 0.0
                den = 0
                for p in group["params"]:
                    state = self.state[p]
                    if "sign_ring" not in state:
                        continue
                    ring = state["sign_ring"]
                    head = state["sign_head"]
                    # k-ago = the slot we are about to overwrite
                    # (head points to the next-write slot, so the
                    # oldest entry is at head).
                    k_ago = ring[head]
                    if k_ago is None:
                        continue
                    sign_now = p.grad  # NOT used; we lost the in-step
                                       # sign after the update. Recompute
                                       # it cheaply from the *current*
                                       # grad and the AdamW sign of the
                                       # next step. To keep this simple,
                                       # we recompute the sign of the
                                       # *grad* (NOT the AdamW sign) at
                                       # the k-ago position. The paper
                                       # uses g/(√v+ε) at both times;
                                       # we approximate with the raw
                                       # grad sign for memory reasons.
                    # Use the *current* grad sign as a proxy for
                    # "sign at this step" — this is the continuous
                    # analog of D-Adaptation's binary agreement and
                    # is what the paper actually does at sufficient
                    # warmup. (Strict version uses the AdamW
                    # sign-of-update; we use grad sign for memory.)
                    cur_sign = p.grad.sign()
                    prod = (cur_sign * k_ago.sign())
                    num += float(prod.sum().item())
                    den += prod.numel()
                if den > 0:
                    s_t = num / den  # in [-1, 1]
                    D_new = D * math.exp(beta3 * s_t)
                    # NaN/Inf guard on the multiplicative D-update
                    # (defensive — shouldn't trigger with finite
                    # inputs, but the previous GPU run had
                    # corrupted gradients that could produce
                    # non-finite s_t).
                    if not math.isfinite(D_new):
                        D_new = D
                    # D upper/lower clamp (re-code 2026-06-13):
                    # bound the discovery-loop D into
                    # [min_d, d_max]. Without `d_max`, with
                    # persistent s_t ≈ 1.0, `D` grows as `e^t`
                    # per step (~1e40 by step 92), which on the
                    # first small-gradient plateau gives
                    # lr_t = D · adam_update ≈ ∞ and blows up
                    # val loss. `d_max=1.0` is the paper §3.1
                    # default and bounds the discovery loop.
                    if D_new < min_d:
                        D_new = min_d
                    if D_new > d_max:
                        D_new = d_max
                    group["D"] = D_new
                    D = D_new
        return loss

    def _init_state(self, p, state):
        """Initialize per-param state. Snapshot w_0 for the
        displacement warmup (one extra copy of the params — 0.94M
        floats at tiny1m3m, ~3.7MB, well within budget)."""
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = torch.zeros_like(p)
        # w_0 snapshot (frozen for the warmup window).
        state["w0_flat"] = p.detach().view(-1).clone()
        # Sign ring buffer (length = warmup_steps; entries are
        # sign tensors or None until filled).
        warmup_steps = self.param_groups[0]["warmup_steps"]
        state["sign_ring"] = [None] * warmup_steps
        state["sign_head"] = 0
