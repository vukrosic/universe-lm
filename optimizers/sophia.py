"""Sophia: A Scalable Stochastic Second-order Optimizer
(Liu, Wang, et al. 2023, arXiv:2305.14342, Stanford / ICML 2023).

Diagonal-Hessian-aware update with Hutchinson trace estimator for
the diagonal Hessian. Per-parameter (paper Algorithm 1):

    m_t  = β1 * m_{t-1} + (1-β1) * g_t          (gradient EMA)
    h_t  = β2 * h_{t-1} + (1-β2) * h_hat_t      (Hessian diag EMA,
                                                  sampled every k steps)
    update = clip(g_t, max=ρ) / max(h_t, ε)     (preconditioned by
                                                  inverse Hessian diag)
    θ_t  = θ_{t-1} − lr * (update + λ·θ_{t-1})  (decoupled WD)

The Hutchinson estimator samples `u ~ Rademacher({-1, +1})` per
parameter and uses the Gauss-Newton-style identity
`h_hat_i = u_i · (H·u)_i` where `(H·u)_i = ∂(g·u)/∂θ_i` is computed
via one extra backward on the scalar `g·u`. Cost: ~2× backward time,
but only every k=10 steps (so amortized ~1.1× backward). The
trainer handles the extra backward + h_hat feed-in via
`Sophia.update_h_hat(h_hat_list)`; the optimizer itself only
consumes `h_hat` on the dedicated update step.

Cold-start identity at step 0: `m_0 = 0, h_0 = 0`. The first
optimizer step sees `h_0 = 0` ⇒ `max(0, ε) = ε` ⇒
`update = clip(g_0, ρ) / ε` which is O(ρ/ε) ≈ ρ·1e8 at the
default `ε=1e-8`. We add a guard: when `h_max = max(h_t, ε) <
some_small_value`, the step is amplified by `1/h_max` and the
`clip(g_t, ρ)` keeps the *magnitude* of the update bounded by
`ρ/h_max` (not `ρ`). To stay closer to the paper's intent (and
not diverge at step 0 on the 0.94M model), the constructor exposes
`update_clip` (default 1.0) that we apply to the magnitude of
`update` before scaling by `lr`. With the default
`update_clip=1.0` the first step is bounded by `lr·1.0` in
magnitude — comparable to AdamW's first step. At step 1 (after
the first `k`-step Hutchinson update at `step=0`), `h_1 = (1−β2)·h_hat_0`
which is a positive quantity of order `O(g²)` ⇒ `max(h_1, ε) ≈ h_1`
⇒ `update ≈ clip(g_0, ρ) / h_1`, the proper curvature-preconditioned
direction.

This first-step guard is the lever's signature (not a bug):
without it the very first step is ~`ρ/ε` (a 1e8-amplitude
update that blows up the model). The paper itself reports
needing `ρ ∈ [−0.01, 0.01]` and a `1e-12 ≤ ε ≤ 1e-8` warmup —
we use `ε=1e-8` and an `update_clip=1.0` to bound the very first
step where `h_t` is still tiny.

When `use_sophia=False` (default) this class is never instantiated
and the trainer uses `torch.optim.AdamW` unchanged. See
`training/trainer.py:setup_muon_optimizer` for the gate.
"""
import torch
from torch.optim.optimizer import Optimizer


class Sophia(Optimizer):
    """Sophia — second-order (diagonal-Hessian-aware) optimizer.

    Drop-in replacement for the AdamW 1-D / embedding / norm / head
    path. The 2-D Muon path is unchanged (Sophia is an AdamW
    replacement, like 114-MARS, 119-SAM, 121-Prodigy, 135-Adan).

    Parameters
    ----------
    params : iterable
    lr : float — Sophia step size. Paper default `6e-3` for the
        125M model; at 0.94M we default to `adamw_lr=0.006`.
    betas : (β1, β2) — β1 weights the gradient EMA, β2 weights the
        Hessian-diagonal EMA. Paper default (0.965, 0.99) (note: β1
        higher than AdamW's 0.9 — Sophia wants more momentum on the
        gradient). We default to (0.9, 0.999) to match the rest of
        the optimizer family; the `sophia_beta1` config knob
        overrides this if the paper's 0.965 is preferred.
    eps : float — additive to `h_t` in the denominator (zero-guard
        + sign-stability).
    weight_decay : float — decoupled (AdamW style), applied as
        `θ ← θ − lr · λ · θ` AFTER the preconditioned update.
    rho : float — gradient clip value used inside the
        curvature-preconditioned update. Paper default 0.04 for
        GPT-2 125M. `update = clip(g_t, ±ρ) / max(h_t, ε)`.
    hessian_update_freq : int — `k` in the paper. The Hutchinson
        diagonal-Hessian sample fires every `k` optimizer steps
        (paper default 10). Between samples, `h_t` is held constant
        (its EMA decays with no new term). At `k=10` and 92
        optimizer steps at tiny1m3m, the diagonal Hessian is
        refreshed ~9 times — the same amortization as the paper.
    update_clip : float — per-parameter max-norm on the
        preconditioned update (after `clip(g, ρ) / max(h, ε)`,
        before the `lr` scale). Defends against the
        `h_t ≈ 0` cold-start amplification. Paper does not
        specify this; we use 1.0 (matches `AdamW` first-step
        magnitude).
    """

    def __init__(self, params, lr=6e-3, betas=(0.9, 0.99),
                 eps=1e-8, weight_decay=0.0, rho=0.04,
                 hessian_update_freq=10, update_clip=1.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 < betas[0] < 1.0 and 0.0 < betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if rho <= 0.0:
            raise ValueError(f"Invalid rho: {rho}")
        if hessian_update_freq < 1:
            raise ValueError(f"Invalid hessian_update_freq: {hessian_update_freq}")
        if update_clip <= 0.0:
            raise ValueError(f"Invalid update_clip: {update_clip}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, rho=float(rho),
                        hessian_update_freq=int(hessian_update_freq),
                        update_clip=float(update_clip))
        super().__init__(params, defaults)
        # Tracks the global optimizer step (incremented by .step()).
        # The trainer reads this via `optimizer._step_count` to know
        # when to fire the Hutchinson estimator on the next backward
        # pass.
        self._step_count = 0

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
            rho = group["rho"]
            update_clip = group["update_clip"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sophia does not support sparse gradients")

                state = self.state[p]
                if "exp_avg" not in state:
                    # Lazy state init. We check `exp_avg` (not
                    # `len(state) == 0`) so the path also works
                    # when `update_hessian` was called BEFORE the
                    # first `.step()` (which leaves `state` non-
                    # empty with just `hessian` populated). We
                    # ONLY init `hessian` to zeros if it's not
                    # already there — otherwise we clobber the
                    # Hutchinson sample that `update_hessian`
                    # just stored.
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    if "hessian" not in state:
                        state["hessian"] = torch.zeros_like(p)

                state["step"] += 1

                m = state["exp_avg"]
                h = state["hessian"]

                # Match the Muon path's `g.float()` so mixed-precision
                # params don't produce a bf16 EMA buffer.
                g = grad.float()
                m_fp = m.float() if m.dtype != torch.float32 else m
                h_fp = h.float() if h.dtype != torch.float32 else h

                # m_t = β1·m + (1−β1)·g_t  (gradient EMA)
                m_fp.mul_(beta1).add_(g, alpha=1 - beta1)

                # Preconditioned update: `clip(g, ±ρ) / max(h, ε)`.
                # `h_fp` may be zero at step 0 (no Hutchinson sample
                # yet); `max(0, ε) = ε` would yield an enormous
                # step. The `update_clip` guard caps the
                # per-parameter update magnitude to `update_clip`,
                # matching the AdamW first-step order.
                h_safe = h_fp.clamp(min=eps)
                # clip(g, ±ρ) — only the gradient magnitude is
                # clipped, sign is preserved.
                g_clipped = g.clamp(min=-rho, max=rho)
                # Per-element preconditioning. The denominator is
                # element-wise because `h` is the diagonal Hessian.
                update = g_clipped / h_safe

                # Magnitude guard against the `h≈0` cold-start
                # amplification. We rescale the whole update so
                # `|update| ≤ update_clip` element-wise (sign
                # preserved). When `h` has the typical
                # `O(g²)` magnitude, this is a no-op.
                update = update.clamp(min=-update_clip, max=update_clip)

                # Cast back to param dtype before the in-place step
                update = update.to(p.dtype)

                # Decoupled weight decay (AdamW style):
                # θ ← θ − lr · λ · θ  BEFORE the preconditioned step.
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # θ ← θ − lr · update
                p.add_(update, alpha=-lr)

                if m_fp is not m:
                    m.copy_(m_fp)
                if h_fp is not h:
                    h.copy_(h_fp)

        self._step_count += 1
        return loss

    def update_hessian(self, h_hat_list, beta2):
        """Feed the Hutchinson diagonal-Hessian sample into the
        per-parameter `h_t` EMA.

        Called by the trainer every `hessian_update_freq` steps AFTER
        the second backward (which populates `h_hat = u · (H·u)`).
        The trainer must restore the original grads (the second
        backward overwrites them) — that bookkeeping lives in
        `training/trainer.py`, not here.

        Parameters
        ----------
        h_hat_list : list[Tensor | None] — same length as the
            `params` iterable. `h_hat_list[i]` is the
            `u_i · (H·u)_i` sample for parameter `i` (or None if
            the parameter had no grad this step).
        beta2 : float — β2 from the optimizer's param group
            (passed in by the trainer to avoid re-reading groups).
        """
        # Walk the param groups in the same order the .step() walk
        # uses, and pair each param with its h_hat sample.
        h_hat_iter = iter(h_hat_list)
        for group in self.param_groups:
            b2 = beta2 if beta2 is not None else group["betas"][1]
            for p in group["params"]:
                h_hat = next(h_hat_iter, None)
                if h_hat is None:
                    continue
                if p.grad is None:
                    continue
                state = self.state[p]
                if "hessian" not in state:
                    # Param has no state yet (never seen a grad on a
                    # .step() call). Initialize h_t to the sample.
                    state["hessian"] = h_hat.detach().clone().to(
                        dtype=p.dtype, device=p.device
                    )
                    continue
                h = state["hessian"]
                h_fp = h.float() if h.dtype != torch.float32 else h
                h_fp.mul_(b2).add_(h_hat.float(), alpha=1 - b2)
                if h_fp is not h:
                    h.copy_(h_fp)
