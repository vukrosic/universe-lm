"""AdamP: Adam with Projection-Based Update
(He, Liu, Mao, Chen, Zhang, "AdamP: Slowing Down the Slowdown for
Momentum Accelerators on Scale-Invariant Weights", arXiv:2006.08217,
NeurIPS 2020).

Standard AdamW computes the update `Δ = lr · m̂_t / (√v̂_t + ε)`
and applies it as `w ← w − Δ + λ · w` (the `+λ·w` is decoupled
weight decay). The update is purely *gradient-driven*, so even
when it shrinks magnitude (via WD) the direction it travels in is
governed by `m̂/√v̂`. Over many steps, this direction can rotate
the weight *away* from its optimal alignment — a known pathology
of L2 + momentum on scale-invariant weights.

AdamP projects `Δ` onto the orthogonal complement of `w` BEFORE
applying it, removing the component of `Δ` that lies along `w`:
    δ        = (Δ · w / ‖w‖²) · w          (component of Δ along w)
    Δ_proj   = Δ − δ                        (orthogonal component)
    w       ← w − Δ_proj + λ · ‖w‖ · w_norm

The projected update `Δ_proj` is now *strictly perpendicular* to
`w`, so applying it rotates the weight's *direction* without
changing its magnitude. The L2 reg acts only on magnitude (`λ‖w‖`)
without rotating — the two are spectrally decoupled.

Identity at step 0: with standard symmetric inits (Kaiming/Xavier),
`w_0` is small and roughly isotropic, so `Δ_0 · w_0 / ‖w_0‖²` is
small (O(1/√fan_in) in expectation). Therefore `Δ_proj ≈ Δ_0` and
the first AdamP step ≈ the first AdamW step modulo an O(1/√d)
correction. With `adamp_lambda = 0` the magnitude reg is removed,
and with `use_adamp=False` (default) plain `torch.optim.AdamW`
is used — baseline path bit-identical.
"""
import torch
from torch.optim.optimizer import Optimizer


class AdamP(Optimizer):
    """AdamP — Adam with projection-based update (He et al. 2020).

    Drop-in replacement for the AdamW 1-D / embedding / norm / head
    path. The 2-D Muon path is unchanged (AdamP is an AdamW
    replacement, like 114-MARS, 119-SAM, 120-DAdapt, 121-Prodigy,
    123-CAME, 124-RAdam, 126-AdaShift, 127-GC, 128-SD).

    Parameters
    ----------
    params : iterable
    lr : float — AdamP step size (paper default matches AdamW).
    betas : (β1, β2) — Adam momentum / 2nd-moment coefficients.
    eps : float — Adam denominator floor.
    weight_decay : float — L2 magnitude shrinkage coefficient
        (paper's `λ · ‖w‖` form — applied along w, no rotation).
    adamp_lambda : float — projection strength. If `0.0`, the
        projection is fully inert (Δ_proj = Δ) and AdamP collapses
        to AdamW-with-magnitude-WD. With `1.0` (paper default)
        the projection is full. Intermediate values give a
        soft-projection lever.
    adamp_eps : float — floor for ‖w‖² to avoid division by zero
        on freshly-zeroed weights.
    """

    def __init__(self, params, lr=0.006, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0,
                 adamp_lambda=1.0, adamp_eps=1e-12):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not (0.0 < betas[0] < 1.0 and 0.0 < betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not (0.0 <= adamp_lambda <= 1.0):
            raise ValueError(f"Invalid adamp_lambda: {adamp_lambda}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        adamp_lambda=float(adamp_lambda),
                        adamp_eps=float(adamp_eps))
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
            adamp_lambda = group["adamp_lambda"]
            adamp_eps = group["adamp_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamP does not support sparse gradients")

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

                # m ← β1·m + (1−β1)·g    (Adam 1st moment)
                m_fp.mul_(beta1).add_(g, alpha=1 - beta1)
                # v ← β2·v + (1−β2)·g²   (Adam 2nd moment)
                v_fp.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # Bias-corrected moments (standard AdamW convention).
                bias_correction1 = 1.0 - beta1 ** step_t
                bias_correction2 = 1.0 - beta2 ** step_t
                m_hat = m_fp.div(bias_correction1) if bias_correction1 > 0 else m_fp
                v_hat = v_fp.div(bias_correction2) if bias_correction2 > 0 else v_fp

                # Adam update direction: Δ = m̂ / (√v̂ + ε)
                denom = v_hat.sqrt().add_(eps)
                delta = m_hat.div(denom)

                # AdamP projection: remove the component of Δ along w.
                # Only meaningful for nd ≥ 2 weights (1-D scalars have
                # no "direction" to project onto). For 1-D params we
                # fall back to the bare Adam step + scalar WD.
                if adamp_lambda > 0 and p.ndim >= 2:
                    w = p.data.float() if p.dtype != torch.float32 else p.data
                    w_norm_sq = w.pow(2).sum().clamp(min=adamp_eps)
                    # Scalar projection coefficient (Δ · w / ‖w‖²)
                    delta_dot_w = (delta * w).sum() / w_norm_sq
                    # δ = delta · w / ‖w‖² (broadcast back to weight shape)
                    delta = delta.sub(w, alpha=delta_dot_w * adamp_lambda)
                # (else: 1-D scalar or projection disabled → bare Δ.)

                # Magnitude-only L2 reg: λ · ‖w‖ · ŵ  (the paper's form).
                # Applied as a positive add (L2 *removes* weight) so we
                # subtract from w. For 1-D scalars and disabled lambda,
                # this collapses to the standard `λ · w` decoupled WD.
                if weight_decay != 0:
                    if adamp_lambda > 0 and p.ndim >= 2:
                        w = p.data.float() if p.dtype != torch.float32 else p.data
                        w_norm = w.pow(2).sum().sqrt().clamp(min=adamp_eps)
                        # λ · ‖w‖ · (w / ‖w‖) = λ · w · (‖w‖/‖w‖) → λ · w
                        # (paper's "L2 reg on magnitude" formula reduces to
                        # the standard decoupled WD vector here, just along
                        # the w direction — equivalent to AdamW's λ·w).
                        wdir = w.div(w_norm)
                        wd_term = wdir.mul(weight_decay * w_norm)
                        wd_term = wd_term.to(p.dtype)
                        p.sub_(wd_term.mul(lr))
                    else:
                        p.mul_(1 - lr * weight_decay)

                # Cast back to param dtype before the in-place step
                delta = delta.to(p.dtype)
                p.add_(delta, alpha=-lr)

                if m_fp is not m:
                    m.copy_(m_fp)
                if v_fp is not v:
                    v.copy_(v_fp)

        return loss