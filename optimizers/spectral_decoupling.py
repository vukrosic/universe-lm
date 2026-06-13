"""Spectral Decoupling (Yong, Pehlivan, Morariu, Tsang 2022,
arXiv:2202.05380, NeurIPS 2022).

Reformulation of L2 weight decay that decouples the
*regularization* (magnitude shrinkage, parallel to w) from the
*gradient-driven* direction learning. Standard L2 update is
    `w ← w − lr · (g + λ · w)`.
The `λ·w` term is parallel to w, so it shrinks magnitude
without rotating — but combined with `g` it changes the
resultant gradient direction. Spectral Decoupling projects
the gradient perpendicular to w before the regularizer is
applied:
    `g_SD = g − (⟨g, w⟩ / ‖w‖²) · w`
    `w ← w − lr · (g_SD + λ · w)`.
The `λ·w` magnitude-shrinking term is unchanged (its job is
along w); only the gradient-driven direction signal is
"cleaned" — its component along w is removed.

Identity at step 0: with symmetric inits (Kaiming, etc.) the
per-parameter `⟨g_0, w_0⟩` is small but nonzero, so the
projection removes an `O(1/n)` component of `g_0`. The first
optimizer step is approximately `w ← w − lr · g_0` (the same
as AdamW's first step modulo an `O(1/n)` correction). With
`use_sd=False` (default) plain `torch.optim.AdamW` is used
— baseline bit-identical.

Compositional: SD is just a per-param gradient transform on
top of any Adam-style optimizer. SD + AdamW, SD + SAM, SD +
Prodigy all reduce to "wrap the inner optimizer's gradient
projection in front of the step". Here we wire it only on the
AdamW bucket (matches the idea's design sketch); the 2-D Muon
path is untouched.
"""
import torch
from torch.optim import AdamW


class SDAdamW(AdamW):
    """AdamW with Spectral Decoupling gradient projection.

    Subclass of `torch.optim.AdamW` whose `step()` first projects
    each per-parameter gradient perpendicular to the weight
    direction (`g ← g − (⟨g,w⟩/‖w‖²)·w`) before delegating to
    the parent AdamW step. The parent's decoupled weight decay
    `λ·w` is unchanged — it acts along w (parallel), so the
    magnitude-shrinking role is preserved.

    Parameters
    ----------
    params : iterable
    lr : float — AdamW step size.
    betas : (β1, β2) — Adam momentum / 2nd-moment coefficients.
    eps : float — Adam denominator floor.
    weight_decay : float — decoupled (AdamW style), passed through.
    sd_lambda : float — if `0.0`, the projection is fully inert
        and `SDAdamW` is bit-identical to plain AdamW. The
        non-zero value `sd_lambda` rescales the projected-out
        component: `g_SD = g − sd_lambda · (⟨g,w⟩/‖w‖²)·w`.
        With `sd_lambda = 1.0` this is the paper's projection
        (full removal). With `sd_lambda = 0.0` it is a no-op.
        The default `1.0` matches the paper.
    sd_eps : float — floor for `‖w‖²` to avoid division by zero
        on freshly-zeroed weights.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, sd_lambda=1.0, sd_eps=1e-12):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay)
        for group in self.param_groups:
            group["sd_lambda"] = float(sd_lambda)
            group["sd_eps"] = float(sd_eps)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            sd_lambda = group.get("sd_lambda", 0.0)
            if sd_lambda == 0.0:
                # Inert: identical to plain AdamW. Skip the
                # projection entirely so the path is bit-identical
                # to the parent class.
                continue
            sd_eps = group.get("sd_eps", 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                w = p.data
                # ‖w‖² with a tiny floor so freshly-zeroed weights
                # don't produce NaN gradients (defensive — shouldn't
                # happen in practice since AdamW only handles
                # trainable params).
                w_norm_sq = w.pow(2).sum().clamp(min=sd_eps)
                # Scalar projection coefficient (⟨g, w⟩ / ‖w‖²).
                g_dot_w = (g * w).sum() / w_norm_sq
                # Project gradient off w direction:
                #   g ← g − sd_lambda · (⟨g, w⟩ / ‖w‖²) · w
                g.sub_(w, alpha=g_dot_w * sd_lambda)
        # Delegate to AdamW — runs bias-corrected m̂_t / (√v̂_t + ε)
        # on the projected gradient, plus decoupled λ·w. The
        # projection never touches w itself, so the decoupled WD
        # applied by the parent is unchanged.
        return super().step(closure)