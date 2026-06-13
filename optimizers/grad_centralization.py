"""127 — Gradient Centralization (Yong et al. 2020, arXiv:2004.01461).

A drop-in replacement for `torch.optim.AdamW` that mean-centers each
gradient tensor along a specified axis BEFORE the optimizer sees it.

Mechanism
---------
Standard optimization: `w ← w − lr · AdamW(g_t)`. GC modifies `g_t`
before the optimizer's update runs:

    μ = mean(g_t, dim=axis)        # mean across the chosen axis
    g_t_centralized = g_t − μ       # subtract the mean
    w ← w − lr · AdamW(g_t_centralized)

For a 2-D weight matrix `W ∈ R^{n×m}`, `axis=1` (the default) is the
output axis, giving each output neuron zero-mean input gradient. For
4-D conv weights we apply per-filter mean subtraction (axis over the
spatial dims `(2, 3)`).

Mathematically, GC is a linear operator `g ↦ (I − (1/m)·1·1^T) · g`
that projects the gradient onto the subspace orthogonal to the
all-ones vector. The projection is rank-(m-1) and removes only the
constant component — the variance of `g` is preserved, so the
optimizer's `(m, v)` updates run on a centered gradient.

Identity at step 0
------------------
The forward graph is unchanged — `val_loss` at step 0 (computed
before any optimizer step) is bit-identical to baseline. The first
optimizer step itself is NOT bit-identical: the centered gradient
has zero mean per output neuron, while AdamW's first step sees the
raw gradient. This is the lever's signature, not a bug. With
`use_gc=False` (default) the trainer uses `torch.optim.AdamW`
unchanged — the `GCAdamW` class is never instantiated.

When to use
-----------
GC is *compositional* with every AdamW replacement. The trainer
routes the AdamW path through `GCAdamW` only when `use_gc=True` AND
no specific AdamW replacement is active. Stacking GC with SAM /
D-Adapt / Prodigy / etc. is meaningful but would need its own
wiring (each replacement has its own `.step()` signature). Default
off → baseline path bit-identical.

Reference
---------
- Yong, Fortuin, Morariu, Salzmann, Ni, "Gradient Centralization: A
  New Optimization Technique for Deep Neural Networks", ICONIP 2020,
  arXiv:2004.01461. https://arxiv.org/abs/2004.01461
- See `autoresearch/ideas/127-grad-centralization/idea.md` for the
  bet and design sketch.
"""
import torch
from torch.optim import AdamW


class GCAdamW(AdamW):
    """AdamW with Gradient Centralization (Yong et al. 2020).

    Subclass of `torch.optim.AdamW`. Before the parent's `.step()`,
    each `param.grad` is mean-centered along `gc_axis` (for 2-D and
    3-D params) or along the spatial axes `(2, 3)` (for 4-D conv
    weights). The per-parameter `(exp_avg, exp_avg_sq)` state and
    the decoupled-WD math are unchanged — only the gradient input
    is centered.

    Parameters
    ----------
    params : iterable
        Standard PyTorch optimizer params iterable.
    lr : float
        Learning rate.
    betas : (β1, β2)
        Adam betas.
    eps : float
        Adam denominator.
    weight_decay : float
        Decoupled weight decay (AdamW style).
    gc_axis : int
        Axis along which to mean-subtract 2-D and 3-D tensors.
        Default `1` (the output axis for `W ∈ R^{n×m}`). Ignored for
        1-D tensors (no-op) and 4-D conv tensors (per-filter spatial
        mean over axes `(2, 3)` is always applied).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, gc_axis=1):
        if gc_axis not in (0, 1):
            raise ValueError(
                f"GCAdamW gc_axis must be 0 or 1, got {gc_axis}"
            )
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay)
        for group in self.param_groups:
            group["gc_axis"] = int(gc_axis)

    @torch.no_grad()
    def _centralize(self, group):
        """Apply Gradient Centralization to each param.grad in `group`.

        - 1-D tensors: pass through (no-op — no clear output-axis
          semantics for a flat vector).
        - 2-D / 3-D tensors: subtract the mean along `gc_axis`.
        - 4-D tensors (conv weights): subtract the per-filter mean
          over the spatial axes `(2, 3)`.

        Operates in-place on `p.grad` (which is a leaf the parent
        will consume). Doesn't allocate a per-step backup tensor —
        the parent's `.step()` runs immediately after.
        """
        axis = group["gc_axis"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            if g.is_sparse:
                raise RuntimeError(
                    "GCAdamW does not support sparse gradients"
                )
            if g.ndim == 1:
                # No output-axis semantics for a flat vector.
                continue
            if g.ndim == 4:
                # Conv weight (out, in, kH, kW): per-filter mean over
                # the spatial axes (2, 3). Keeps (out, in) statistics.
                dims = (2, 3)
            else:
                dims = (axis,)
            # `g.mean(dims, keepdim=True)` then subtract. Equivalently
            # to `g -= g.mean(dims, keepdim=True)`. In-place form
            # avoids an extra full-sized allocation.
            g.sub_(g.mean(dim=dims, keepdim=True))

    @torch.no_grad()
    def step(self, closure=None):
        # Apply GC to every param's grad BEFORE the parent's step
        # so AdamW's (m, v) EMA updates see the centered gradient.
        # Backwards-mutating p.grad would normally break the autograd
        # graph for any further backward passes — but `.step()` is
        # called after `loss.backward()` completes, so no further
        # backward is pending. (The trainer pattern is fwd → bwd →
        # step, exactly like vanilla AdamW.)
        for group in self.param_groups:
            self._centralize(group)
        return super().step(closure)