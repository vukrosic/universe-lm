"""SAM: Sharpness-Aware Minimization (Foret et al. 2020, arXiv:2010.01412).

Wraps an inner AdamW optimizer with an adversarial ascent step
followed by descent at the perturbed point. The mechanism:

  Standard SGD:  w ← w − lr · ∇L(w).
  SAM:            w ← w − lr · ∇L(w + ε̂),
                  where ε̂ = ρ · ∇L(w) / ‖∇L(w)‖.

The intuition: the gradient at the worst-case nearby point
approximates the *sharpness* of the local loss surface; descending
on it makes the optimizer prefer flat minima (which generalize
better) over narrow spikes.

This implementation uses the Adam-SAM flavor: ε̂ = ρ · ∇L(w) /
‖∇L(w)‖, and the descent uses AdamW's per-parameter adaptive
step on ∇L(w + ε̂). Validated at ImageNet ResNet-50 (top-1
+0.4-1.3%) and GPT-2 fine-tuning (BMRC 2022 follow-up).

Identity at step 0: with `w = w_init` and `rho > 0`, the first
step is NOT bit-identical to AdamW (the perturbation is O(ρ) in
the gradient direction). With `rho = 0.0`, SAM collapses to AdamW
— the `first_step` is a no-op (no ascent), the second backward
at w+ε̂ is at w (no perturbation), and the `second_step` runs
the parent's `step()` on the same grad. The first_step is gated
on `rho > 0`, so `AdamSAM(..., rho=0.0)` is bit-identical to
plain AdamW. The deviation at rho > 0 is bounded by O(ρ) and
becomes the pass-bar of the experiment.

When `use_sam=False` (default), this class is never instantiated
— the trainer uses `torch.optim.AdamW` unchanged. See
`training/trainer.py:setup_muon_optimizer` for the gate.
"""
import torch
from torch.optim import AdamW


class AdamSAM(AdamW):
    """AdamW with SAM (Sharpness-Aware Minimization) ascent-descent.

    Subclass of `torch.optim.AdamW`. The SAM flow is split into
    three methods so the trainer can interleave the SAM
    ascent/descent with non-SAM optimizers (e.g. Muon) that don't
    participate in the perturbation:

      1. `first_step(zero_grad=True)`: ascent to w + ε̂, store ε̂
         in state, zero the grad.
      2. *Caller* runs a second forward+backward at w + ε̂ (the
         perturbed grad populates `p.grad` for SAM-managed params).
      3. `second_step(zero_grad=True)`: restore w (subtract ε̂),
         then delegate to the parent AdamW `.step()` on the
         perturbed grad.

    The class also exposes a single-call `.step(closure)` for
    convenience when the caller can supply a closure that runs
    the second forward+backward.

    Parameters
    ----------
    params : iterable
    lr : float
    betas : (β1, β2) — Adam
    eps : float — Adam denominator
    weight_decay : float — decoupled (AdamW style)
    rho : float — SAM perturbation radius. Paper default 0.05 for
        SGD; 0.01-0.05 for Adam-versions. `rho = 0.0` collapses
        to plain AdamW (first_step is a no-op, second_step is
        parent's step on the same grad).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, rho=0.05):
        if rho < 0.0:
            raise ValueError(f"SAM rho must be >= 0, got {rho}")
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay)
        # Stash SAM-specific knob in the default group dict so
        # `.state_dict()` / `.load_state_dict()` round-trip it.
        for group in self.param_groups:
            group["rho"] = float(rho)

    @torch.no_grad()
    def _grad_norm(self):
        """Compute the L2 norm of the gradient across all params."""
        norm = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                norm = norm + p.grad.detach().pow(2).sum()
        return norm.sqrt()

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Ascent step: w ← w + ε̂ where ε̂ = ρ · ∇L(w) / ‖∇L(w)‖.

        Stores ε̂ in `state[p]["e_w"]` for restoration in
        `second_step`. With `rho = 0.0` this is a no-op (no
        ascent, the SAM path is bit-identical to the parent's
        `step()`).
        """
        rho = self.param_groups[0]["rho"]
        if rho == 0.0:
            if zero_grad:
                self.zero_grad()
            return
        grad_norm = self._grad_norm()
        scale = rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (p.grad * scale).detach()
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Descent step: restore w, then AdamW step on perturbed grad.

        w ← w - ε̂ (restore), then w ← w - lr · AdamW(∇L(w + ε̂)).
        Delegates to the parent `AdamW.step()` so the AdamW math
        (m, v, bias-correction, decoupled WD) runs unchanged on
        the perturbed-point gradient.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])
                    del self.state[p]["e_w"]
        # Delegate to parent AdamW — runs on the perturbed grad
        # populated by the caller's second forward+backward.
        super().step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        """Full SAM step. Requires a closure for the second backward.

        Sequence (when `rho > 0`):
          1. `first_step(zero_grad=True)` — ascent to w + ε̂.
          2. `closure()` — second forward+backward at w + ε̂.
          3. `second_step(zero_grad=True)` — restore w, AdamW step
             on perturbed grad.

        With `rho = 0.0` the first_step is a no-op and the closure
        is the only thing that runs (effectively a single backward
        pass at w). This is bit-identical to plain AdamW with a
        closure-based pattern.
        """
        if closure is None:
            raise ValueError(
                "AdamSAM.step requires a closure for the second "
                "backward pass. Use first_step()/second_step() "
                "directly when interleaving with non-SAM optimizers."
            )
        with torch.enable_grad():
            self.first_step(zero_grad=True)
            closure()
            self.second_step(zero_grad=True)
