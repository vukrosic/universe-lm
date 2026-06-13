"""LookSAM: Periodic Sharpness-Aware Minimization
(Du, Yan, Feng, Zhu, Yang, Sui, ICLR 2023, arXiv:2205.13539).

The compute-efficient variant of SAM (119). Standard SAM does a
2-backward ascent-descent dance on *every* step (2x compute).
LookSAM does the SAM-style 2-backward step only every K steps;
the K-1 steps in between are plain AdamW. With paper default
K=5, the effective compute is ~1.2x of plain AdamW (vs. SAM's
2x), at the cost of ~80% of the flatness benefit.

The mechanism (per step, with inner AdamSAM as in 119):

  if step mod K == 0:                       # SAM step
    ε̂ = ρ · ∇L(w) / ‖∇L(w)‖
    w ← w − lr · AdamW(∇L(w + ε̂))         # perturbed-point descent
  else:                                      # plain step
    w ← w − lr · AdamW(∇L(w))              # baseline

Intuition: SAM's flat-minima benefit comes from *occasional*
sharpness-aware steps, not from every step being sharpness-
aware. The plain steps between SAM steps benefit from the
SAM-induced flat region without paying the 2x cost.

Identity at step 0: with K=5, the first step is a plain AdamW
step (`step_count=0`, `next_is_sam=False`). The first SAM step
fires at `step_count=4` (i.e. the 5th step). So LookSAM is
bit-identical to AdamW at steps 0..K-1 and SAM-shaped at step
K+. This is *more* bit-identical at step 0 than full SAM (119),
which always has the SAM-style first step.

With `use_looksam=False` (default) this class is never
instantiated — the trainer uses `torch.optim.AdamW` unchanged.
"""
import torch
from .sam import AdamSAM


class LookSAM(AdamSAM):
    """AdamW with *periodic* SAM: SAM dance every K steps, plain
    AdamW otherwise.

    Subclass of `optimizers.sam.AdamSAM`. The trainer uses the
    exact same `isinstance(opt, AdamSAM)` machinery as 119 but
    routes LookSAM into either the SAM group (when
    `next_is_sam` is True) or the non-SAM group (when False) on
    each step. When in the non-SAM group, the trainer calls
    `opt.step()` which is the parent's plain `AdamW.step()` on
    the w-grad (no ascent, no closure, no descent). When in the
    SAM group, the trainer calls `opt.first_step(zero_grad=True)
    → closure() → opt.second_step(zero_grad=True)` exactly like
    for plain AdamSAM.

    Parameters
    ----------
    params : iterable
    lr : float
    betas : (β1, β2) — Adam
    eps : float — Adam denominator
    weight_decay : float — decoupled (AdamW style)
    rho : float — SAM perturbation radius. Paper default 0.05
        for Adam-SAM. `rho=0.0` collapses the SAM step to plain
        AdamW (first_step is a no-op, second_step is parent's
        step on the same grad), so even on SAM steps the path
        is bit-identical to plain AdamW.
    k : int — period of the SAM dance. Paper default 5
        (≈1.2x compute vs. 2x for full SAM). `k=1` degenerates
        to full SAM (119). `k` is stashed in the default group
        dict so `.state_dict()` / `.load_state_dict()` round-trip
        it.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, rho=0.05, k=5):
        if k < 1:
            raise ValueError(f"LookSAM k must be >= 1, got {k}")
        if rho < 0.0:
            raise ValueError(f"LookSAM rho must be >= 0, got {rho}")
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, rho=rho)
        # Stash k in the default group dict so it round-trips with
        # the optimizer state_dict. rho is already there from the
        # parent constructor.
        for group in self.param_groups:
            group["k"] = int(k)
        # Step counter: how many inner steps have elapsed (SAM or
        # not). Used to decide which group the trainer routes
        # this object into on the *next* call.
        self.step_count = 0

    @property
    def next_is_sam(self):
        """True iff the next step should be the SAM dance.

        Convention: with `step_count = s` at the start of a step,
        the step is SAM iff `s mod K == K - 1`. So `step_count=0`
        is a plain AdamW step (the first SAM step fires at
        `step_count=K-1`, i.e. the K-th call). After the SAM
        step runs, `step_count` is advanced to `K`, and the next
        K-1 steps are plain AdamW.
        """
        k = self.param_groups[0]["k"]
        return (self.step_count % k) == (k - 1)

    def step(self, closure=None):
        """Plain AdamW step (the non-SAM path).

        Delegates to the parent `AdamSAM.step`-via-`AdamW.step`,
        i.e. runs the baseline AdamW math on the w-grad. No
        ascent, no closure, no descent. With `rho=0.0` the SAM
        step (`first_step + closure + second_step`) also
        collapses to a plain AdamW step (first_step is a no-op,
        second_step is parent's step on the same grad), so the
        SAM and non-SAM paths are bit-equivalent at `rho=0.0`.
        `closure` is silently ignored — LookSAM's SAM path uses
        `first_step` / `second_step` directly (the trainer
        supplies the closure between them).
        """
        # Delegate to the grandparent's plain AdamW.step() to
        # avoid recursing through AdamSAM.step (which requires a
        # closure and runs the full ascent+closure+descent dance).
        # PyTorch MRO: LookSAM -> AdamSAM -> AdamW; calling
        # `AdamW.step(self)` invokes the parent's plain AdamW
        # step directly.
        torch.optim.AdamW.step(self)
        self.step_count += 1

    def first_step(self, zero_grad=False):
        """SAM ascent: w ← w + ε̂.

        Override only to suppress the SAM dance on non-SAM
        steps. The trainer is expected to route this object
        into the SAM group only when `next_is_sam` is True, so
        we trust that here. (`next_is_sam` is a property; the
        trainer checks it before the SAM dance.)
        """
        super().first_step(zero_grad=zero_grad)

    def second_step(self, zero_grad=False):
        """SAM descent: restore w, AdamW step on perturbed grad.

        Advances `step_count` by 1 (the SAM step counts as one
        inner step, just like the plain `step()` path).
        """
        super().second_step(zero_grad=zero_grad)
        self.step_count += 1
