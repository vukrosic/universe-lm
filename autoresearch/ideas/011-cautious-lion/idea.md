---
id: 011-cautious-lion
status: needs-taste
round: 1
updated: 2026-06-09T09:35:00Z
---

# 011 — Cautious Lion

## Source
Liang et al., "Cautious Optimizers: Improving Training with One Line of Code" (arXiv:2411.16085, 2024). The "Cautious" mask trick generalized from Adam to *any* sign-based optimizer. Lion (Chen et al., 2023) is a natural target: its update is `update = sign(β₁·m + (1-β₁)·g)` — already a sign — so applying the cautious mask is the cleanest possible one-liner.

## Mechanism
Lion's update direction is already sign-based. The Cautious mask zeroes the update where `sign(update) ≠ sign(g)`. Equivalent to: skip a step when the momentum and the current gradient disagree on sign. Implementation: in `optimizers/lion.py`, add a `use_cautious: bool = False` flag; in the step, after computing `update = sign(...), set update = update * (update * g > 0).float()`. Then rescale the update by `1 / mask.mean().clamp(min)` to keep the effective LR constant. ~5-10 LoC.

## Why it's worth a slot
- **Mechanism is orthogonal to 001/002.** Cautious-Muon and Cautious-AdamW both apply the mask to momentum-based optimizers. Lion has *no momentum buffer equivalent* — its "momentum" is the sign-of-momentum, and masking there is a different signal (it's about the update direction, not the variance).
- **Strong paper results across scales.** Liang et al. report consistent gains for Cautious variants at 7B-70B.
- **Identity-safe, ~5 LoC, no compute cost.** Drop-in for Lion (which is a Muon competitor in our routing table).
- **Negative result would still be informative** — would tell us Cautious is momentum-variance-specific, not sign-update-general.

## Hypothesis
Δ in [−0.01, −0.03] val loss on tiny1m3m / seed 42 vs Lion baseline. Mechanism: skipping disagreement steps reduces wasted LR on noisy gradient signs.

## Wiring
- New file: `optimizers/cautious_lion.py` (~30 LoC) subclassing `torch.optim.Lion`, mirroring `optimizers/cautious_muon.py` structure.
- `LLMConfig.use_cautious_lion: bool = False` next to existing flags.
- `training/trainer.py:142` — gate the swap where Muon/Lion/etc. are picked.
- Pass/fail: PASS ≤ −0.005 vs ctrl Lion. NULL = |Δ| < 0.005. DRIFT > +0.01.

## Notes
- Unlike 001/002, no Muon/AdamW hyperparameter conflict — Lion uses its own `lion_lr` separate from `adamw_lr`.
- Lion is *not* currently in our routing table per the code I last saw; if absent, this idea implicitly requires adding Lion too. Flag this at review.
