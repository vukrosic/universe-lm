---
id: 091-scale-anchor-loss
status: needs-plan
round: 1
updated: 2026-06-11T01:17:52Z
transfer-risk: low
---

# 091 — Scale anchoring auxiliary loss

## Source
Same source as TaperNorm: Kanavalau, Amo Alonso, and Lall, "Gated Removal of Normalization in Transformers Enables Stable Training and Efficient Inference" (arXiv:2602.10408). The paper argues that an explicit target on the pre-logit residual scale can substitute for some of the work done by output normalization.

## Mechanism
Keep the transformer stack unchanged, but add a small auxiliary penalty that pulls the residual stream's norm toward a fixed target right before the logits. This is an anchor, not a normalization layer: it nudges the model away from logit chasing without computing per-token statistics inside the block.

## Scale evidence
The source paper treats this as a supporting mechanism for final-norm removal and uses it to explain why output normalization matters so much. That makes it a cheap lever with high diagnostic value: if a tiny scalar anchor moves loss, then the model is sensitive to radial control at the very end.

## Why it's worth a slot
This is the smallest possible test of the "scale anchoring" hypothesis. It costs almost nothing, is easy to fold into the existing training loop, and tells us whether the final normalization site matters because of statistics or because of a simple radius target.
