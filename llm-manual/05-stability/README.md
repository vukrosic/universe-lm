# §05 — Stability: how to not diverge

A training run that spikes and diverges wastes everything spent so far. This section is the
field's standing toolkit for keeping large runs on the rails — and a rare case where the
mitigations are **directly testable at small scale**, because the instabilities can be
*provoked* in small models with high learning rates.

## Rules in this section

| ID | Rule | Confidence |
|---|---|---|
| [FM-05.1](FM-05.1-z-loss-qk-norm.md) | Control logit growth with z-loss + qk-norm to stabilize high-LR training | **[E]** |

## The mental model

Two recurring failure modes (Wortsman et al. 2023, "small-scale proxies"):
1. **Attention logit growth** — Q·K scores blow up → softmax saturates to one-hot → gradients
   vanish → divergence.
2. **Output logit divergence** — final logits drift from log-probabilities late in training.

Both **scale-couple with learning rate**: bigger models diverge at *lower* LRs, so a recipe
that's stable small can blow up large. Crucially, you can **reproduce both in small models** by
cranking the LR — which is exactly why this is the most *small-scale-friendly* section of the
manual, and the one our ledger could most plausibly contribute to.

## The toolkit (beyond FM-05.1)

z-loss · qk-norm (qk-layernorm) · softmax/logit capping · longer warmup · lower peak LR ·
higher weight decay · σReparam · LayerScale · careful precision (FP32 > BF16 for stability) ·
gradient clipping (FM-04.1). Newer work argues the growth affects *all* linear-layer outputs,
not just attention, and combines QKV-norm + softmax capping for higher stable LR.
