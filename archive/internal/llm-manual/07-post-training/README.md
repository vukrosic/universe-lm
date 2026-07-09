# §07 — Post-training: from base model to assistant

Pretraining (§01–§06) produces a model that *completes text*. Post-training makes it *follow
instructions and behave*. This is a different game — smaller data, preference signals, and a
fast-moving method landscape — but the high-level pipeline has stabilized.

## Rules in this section

| ID | Rule | Confidence |
|---|---|---|
| [FM-07.1](FM-07.1-sft-then-preference.md) | SFT first, then preference optimization (DPO or RLHF) | **[E]** |

## The pipeline at a glance

```
base model ──SFT──▶ instruction-following model ──preference opt──▶ aligned assistant
            (imitate                              (DPO / RLHF / GRPO:
             good answers)                         prefer better over worse)
```

This is mostly **out of our repo's current scope** — the ledger is a *pretraining* structural
lab. Post-training is documented here for completeness so the manual is a full
training-an-LLM reference, not because we measure it. One connection worth noting (FM-06.2):
*you cannot fine-tune in knowledge that pretraining didn't make extractable* — post-training
surfaces and shapes behavior, it does not implant new facts.
