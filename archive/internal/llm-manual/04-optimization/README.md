# §04 — Optimization: optimizer, schedule, batch, weight decay, μP

The hyperparameters that decide whether your allocated compute (§01) actually converts into a
low loss. The good news: this is the most *settled* recipe in the manual — the LLaMA-lineage
defaults below are a safe starting point at almost any scale. The hard part is **transfer**:
you cannot afford to tune these at full scale, so the real skill is tuning small and
projecting up (μP, FM-04.4).

## Rules in this section

| ID | Rule | Confidence |
|---|---|---|
| [FM-04.1](FM-04.1-adamw-defaults.md) | AdamW with β=(0.9, 0.95), wd 0.1, grad-clip 1.0 is the default | **[E]** |
| [FM-04.2](FM-04.2-lr-schedule.md) | Warmup then decay (cosine→10%, or linear→0) tuned to run length | **[E]** |
| [FM-04.3](FM-04.3-batch-size-weight-decay.md) | Batch size and weight decay scale with *data*, not model size | **[C]** |
| [FM-04.4](FM-04.4-mup-transfer.md) | μP lets you transfer hyperparameters from small to large models | **[C]** |

## ⚠️ Scope note for our ledger

The ledger operates under **RULE 0: structural levers only — never optimizer / LR / batch /
weight decay.** This section is therefore *deliberately outside* what our measured rules
cover: it
documents the field's optimizer recipe so we know the **fixed background** our structural
experiments run against. Do not mine these knobs in the ledger; hold them at these defaults so
a structural Δ measures the structure, not a re-tuned optimizer.

(One historical exception lives in a draft: `tiny-update-starvation` found LR×momentum
compound super-additively at our tiny scale — recorded, then the axis was *closed* per RULE 0.
That episode is *why* the rule exists.)
