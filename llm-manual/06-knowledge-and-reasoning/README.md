# §06 — Knowledge & Reasoning: the *Physics of Language Models*

This section is mostly one research program: **"Physics of Language Models"** by **Zeyuan
Allen-Zhu** (then Meta FAIR) and collaborators (Yuanzhi Li, Tian Ye, others). It is the
closest thing the field has to *our* methodology applied to *understanding* rather than
*optimizing* — and it's the program Vuk named as the model for this whole effort.

## Why this program is the template for our work

Allen-Zhu's method is deliberately the opposite of benchmark-chasing:

- **Synthetic, fully-controlled data** where the ground-truth information content is *known in
  bits* — so you can measure what the model learned exactly, not approximately.
- **Controlled interventions** isolating one variable (data diversity, depth, quantization).
- **Universal laws** that hold across model sizes, stated with scope — not leaderboard wins.

That is the same epistemic stance as our measured rules (`L###/D###/C###`): measure a clean Δ, state its
scope, name its falsifier. The difference is altitude — they run at up to billions of params
on synthetic corpora; we run at 1–3M on real text. **The methods rhyme; that's the point.**

## The series map

| Part | Topic | Rule here |
|---|---|---|
| 1 | Learning hierarchical (CFG) language structure | — (background) |
| 2.1 | Grade-school math & the hidden reasoning process | [FM-06.3](FM-06.3-reasoning-process.md) |
| 3.1 | Knowledge storage & extraction | [FM-06.2](FM-06.2-knowledge-extraction-needs-augmentation.md) |
| 3.2 | Knowledge manipulation (the limits) | covered in FM-06.2 |
| 3.3 | Knowledge capacity scaling laws | [FM-06.1](FM-06.1-knowledge-capacity-2bits.md) |

## Rules in this section

| ID | Rule | Confidence |
|---|---|---|
| [FM-06.1](FM-06.1-knowledge-capacity-2bits.md) | A model stores ~2 bits of knowledge per parameter (a hard ceiling) | **[F]** |
| [FM-06.2](FM-06.2-knowledge-extraction-needs-augmentation.md) | Facts become *extractable* only if seen in varied phrasings during pretraining | **[F]** |
| [FM-06.3](FM-06.3-reasoning-process.md) | Models learn a genuine, probe-able reasoning process — not just templates | **[F]** |

These are tagged **[F] Frontier** not because the work is weak — it's exceptionally rigorous —
but because it rests largely on **one group's synthetic-data program**, not yet broadly
replicated by independent labs on independent setups. That is exactly the "single strong
source" bar for Frontier; independent replication on a different protocol would upgrade them.
The findings are among the most *actionable* in the manual; treat them as
high-value bets, and note where independent replication would upgrade them.
