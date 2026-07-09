# §03 — Architecture: the convergent recipe

The most striking fact about modern LLM architecture is that **the field converged.** Llama,
Mistral, Gemma, Qwen, Phi, DeepSeek — independently — settled on the same decoder-only
transformer with the same four swaps over the 2017 original. That convergence is the
strongest available evidence that, for this generation, the architecture is *roughly solved*.

This section documents the recipe and **why each piece is there** — because "why" is exactly
what tells you when a piece might *not* apply (e.g. at our tiny scale, or for non-text data).

## Rules in this section

| ID | Rule | Confidence |
|---|---|---|
| [FM-03.1](FM-03.1-convergent-recipe.md) | Decoder-only + pre-RMSNorm + RoPE + SwiGLU + GQA is the converged default | **[E]** |

(Each component below is a subsection of FM-03.1 rather than its own file, because they ship
as a bundle and are rarely adopted in isolation. Split out later if a component earns its own
contested debate.)

## The recipe at a glance

| Component | 2017 original | Modern default | Primary reason for the swap |
|---|---|---|---|
| Norm type | LayerNorm | **RMSNorm** | ~10–50% cheaper, no quality loss |
| Norm placement | Post-norm | **Pre-norm** | stable gradients at init → trains at depth |
| Positional | sinusoidal absolute | **RoPE** | relative + extrapolates to longer context |
| FFN activation | ReLU | **SwiGLU** (gated) | better quality at equal FLOPs |
| Attention | Multi-Head | **GQA** | 4–8× smaller KV cache, ~free quality |
| Objective | — | next-token prediction | unchanged |

## Why this section matters most to *us*

Our measured ledger (the `L###/D###/C###` entries + [`../drafts/`](../drafts/)) is a
structural-mechanism lab at 1–3M params. This section is the
field's baseline for the exact loci we mine: positional (RoPE vs our learnable-ALiBi
champion), normalization, FFN, attention output. **The deepest open question we can answer:
do these >7B-validated structural choices keep their sign at 1–3M, and does anything we find
that beats the recipe at tiny scale survive scaling up?** Several ledger drafts already probe
the positional and residual loci against exactly these defaults.
