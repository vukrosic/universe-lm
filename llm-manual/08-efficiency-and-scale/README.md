# §08 — Efficiency & scale: decoupling capacity and cost from compute

The recipe in §03 is the *dense, short-context, BF16* default. Frontier models add three
levers that **break the coupling** between what a model can do and what it costs to train/run:

- **MoE** decouples *capacity* (total params) from *per-token compute* (active params).
- **Low precision** (BF16 → FP8) decouples *throughput* from *parameter count*.
- **Context extension** decouples *usable sequence length* from *training length*.

These are the engineering that the §00 map flagged as "decoupling model capacity from inference
cost." None is mandatory for a first model; all three are how you get a *competitive* one.

## Rules in this section

| ID | Rule | Confidence |
|---|---|---|
| [FM-08.1](FM-08.1-mixture-of-experts.md) | MoE buys capacity at near-constant compute; many small experts beat few big ones | **[C]** |
| [FM-08.2](FM-08.2-precision-bf16-fp8.md) | BF16 is the safe default; FP8 works with fine-grained scaling + high-precision accumulation | **[C]** |
| [FM-08.3](FM-08.3-long-context-extension.md) | Extend context by interpolating RoPE (PI → NTK → YaRN), then fine-tune | **[C]** |

## Not yet covered (honest gaps)

This manual does **not** yet have first-class sections for **inference/serving** (KV-cache
management, PagedAttention, speculative decoding, continuous batching) or **evaluation**
(benchmark design, contamination, the metric-choice problem — see FM-01.4 for the one piece
that's here). These are real decisions in shipping an LLM; they're flagged here rather than
padded with unverified content. Candidates for the next pass.

## Relation to our ledger

Out of scope for our 1–3M structural runs — these are scale-and-systems levers, not tiny-model
mechanisms. One exception with a thread to us: **FM-06.1** shows MoE keeps most of its
knowledge capacity despite sparse activation, so MoE is "cheap capacity for facts." That
*capacity* framing is testable in principle at small scale with synthetic data, but it's far
from our current real-text structural frame — background, not an action item.
