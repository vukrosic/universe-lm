# The plan — and what your compute buys

**If you're considering giving compute, read this first.**

**In one paragraph.** universe-lm trains a ~135M-parameter decoder-only transformer to beat SmolLM2-135M on a public eval suite at a *matched* token budget — and publishes every run, config, and number the same day it happens. The architecture is deliberately **standard and known-good**; the research lives in the **data recipe** plus a few cheap, zero-initialized levers, each found at small scale and confirmed at 135M. The code, the data pipeline, and the side-by-side benchmark harness are **already built and runnable**. The only blocker is GPU time.

**What we're doing, concretely:**
1. **Re-measure SmolLM2-135M on our exact harness** — so the comparison is apples-to-apples, not card-vs-card. (Ready today; one command.)
2. **Train the `135m` preset** on the open data pipeline and score it **head-to-head** against that baseline.
3. **Search recipes cheaply at 10M params**, then promote only the winners up to 135M (overnight runs, not month-long shots).
4. **Lock the headline number** with confirmation runs + ablations, every artifact public.

**The compute, tiered (≈1,200–1,500 A100-hours total, estimates refined against the real cluster):**

| Phase | What runs on the GPUs | Est. A100-h | Decision gate |
|---|---|---|---|
| **0 — Pilot** | validate the pipeline end-to-end + first full 135M run, scored vs SmolLM2-135M on the same harness | **~200** | a real head-to-head number → go / no-go |
| **1 — Data** | FineWeb-Edu / DCLM-style filtering vs the current mix at 135M — the single biggest lever | ~400 | best data recipe locked |
| **2 — Levers** | confirm the cheap, zero-init levers (value embeddings, per-head Q-gain, RoPE base, sliding window) at 135M | ~400 | which levers actually help |
| **3 — Lock-in** | best-recipe full runs + ablation confirmations to publish a reproducible win | ~400 | the headline result |

**The first commitment is only ~200 hours** — enough to see a real, reproducible number before you scale further. Every hour runs **exact public code** in this repo: nothing hidden, nothing proprietary, fully reproducible, published daily. 135M fits on a single GPU, so it's plain **single-node data parallelism** (DDP/FSDP) — no tensor/pipeline setup, any A100/H100 box works.

A one-page architecture-and-plan note is in [`architecture.pdf`](architecture.pdf). To talk: open an issue, or reach **[@vukrosic](https://github.com/vukrosic)** — based in **Zhongguancun, Beijing**, happy to meet in person.

## The headline architecture

The headline target is the `135m` preset — a **basic transformer we know works**, matched to SmolLM2-135M's exact shape (576-d, 30 layers) so the comparison isolates *data + recipe*, not architecture gimmicks.

| | Default (`default`) | Headline target (`135m`) |
|---|---|---|
| Parameters | ~88M | **~134.5M** |
| `d_model` | 512 | 576 |
| Layers | 22 | 30 |
| Attention | 8 query / 4 KV (GQA) | 9 query / 3 KV (GQA) |
| `d_ff` | 2048 | 2304 (4× `d_model`) |
| Vocab | 49,152 (SmolLM2 tokenizer) | 49,152 |
| Context | 2048 | 2048 |
| Token budget | — | ~2.7B (Chinchilla ~20×) |

Both presets are the same family: **decoder-only, GQA + RoPE, pre-norm RMSNorm, QK-normalization, squared-ReLU (Primer) FFN, tied embedding/LM head, trained with Muon.** Experimental levers (value embeddings, per-head Q-gain, RoPE-base tuning, sliding-window attention, …) are **off by default** behind config flags — *experiments tested on top of* this baseline, not part of the headline run.

The win condition and pinned benchmark protocol live in [`plans/`](plans/). GSM8K, HumanEval and MBPP harnesses live in `benchmarks/` and `evals/` for later.
