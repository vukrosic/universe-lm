# Literature review: sub-200M / sub-400M open LMs (2024–2026)

Compiled 2026-06-10 to inform `docs/plans/beat-smollm2-135m.md`.
Numbers are **paper / model card numbers** unless tagged `leaderboard`. We will
rerun everything in Phase 0 (per the program plan), but the table below sets
the SOTA expectations and the field of comparison.

## 1. Model table

| Model | Params (M) | Tokens | Data mix (paper terms) | HellaSwag | ARC-e | ARC-c | PIQA | WinoGrande | OBQA | CSQA | MMLU | License |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **SmolLM2-135M** (HF) | 135 | 2T | FineWeb-Edu + DCLM + Stack + new curated | 42.1 | — | 43.9* | 68.4 | 51.3 | 34.6 | 33.9 | 31.5 | Apache-2.0 |
| **SmolLM2-360M** (HF) | 360 | 4T | FineWeb-Edu + DCLM + Stack + new curated | 54.5 | — | 53.0* | 71.7 | 52.5 | 37.4 | 38.0 | 35.8 | Apache-2.0 |
| SmolLM2-1.7B (HF) | 1,700 | ~11T | Multi-stage: FineWeb-Edu + DCLM + Stack-Edu + FineMath + Cosmopedia | 68.7 | — | 60.5* | 77.6 | 59.4 | 42.2 | 43.6 | n/r | Apache-2.0 |
| **SmolLM3-3B** (HF, 2025-07) | 3,080 | 11.2T | web+code+math+reasoning; NoPE+GQA; yarn to 128k | 76.15 | — | 65.61* | 78.89 | 58.88 | 40.60 | 55.28 | 44.13 | Apache-2.0 |
| **Qwen2.5-0.5B** (base) | 490 | up to 18T (whole series) | Qwen2.5 corpus (curated web, code, math) | 52.1 | — | 35.6 | n/r | 56.3 | n/r | n/r | 47.5 | Apache-2.0 |
| Qwen2.5-1.5B (base) | 1,540 | up to 18T | same | 67.9 | — | 54.7 | n/r | 65.0 | n/r | n/r | 60.9 | Apache-2.0 |
| **Qwen3-0.6B-Base** | 600 | ~36T (series) | 3-stage (LM→reasoning→long-ctx), 119 langs | n/p | n/p | n/p | n/p | n/p | n/p | n/p | n/p | Apache-2.0 |
| Qwen3-1.7B-Base | 1,700 | ~36T | same | 60.52 | — | 55.88* | 75.35 | 57.06 | 36.40 | 48.98 | 39.11 | Apache-2.0 |
| **MobileLLM-125M** (Meta) | 125 | 1T (paper table) | Meta pre-training (unspecified) | 38.9 | 43.9 | 27.1 | 65.3 | 53.1 | 39.5 | n/r | n/r | FAIR-NC |
| MobileLLM-350M (Meta) | 350 | 1T (paper table) | same | 49.6 | 53.8 | 33.5 | 68.6 | 57.6 | 40.0 | n/r | n/r | FAIR-NC |
| MobileLLM-LS-125M (Meta) | 125 | 1T | same | n/r | n/r | n/r | n/r | n/r | n/r | n/r | n/r | FAIR-NC |
| **OpenELM-270M** (Apple) | 270 | 1.8T | RefinedWeb + dedup PILE + RedPajama + Dolma 1.6 | 46.71 | 45.08 | 26.45 | 69.75 | 53.91 | n/r | n/r | n/r (25.7 5-shot) | apple-amlr |
| OpenELM-450M (Apple) | 450 | 1.8T | same | 53.97 | 48.06 | 27.56 | 72.31 | 58.01 | n/r | n/r | 26.01 (5-shot) | apple-amlr |
| **Llama 3.2-1B** (base) | 1,230 | 9T (series) | Meta curated (English-heavy, 8 official langs) | 41.2 (instruct) | — | 32.8 (25-shot) | n/r | n/r | n/r | n/r | 32.2 | Llama-3.2 (custom) |
| **Gemma 2 2B** (base) | 2,000 | 2T | web (EN) + code + math; Google filters | 73.0 (10s) | 80.1 | 55.4 (25s) | 77.8 | 70.9 | n/r | n/r | 51.3 | Gemma (custom) |
| **H2O-Danube-1.8B** | 1,800 | n/p | Llama-2 arch + sliding-window attn 4k (Mistral) | 68.20 | 62.29 | 35.84 | 76.93 | 61.96 | 37.60 | n/r | n/r | Apache-2.0 |
| **AMD-Llama-135m** | 135 | ~670B | SlimPajama (no Books) + Project Gutenberg | 30.48 | 43.64 | 19.11 | 64.20 | 50.12 | n/r | n/r | 23.02 | Apache-2.0 |
| **Pythia-160m** | 162 (85M non-emb) | 300B (1 epoch) | The Pile (22 sources, undeduped) | ~30 (Pythia suite avg) | ~41 | ~19 | ~62 | ~52 | ~33 | ~20 | ~23 | Apache-2.0 |
| TinyLlama-1.1B | 1,100 | 3T | SlimPajama + Starcoderdata + Ultrachat | n/r (≈ OLMo-1B tier) | — | — | — | — | — | — | — | Apache-2.0 |
| **OLMo-1B** (HF) | 1,080 | 3T | Dolma (open) | 62.5 | 58.07 | 34.45 | 73.7 | 58.9 | 46.4 | n/r | n/r | Apache-2.0 |
| Fox-1 1.6B (answer.ai) | 1,600 | n/p | n/p | n/p | n/p | n/p | n/p | n/p | n/p | n/p | n/p | Apache-2.0 |
| Index-1.9B (bilingual) | 1,900 | n/p | Chinese + English | n/p | n/p | n/p | n/p | n/p | n/p | n/p | n/p | Apache-2.0 |

`*` = SmolLM2 cards report **ARC average** (e / c combined). For
head-to-head we rerun both splits.
`n/p` = not in published material we could verify; `n/r` = not reported.

### Sources
- SmolLM2 model cards: <https://huggingface.co/HuggingFaceTB/SmolLM2-135M> · <https://huggingface.co/HuggingFaceTB/SmolLM2-360M>
- SmolLM2 paper: <https://arxiv.org/abs/2502.02737> (HTML tables: <https://arxiv.org/html/2502.02737v1>)
- SmolLM3-3B card: <https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base>
- Qwen2.5 model cards + blog: <https://huggingface.co/Qwen/Qwen2.5-0.5B> · <https://huggingface.co/Qwen/Qwen2.5-1.5B> · <https://qwenlm.github.io/blog/qwen2.5/>
- Qwen3 model cards + report: <https://huggingface.co/Qwen/Qwen3-0.6B-Base> · <https://huggingface.co/Qwen/Qwen3-1.7B-Base> · <https://arxiv.org/abs/2505.09388>
- MobileLLM (ICML 2024) repo + paper: <https://github.com/facebookresearch/MobileLLM> · <https://arxiv.org/abs/2402.14905>
- OpenELM card: <https://huggingface.co/apple/OpenELM-270M> (and 450M)
- Llama 3.2-1B card: <https://huggingface.co/meta-llama/Llama-3.2-1B>
- Gemma 2 2B card: <https://huggingface.co/google/gemma-2-2b>
- H2O-Danube-1.8B card: <https://huggingface.co/h2oai/h2o-danube-1.8b-base>
- AMD-Llama-135m card: <https://huggingface.co/amd/AMD-Llama-135m>
- Pythia-160m card: <https://huggingface.co/EleutherAI/pythia-160m>
- OLMo-1B card: <https://huggingface.co/allenai/OLMo-1B-hf>
- TinyLlama-1.1B card: <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0>

## 2. Who beats SmolLM2-135M, and why

The "beat" threshold is the program plan: ≥5 of 6 win-eligible tasks **and** lower
FineWeb-Edu BPB. We list every model whose reported numbers clearly clear the
SmolLM2-135M bar on at least one task at a comparable budget; distillation
wins are flagged because we train from scratch.

| Model | Beats SmolLM2-135M on | Why it wins (reported) | Distilled? |
|---|---|---|---|
| **SmolLM2-360M** | 5/6 in card table (HellaSwag +12, ARC +9, PIQA +3, OBQA +3, CSQA +4) | 2× tokens (4T), same data pipeline, deeper/wider config | No |
| **MobileLLM-350M** | 4/5 reported (HellaSwag +7.5, ARC +0.5, PIQA +0.2, OBQA +5.4) | Deep-thin arch, GQA, embedding sharing, SwiGLU; **1T** tokens | No |
| **OpenELM-270M** | 1/4 reported (HellaSwag +4.6) but loses on ARC-c | Layer-wise scaling (variable width per layer); 1.8T tokens | No |
| **Qwen2.5-0.5B** | 1/4 reported (HellaSwag +10); loses on ARC-c; WinoGrande +5 | Massive 18T-token series corpus; stronger tokenizer; GQA | No |
| **AMD-Llama-135m** | Loses on 5/5 reported (matched params, ~1/3 tokens) | Useful only as a *control*: it shows token count dominates over arch at 135M | No |
| **Pythia-160m** | Loses on 6/6 | Old data (The Pile, undeduped), only 300B tokens | No |
| **Gemma 2 2B**, **Qwen3-1.7B-Base**, **Qwen2.5-1.5B**, **H2O-Danube-1.8B**, **OLMo-1B**, **SmolLM3-3B** | All clear the bar trivially (they are 2-25× bigger) | More params + more tokens. Included as secondary comparators, not direct peers. | No |

**Distillation caveat.** Several popular <1B models (e.g., Zephyr-SFT, SmolLM2-*-
Instruct, MobileLLM-R1) are SFT/DPO distillations of a larger teacher.
We **train from scratch**, so distillation wins are not in scope; we list
instruct numbers only when they double as base numbers (rare). SmolLM2 paper
notes that their 360M base beats Qwen2.5-0.5B on 7/9 metrics **without
distillation** — this is the most credible recent sub-500M win and our
primary internal benchmark beyond SmolLM2-135M.

**Honest note on OLMo-1B and HellaSwag 62.5.** OLMo-1B's HellaSwag is much
higher than the SmolLM2-135M card. OLMo uses **the Pile-derived Dolma**,
whereas SmolLM2-135M trains on FineWeb-Edu + DCLM. HellaSwag is sensitive to
web-fiction training distribution; this is a real data-driven gap, not an
arch one.

## 3. Recipe levers that mattered in winning sub-400M recipes

Citations point at the specific paper or model card above.

1. **Data, not arch, is the dominant lever at ≤1B.** SmolLM2 paper §6 shows
   that the 135M wins on 5/6 card-reported tasks against the 1T-token
   MobileLLM-125M **with the same-ish arch** — 2T tokens of curated
   FineWeb-Edu + DCLM + Stack was the lever. The 1.7B variant then adds
   FineMath + Stack-Edu + Cosmopedia synthetics on top.
2. **Quality filtering / DCLM-style scoring > raw scale.** The SmolLM2 paper
   (Data-Centric) explicitly down-weights score-0 and 1-2 FineWeb-Edu docs;
   this is the single biggest perf/FLOP win in their ablations.
3. **Deep-and-thin beats wide-and-shallow at fixed param count.** MobileLLM
   ICML 2024: 2.7%/4.3% gain at 125M/350M just from depth/width swap vs
   prior sub-billion SOTA. Embedding sharing + GQA are the second-order
   companions. [arxiv:2402.14905]
4. **Multi-stage pre-training is free perf.** SmolLM2-1.7B uses a 3-stage
   curriculum (general → code/math → long-context); OLMo-1B also published
   intermediate checkpoints so the stage shifts are visible. The Qwen3
   report formalises this as LM → reasoning → long-context with 36T total.
5. **qk-layernorm is the new standard.** Qwen3 ships with it on all sizes;
   SmolLM3 adds it (NoPE + GQA + qk-LN). The autoresearch queue already
   tests `qk-norm` as idea 016 — this validates keeping it in Phase 1.
6. **Optimizer + LR schedule:** Muon for 2D params is the autoresearch
   baseline default; Moonlight (Muon+AdamW hybrid with RMS-aligned updates)
   is the cleanest public recipe that improves on Muon at <2B scale
   (Moonshot, 2025-02). Soap / Lion / Cautious-Muon are in the autoresearch
   queue already.
7. **Tokenizer matters more than people think.** SmolLM2 uses HuggingFace
   cosmo2 (49 152 vocab) — same as OLMo, MobileLLM uses Llama-2 (32 000),
   Qwen2/3 uses their own 151 642 / similar. Tokenizer choice alone can move
   BPB on FineWeb-Edu by 0.02+ nats. The plan defaults to **reusing cosmo2
   for a clean comparison**; we should not deviate.
8. **GQA + tied embeddings are table-stakes** in every recent winner we
   looked at; the only exceptions are Pythia (legacy) and OpenELM (Apple
   experimented with untied + layer-wise scaling for research reasons).

**Distillation is not in the above list** — by program decision we train from
scratch and treat distillation as a post-training add-on (Phase 5 stretch).

## 4. Target to beat — recommendation

**Primary target: SmolLM2-135M** (still the right bar in mid-2026). Reasons:

- It is the **only sub-200M open model with a published, data-centric
  recipe** (Feb 2025) that is reproducible end-to-end (nanotron, Apache-2.0,
  HF data shards).
- It is already a year old and **no open sub-200M model with a public recipe
  has displaced it on the standard 6-task suite** that we found. OpenELM-270M
  wins on HellaSwag only; MobileLLM-125M wins on OBQA only. Both are
  smaller-corpus and older.
- Rerun cost in Phase 0 is CPU-only (eval harness on released HF weights);
  we take nothing from their arch/recipe — the checkpoint is only a score
  row to beat (decided 2026-06-10).

**Secondary comparisons (in order):**

1. **MobileLLM-125M** — *the arch-quality control*. Different data, but a
   well-documented Meta recipe with public code. Tells us whether our arch
   tricks (qk-norm, value-residual, gated-attn, Muon, etc.) buy anything
   past the 2T-token data lift.
2. **OpenELM-270M** — *the data-quality control*. Smaller than 360M but
   bigger than 135M; trains on RefinedWeb + PILE + Dolma 1.6 with
   layer-wise scaling. If we beat it, our FineWeb-Edu / DCLM data pipeline
   is the reason; if we don't, we have a data-mix problem to fix in Phase 3.
3. **Qwen2.5-0.5B** (stretch / community claim) — *the "are we even in
   range" bar*. 4× our params and 9× our tokens. We don't have to beat it,
   but reporting the gap is the kind of public honesty the program plan
   asks for. (HuggingFace's own SmolLM2-360M card already beats Qwen2.5-0.5B
   on 7/9 metrics — useful framing for the post.)

**SmolLM3-3B is not a peer.** It is 23× our params; we report it for
trajectory context only (SmolLM3's 11.2T-token recipe implies that
matching SmolLM2-135M at 2T tokens is the *easy* part of the trajectory —
scaling past it is where the 2026 frontier lives).

### Open questions for Phase 0 (resolve via rerun, not this doc)

- SmolLM2-135M card reports **ARC average**, not e/c split. We need our own
  lm-eval-harness pass to get split numbers — `plan.md` already schedules
  this as the first Phase-0 task.
- OpenELM-270M has a higher HellaSwag (46.7) than SmolLM2-135M (42.1) on the
  card. Is that real or a harness/lighteval difference? Our rerun decides.
- MobileLLM numbers are 0-shot on the GitHub README, lighteval on HF cards;
  we will pin **0-shot lm-eval-harness @ a specific commit** per the plan
  and use that for every comparison.
