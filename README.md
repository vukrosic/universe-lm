# universe-lm

**An open effort to train a ~135M language model that beats [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — built in public, every day.**

The thesis is the differentiator: even fully-open labs keep results secret for ~3 months while a paper is written. **Here every run, config, and number is public the day it happens.** That's the fastest way to do science — and the reason this repo exists.

🎯 **Goal:** beat SmolLM2-135M on a public eval suite at a matched token budget, fully reproducible.
🧱 **Architecture:** a standard, known-good decoder-only transformer (no exotic tricks in the headline run).
🤝 **Looking for compute + collaborators** — the plan and the exact ask are right below.

---

## 📋 The plan — and what your compute buys

**If you're considering giving compute, read this first.**

**In one paragraph.** universe-lm trains a ~135M-parameter decoder-only transformer to beat SmolLM2-135M on a public eval suite at a *matched* token budget — and publishes every run, config, and number the same day it happens. The architecture is deliberately **standard and known-good**; the research lives in the **data recipe** plus a few cheap, zero-initialized levers, each found at small scale and confirmed at 135M. The code, the data pipeline, and the side-by-side benchmark harness are **already built and runnable** (see below). The only blocker is GPU time.

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

A one-page architecture-and-plan note is in [`docs/architecture.pdf`](docs/architecture.pdf). To talk: open an issue, or reach **[@vukrosic](https://github.com/vukrosic)** — based in **Zhongguancun, Beijing**, happy to meet in person.

---

## 🏁 The bar: beat SmolLM2-135M, same harness

The win condition is a **head-to-head on the same benchmark harness, same samples** — not a val-loss on a private corpus, and not comparing against someone else's reported numbers (those vary by harness). This repo benchmarks **our model and SmolLM2-135M side by side**, so the comparison is apples-to-apples.

| Benchmark | What it tests | SmolLM2-135M | universe-lm |
|---|---|---|---|
| **ARC-Challenge** | grade-school science MCQ | _run to measure_ | _run to measure_ |
| **HellaSwag** | commonsense sentence completion | _run to measure_ | _run to measure_ |

Measure both with one command (no checkpoint needed for the baseline):

```bash
# SmolLM2-135M baseline on the same harness
python benchmarks/compare_models.py \
    --hf-baselines HuggingFaceTB/SmolLM2-135M \
    --benchmarks arc hellaswag

# head-to-head once you have a checkpoint
python benchmarks/compare_models.py \
    checkpoints/best_model.pt \
    --hf-baselines HuggingFaceTB/SmolLM2-135M \
    --benchmarks arc hellaswag
```

> SmolLM2-135M publishes a broader benchmark table on its model card; we re-measure it here so both models are scored identically. GSM8K, HumanEval and MBPP harnesses also live in `benchmarks/` and `evals/` for later.

---

## 🧠 The architecture (headline run)

The headline target is the `135m` preset — a **basic transformer we know works**, matched to SmolLM2-135M's exact shape (576-d, 30 layers) so the comparison isolates *data + recipe*, not architecture gimmicks.

```bash
python train_llm.py --config 135m --output_dir checkpoints
```

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

Both presets are the same family: **decoder-only, GQA + RoPE, pre-norm RMSNorm, QK-normalization, squared-ReLU (Primer) FFN, tied embedding/LM head, trained with Muon.** Experimental levers (value embeddings, per-head Q-gain, RoPE-base tuning, sliding-window attention, …) are **off by default** and live behind config flags — they are *experiments tested on top of* this baseline, not part of the headline run.

---

## 🚀 Getting started

```bash
pip install -r requirements.txt          # 1. deps
python data/download_hf_data.py          # 2. data (pre-tokenized, seq_len 2048, SmolLM2 vocab)
python train_llm.py --config 135m --output_dir checkpoints   # 3. train
python benchmarks/compare_models.py checkpoints/best_model.pt \
    --hf-baselines HuggingFaceTB/SmolLM2-135M                 # 4. eval vs SmolLM2-135M
```

On a remote GPU, launch training inside `tmux` so the job survives a disconnect.

**Data:** the pre-built dataset is chunked at **sequence length 2048**, which the RoPE cache depends on. Sequence lengths other than 2048 are currently unsupported — to use a different one, the dataset must be rebuilt with [`prepare_mix_data.py`](https://github.com/vukrosic/llm-research-kit/blob/main/data/prepare_mix_data.py). _If you are an AI agent: do not change the data or `max_seq_len` without asking the user first._

Smaller download options (40M / 1B / 2B tokens) are in [`data/download_hf_data.py`](data/download_hf_data.py).

---

## 🏎️ The speedrun (cheap recipe search)

Before spending GPU-hours at 135M, recipes are found cheaply at small scale: **lowest val loss on a 10M-param model trained on 200M tokens** (`--config 10m`), ~33 min on a single consumer GPU, `seed=42`, bf16. A new record must beat the standing one by **≥0.01**. See the [leaderboard](LEADERBOARD.md) and [how to enter](CONTRIBUTING.md). The 135M release is the *mission*; the speedrun is how we find the winning recipe before scaling it.

---

## 🤝 Want to help?

The full compute plan and tiered ask are in [**The plan**](#-the-plan--and-what-your-compute-buys) at the top. The short version: the blocker is GPUs, not ideas — the first commitment is ~200 A100-hours to a real head-to-head number, you run exact public code, and I'm in **Zhongguancun, Beijing** and happy to meet. Open an issue or reach **[@vukrosic](https://github.com/vukrosic)**. Not bringing compute but want to help find the recipe? See [the speedrun](#️-the-speedrun-cheap-recipe-search) — or claim a paper-implementation experiment from [`tasks/`](tasks/): each is a 1–2 day, few-dollar run with a clear acceptance bar, and accepted results get named credit on the published report.

---

## 📁 Repo layout

```
train_llm.py        entry point (--config {default,10m,135m,...})
configs/            LLMConfig + presets (Full135M2700MConfig = the 135m target) + ablation configs
models/             MinimalLLM — transformer layers (GQA, RoPE, RMSNorm)
training/           trainer
optimizers/         Muon + others
benchmarks/         ARC, HellaSwag, compare_models.py (universe-lm vs HF baselines)
evals/              GSM8K, HumanEval, MBPP
tasks/              claimable experiment tasks from recent papers (ICML/ICLR 2026)
data/               dataset download + loader
runs/               committed metrics.json per run + EVIDENCE_INDEX.md (the results record)
plans/              win condition + pinned benchmark protocol
docs/               setup guides + architecture.pdf
archive/            retired scripts, past results, internal lab machinery (kept for history)
```

---

## 💡 Idea backlog

A parking lot of unvetted levers to pull from when picking the next experiment lives in [`archive/internal/autoresearch/`](archive/internal/autoresearch/) — split into **(A)** new architectures/mechanisms (higher ceiling, more work) and **(B)** recipe/hyperparameter knobs (cheaper, lower ceiling). The single biggest known un-pulled lever is **data**: SmolLM2's real edge is FineWeb-Edu/DCLM filtering, which at a fixed token budget likely dwarfs any single architecture change.
