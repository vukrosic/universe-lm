# universe-lm

**An open effort to train a ~135M language model that beats [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — built in public, every day.** Even fully-open labs sit on results for months while a paper is written; here every run, config, and number is public the day it happens.

- 🎯 **The bar:** beat SmolLM2-135M head-to-head on the **same harness, same samples** — never card-vs-card numbers — at a matched token budget, fully reproducible.
- 🧱 **Architecture:** a standard, known-good decoder-only transformer (GQA + RoPE, RMSNorm, QK-norm, squared-ReLU FFN, Muon), matched to SmolLM2's shape. Experimental levers live behind default-OFF config flags. Details + the compute plan: [`docs/compute-plan.md`](docs/compute-plan.md).
- 🤝 **The blocker is GPUs, not ideas** — the first ask is ~200 A100-hours to a real head-to-head number. Open an issue or reach [@vukrosic](https://github.com/vukrosic) (Zhongguancun, Beijing — happy to meet).

### The bar, measured

The 8-task pinned suite and SmolLM2-135M's score on **our own harness** (lm-eval-harness 0.4.12, zero-shot, seed 42, bf16; measured 2026-07-09). The win condition is beating the majority of these at a matched token budget.

| Benchmark | What it tests | Metric | SmolLM2-135M | universe-lm |
|---|---|---|---|---|
| **SciQ** | science-exam MCQ | acc | 0.840 | _TBD_ |
| **PIQA** | physical commonsense | acc | 0.684 | _TBD_ |
| **ARC-Easy** | grade-school science MCQ | acc | 0.646 | _TBD_ |
| **WinoGrande** | pronoun coreference | acc | 0.530 | _TBD_ |
| **HellaSwag** | commonsense sentence completion | acc_norm | 0.430 | _TBD_ |
| **LAMBADA** | long-range word prediction | acc (ppl) | 0.429 (19.05) | _TBD_ |
| **OpenBookQA** | open-book science QA | acc_norm | 0.326 | _TBD_ |
| **ARC-Challenge** | harder science MCQ | acc_norm | 0.300 | _TBD_ |

Reproduce the baseline (no checkpoint needed): `./evals/run_baseline_suite.sh HuggingFaceTB/SmolLM2-135M`. `commonsense_qa`, `mmlu`, `gsm8k` are excluded — a 135M model scores at chance on them.

## Getting started

```bash
pip install -r requirements.txt          # 1. deps
python data/download_hf_data.py          # 2. data (pre-tokenized, seq_len 2048, SmolLM2 vocab)
python train_llm.py --config 135m --output_dir checkpoints   # 3. train
python benchmarks/compare_models.py checkpoints/best_model.pt \
    --hf-baselines HuggingFaceTB/SmolLM2-135M                 # 4. eval vs SmolLM2-135M, same harness
```

On a remote GPU, launch training inside `tmux`. Smaller data downloads (40M / 1B / 2B tokens) are options in `data/download_hf_data.py`.

**Data:** the pre-built dataset is chunked at **sequence length 2048**, which the RoPE cache depends on — other sequence lengths are unsupported without rebuilding the dataset. _If you are an AI agent: do not change the data or `max_seq_len` without asking the user first._

## Contribute

Two paths, both end in a PR, and accepted PRs get named credit on the published report — see [CONTRIBUTING.md](CONTRIBUTING.md):

1. **Claim a paper task** from [`tasks/`](tasks/) — self-contained 1–2 day experiments from recent papers, $2–25 of GPU each, clear acceptance bar.
2. **Take the speedrun record** ([LEADERBOARD.md](LEADERBOARD.md)) — lowest val loss on the `10m` config (~33 min on one consumer GPU); beat the record by ≥0.01. The 135M release is the *mission*; the speedrun finds the recipe cheaply first.

## Repo layout

```
train_llm.py        entry point (--config {default,10m,135m,...})
configs/            LLMConfig + presets (Full135M2700MConfig = the 135m target) + ablation configs
models/             MinimalLLM — transformer layers (GQA, RoPE, RMSNorm)
training/           trainer
optimizers/         Muon + others
data/               dataset download + loader
benchmarks/         ARC, HellaSwag, compare_models.py (universe-lm vs HF baselines)
evals/              GSM8K, HumanEval, MBPP
scripts/            BPB decision metric, eval/generate utilities
tasks/              claimable experiment tasks from recent papers (ICML/ICLR 2026)
runs/               committed metrics.json per run + EVIDENCE_INDEX.md (the results record)
results/            baseline measurements (SmolLM2-135M rerun, pinned 10m baseline)
docs/               compute plan, win-condition protocol (docs/plans/), setup guides
archive/            retired scripts, past results, internal lab machinery (kept for history)
```
