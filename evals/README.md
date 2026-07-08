# evals — model evaluation

Two things live here:

1. **`lm_eval_universe.py`** — the main harness. Wraps any lab checkpoint as
   an EleutherAI lm-evaluation-harness model, so every standard benchmark is
   scored with the *same code the SmolLM2-135M numbers were published with*.
   This is what the flagship "beat SmolLM2" verdict uses. Prefer it over the
   hand-rolled scripts below and in `benchmarks/` for anything it covers.
2. **`humaneval.py` / `mbpp.py`** — legacy standalone coding pass@1 scripts,
   kept for the code-execution path until `lm_eval` code tasks are validated.

## Two-tier philosophy (important)

Lab models are 23M–135M params on ~1B tokens. Codegen/math/graduate-bio
abilities emerge much later, so we split the suite:

| Suite | Tasks | At lab scale | Purpose |
|---|---|---|---|
| `core` | hellaswag, arc_easy/challenge, piqa, winogrande, openbookqa, boolq, commonsense_qa | above random — the real signal | flagship verdict vs SmolLM2-135M |
| `bio`  | mmlu college/hs biology, anatomy | ~25% (random floor) | curve exists from day one |
| `math` | gsm8k (5-shot) | ~0 | aspirational; tracked |
| `code` | humaneval pass@1 (executes code) | ~0 | aspirational; **box-only** |

Only `core` should drive promotion decisions right now. The others are wired
so the score curves start accumulating before the models are big enough to
move them.

## Local-first, box is disposable

Run and store evals **locally** (this is the source of truth); the GPU box is
interruptible and replaceable. Typical loop: box trains → we pull the
checkpoint (or its `model.pt`) down → eval locally on MPS/CPU → results commit
to git → push. `core` runs fine on an M-series Mac (`--device mps`). Only
`--suite code` needs the box (it executes model-generated code — never run it
on your Mac).

```bash
# local smoke test (fast, MPS)
python -m evals.lm_eval_universe --checkpoint lab_runs/B-S0/model.pt \
    --suite core --limit 20 --device mps

# full core suite
python -m evals.lm_eval_universe --checkpoint lab_runs/B-S0/model.pt \
    --suite core --out results/evals

# code, on the box only
python -m evals.lm_eval_universe --checkpoint lab_runs/B-S0/model.pt \
    --suite code
```

Results: `results/evals/<run_name>/<suite>.json`. Legacy coding harness below.

## Quick start

```bash
# Smoke test on the 0.5B reference model (MPS, ~1-2 min on M-series Mac)
python -m evals.run_baseline \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --device mps --limit 20

# Full suite on a real GPU
python -m evals.run_baseline \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --device cuda

# Eval a local checkpoint from continued pretraining
python -m evals.run_baseline \
  --model ./checkpoints/coding-model-001/step-10000 \
  --device cuda
```

Each run writes:
- `results/humaneval__<model>.jsonl` — raw samples (consumed by `human_eval` scorer)
- `results/humaneval__<model>.results.json` — per-problem pass/fail
- `results/mbpp__<model>.jsonl` — raw samples
- `results/mbpp__<model>.summary.json` — pass@1
- `results/report__<model>__<timestamp>.json` — combined run summary

## Why these evals and not others

- **HumanEval** is the lingua franca. Every coding model reports it. You can compare to any paper or leaderboard with a single number.
- **MBPP** is a different shape: shorter, more "real beginner problems." Tests whether the model can solve a problem stated in plain English, not just fill in a signature.
- **Both** together catch the common failure mode where a model is good at in-filling (HumanEval) but bad at problem → code translation (MBPP).

## What pass@1 is and is not

- **pass@1** = fraction of problems solved on the *first* sample, greedy decoding (temperature = 0).
- It rewards **calibrated, single-shot** code generation. Good for: autocomplete, code assistants.
- It does **not** reward: diversity, exploration, or long chain-of-thought. For that you'd want pass@10 or pass@100.
- This lab targets the assistant use case → pass@1 is the right headline metric.

## Adding a model to the leaderboard

After a successful eval, append a row to `coding-model-leaderboard.md` in the project root:

```md
| model | size | humaneval | mbpp | date | run_id |
|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B-Instruct | 0.5B | 0.43 | 0.55 | 2026-06-07 | report__... |
```

The point of the leaderboard is to compare **your checkpoints** to reference models, not to compete with frontier labs. A small model that beats Qwen 2.5 Coder 0.5B at the same size on both is a real result.

## Adding a new eval

1. Drop a new file `evals/<name>.py` exposing a `run(model, device, limit, instruct, out_path) -> dict` function.
2. Add the name to `SUITE` in `run_baseline.py`.
3. Document the eval in this README.
4. Add a row to the leaderboard template.

Keep the runners thin: one model load, one generation loop, one scoring step. No abstractions until you have three of them.

## What this eval suite is *not*

- Not a replacement for `lm-evaluation-harness` or `bigcode-evaluation-harness`. Those are the standards; we use them as references but keep our own runner because (a) it works on MPS out of the box, (b) it logs results in the lab's format, and (c) it costs zero extra dependencies beyond `transformers` + `datasets` + `human-eval`.
- Not a contamination check. If you train on GitHub code, your model has seen HumanEval. Use `humanevalplus` and `livecodebench` for honest numbers.
- Not a deployment test. Real assistants need latency, cost, and safety evals too — those are a different folder, not this one.
