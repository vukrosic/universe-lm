---
name: experiment-runner
description: Executes baseline and experimental training runs with mandatory multi-seed variance measurement and statistical significance testing. All runs at 1M tokens.
---

# Experiment Runner Skill

You are a research engineer responsible for rigorous execution. Every claimed result must be backed by statistical evidence. No single-run claims are ever acceptable.

## Core Principle: Variance First

**Before any experiment can be declared "better" or "worse", you MUST know the natural run-to-run variance of the baseline.** A delta smaller than 1 standard deviation is noise, not signal.

## Standard Scale

**All experiments are run at 1M tokens (1,000,000).** This is the standard benchmark scale for this project.

## Execution Protocol

### Phase 1: Baseline Variance Measurement (MANDATORY for first time)

Before running ANY experiment, establish baseline statistics:

1.  **Check if baseline variance exists**: Look for `docs/research/baseline_variance_1M.md`. If it exists with 5-seed data, skip to Phase 2.
2.  **Run the baseline 5 times** with different random seeds (42, 137, 256, 512, 1024):
    ```bash
    python train_llm.py --train_tokens 1000000 --seed 42 --output_dir checkpoints/baseline_1M_seed42
    python train_llm.py --train_tokens 1000000 --seed 137 --output_dir checkpoints/baseline_1M_seed137
    python train_llm.py --train_tokens 1000000 --seed 256 --output_dir checkpoints/baseline_1M_seed256
    python train_llm.py --train_tokens 1000000 --seed 512 --output_dir checkpoints/baseline_1M_seed512
    python train_llm.py --train_tokens 1000000 --seed 1024 --output_dir checkpoints/baseline_1M_seed1024
    ```
    **Run these sequentially (one at a time).**
3.  **Record for each run**: `val_loss`, `wall_time`
4.  **Compute**: mean (μ), standard deviation (σ), min, max for each metric.
5.  **Save** the variance report to `docs/research/baseline_variance_1M.md`
6.  **Significance threshold**: An experiment must beat the baseline mean by at least **2σ** to be declared a winner. Between 1σ and 2σ is "suggestive but inconclusive." Below 1σ is noise.
7.  **Clean up**: `rm -rf checkpoints/baseline_1M_seed*`

### Phase 2: Experiment Execution

1.  **Run the experiment 3 times minimum** with seeds (42, 137, 256):
    ```bash
    python train_llm.py --train_tokens 1000000 --seed 42 --<experiment_flag> --output_dir checkpoints/exp_1M_seed42
    python train_llm.py --train_tokens 1000000 --seed 137 --<experiment_flag> --output_dir checkpoints/exp_1M_seed137
    python train_llm.py --train_tokens 1000000 --seed 256 --<experiment_flag> --output_dir checkpoints/exp_1M_seed256
    ```
    **Run these sequentially (one at a time).** Replace `--<experiment_flag>` with actual flags.

2.  If the experiment mean falls within 2σ of the baseline, run 2 more seeds (512, 1024) to increase confidence.

### Phase 3: Statistical Comparison

1.  Compute experiment mean and std for each metric.
2.  Calculate the **effect size** (Cohen's d):
    ```
    d = (μ_experiment - μ_baseline) / σ_pooled
    ```
    where `σ_pooled = sqrt((σ_baseline² + σ_experiment²) / 2)`
3.  Classify:
    - `|d| < 0.2`: **Negligible** — not a real effect
    - `0.2 ≤ |d| < 0.5`: **Small** — suggestive, needs more data
    - `0.5 ≤ |d| < 0.8`: **Medium** — likely real
    - `|d| ≥ 0.8`: **Large** — strong effect
4.  **Wall-clock comparison**: Report actual speedup/slowdown as a percentage.

### Phase 4: Result Reporting

Save results to `docs/research/experiment_<name>_1M.md` with this format:

```markdown
# Experiment: <Name>
## Baseline Statistics (N=5 runs)
| Metric | Mean | Std | Min | Max |
## Experiment Statistics (N=3+ runs)
| Metric | Mean | Std | Min | Max |
## Statistical Test
| Metric | Cohen's d | Effect Size | Significant? |
## Wall Clock
| | Baseline Mean | Experiment Mean | Delta |
## Verdict: [WINNER / NEUTRAL / LOSER]
```

Clean up checkpoints after recording results: `rm -rf checkpoints/exp_1M_seed* checkpoints/control_1M_seed*`

## How to Extract Results

After each training run, look for the final validation loss in the terminal output. The training script prints lines like:
```
Step XXX | val_loss: X.XXXX | ...
```
Record the **last** val_loss value from each run.
