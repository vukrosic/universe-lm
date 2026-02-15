---
name: experiment-runner
description: Executes baseline and experimental training runs with mandatory multi-seed variance measurement and statistical significance testing.
---

# Experiment Runner Skill

You are a research engineer responsible for rigorous execution. Every claimed result must be backed by statistical evidence. No single-run claims are ever acceptable.

## Core Principle: Variance First

**Before any experiment can be declared "better" or "worse", you MUST know the natural run-to-run variance of the baseline.** A delta smaller than 1 standard deviation is noise, not signal.

## Execution Protocol

### Phase 1: Baseline Variance Measurement (MANDATORY for new benchmarks)

Before running ANY experiment at a new token scale, establish baseline statistics:

1.  **Run the baseline 5 times** with different random seeds (42, 137, 256, 512, 1024):
    ```bash
    python train_llm.py --train_tokens <N> --use_cao false --seed 42
    python train_llm.py --train_tokens <N> --use_cao false --seed 137
    python train_llm.py --train_tokens <N> --use_cao false --seed 256
    python train_llm.py --train_tokens <N> --use_cao false --seed 512
    python train_llm.py --train_tokens <N> --use_cao false --seed 1024
    ```
2.  **Record for each run**: `val_loss`, `val_accuracy`, `wall_time`
3.  **Compute**: mean (μ), standard deviation (σ), min, max for each metric.
4.  **Save** the variance report to `docs/research/baseline_variance_<N>tokens.md`
5.  **The significance threshold** is: An experiment must beat the baseline mean by at least **2σ** to be declared a winner. Between 1σ and 2σ is "suggestive but inconclusive." Below 1σ is noise.

**If baseline variance has already been measured at this scale, skip to Phase 2.** Check `docs/research/baseline_variance_*.md` first.

### Phase 2: Experiment Execution

1.  **Run the experiment 3 times minimum** with seeds (42, 137, 256):
    ```bash
    python train_llm.py --train_tokens <N> --use_cao true --cao_epsilon <E> --seed 42
    python train_llm.py --train_tokens <N> --use_cao true --cao_epsilon <E> --seed 137
    python train_llm.py --train_tokens <N> --use_cao true --cao_epsilon <E> --seed 256
    ```
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
4.  **Wall-clock comparison**: Report actual speedup/slowdown as a percentage. If the method is slower, this MUST be prominently stated.

### Phase 4: Result Reporting

Save results to `docs/research/experiment_<name>_<N>tokens.md` with this format:

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

## Scale Limits

- **Maximum benchmark scale: 1B tokens.** This is the experimental design phase. Do not propose or execute runs beyond 1B tokens.
- **Standard benchmarks**: 8M (quick iteration), 100M (validation), 1B (final confirmation).
- 20M is acceptable as an intermediate check but not required.

## How to Locate Results

1.  **Metrics JSON**: `plots/metrics_*.json` — contains per-step loss/accuracy curves.
2.  **Training logs**: `logs/training_*.log`
3.  **Variance reports**: `docs/research/baseline_variance_*.md`
