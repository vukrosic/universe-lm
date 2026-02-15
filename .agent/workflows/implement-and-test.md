---
description: An autonomous pipeline for exhaustive research exploration, reasoning, and discovery. Maximum scale: 1B tokens.
---

# Autonomous Research Pipeline

This workflow is an iterative loop designed to find statistically significant improvements through rigorous experimentation.

**Maximum experiment scale: 1B tokens.** This is the experimental design and discovery phase. Do not propose or execute runs beyond 1B tokens.

// turbo-all

## Stage 1: Baseline Variance (MANDATORY — Run Once Per Scale)

1.  **Check** if `docs/research/baseline_variance_8Mtokens.md` exists.
2.  If NOT: Use `experiment-runner` to run 5 baseline seeds at 8M tokens and save the variance report.
3.  If YES: Load the variance report and note μ and σ for val_loss and val_accuracy.

## Stage 2: Idea Generation & Review

1.  **Generate ideas** using `ai-research-innovator` — produce 3-5 diverse ideas.
2.  **Review** the most promising idea using `idea-reviewer` with strict novelty/falsifiability checks.
3.  **Revise** using `idea-revisor` to address weaknesses and add trivial baselines.
4.  **Log** the idea in the research journal using `research-journal`.

## Stage 3: Experiment Planning

1.  Use `experiment-planner` to design the full experiment matrix:
    - Baseline (already measured)
    - Full experiment (3+ seeds)
    - Ablation 1: simplest possible version of the idea
    - Ablation 2: trivial alternative (e.g., random gating)
2.  Define success criteria in terms of baseline σ BEFORE running.

## Stage 4: Implementation & Review

1.  Implement the idea in code using the experiment plan.
2.  Use `implementation-reviewer` to verify:
    - Baseline is preserved when flag is OFF
    - Seed reproducibility works
    - Overhead is measured

## Stage 5: Execution (8M Quick Iteration)

1.  Use `experiment-runner` to run ALL planned experiments at 8M tokens.
2.  Use `experiment-analyzer` to perform honest statistical analysis.
3.  **Decision gate**:
    - If Cohen's d < 0.2: Mark as NOISE. Go back to Stage 2 with a new idea.
    - If 0.2 ≤ d < 0.5: Mark as SUGGESTIVE. Run 2 more seeds. If still < 0.5, go back to Stage 2.
    - If d ≥ 0.5: Proceed to Stage 6.

## Stage 6: Validation (100M Confirmation)

1.  **Check** if `docs/research/baseline_variance_100Mtokens.md` exists. If not, run it first.
2.  Run the winning experiment + baseline at 100M tokens (3 seeds each).
3.  Use `experiment-analyzer` for statistical analysis.
4.  **Decision gate**:
    - If the effect holds (d ≥ 0.5): Proceed to Stage 7.
    - If the effect vanishes: Mark as "scale-dependent" and go back to Stage 2 with a new idea.

## Stage 7: Final Report

1.  Use `research-reporter` to create the full research report with all experiments.
2.  Update the `research-journal` with final verdicts.
3.  Archive intermediate artifacts using `repo-cleaner`.

## Stage 8: Paper (Only If Warranted)

**Only if Stage 6 passed** (statistically significant effect at 100M):
1.  Use `paper-drafter` to draft the paper with all experimental evidence.
2.  Use `paper-reviewer` to critique it.
3.  Use `paper-revisor` to address feedback.
4.  Optionally run at 1B tokens for stronger evidence before final paper.

## Instructions
- ALWAYS continue to the next stage unless the decision gate says otherwise.
- NEVER skip the baseline variance measurement.
- NEVER declare a winner based on a single run.
- NEVER ask the user "should I continue?" — just continue.
- STOP when Stage 7 is complete (or Stage 8 if paper is warranted).
