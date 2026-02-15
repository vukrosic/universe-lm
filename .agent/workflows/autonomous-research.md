---
description: A fully autonomous, end-to-end research workflow that goes from idea generation to final paper PDF.
---

# Autonomous Research Workflow

This workflow executes the entire research lifecycle without user intervention. It generates ideas, implements them, runs rigorous experiments, analyzes results, and if successful, drafts and compiles a research paper.

// turbo-all

## Phase 1: Preparation & Baselines

1.  **Environment Check**:
    - Ensure `train_llm.py` and `configs/` are ready.
    - Setup specific logging directory for this research run.

2.  **Variance Limit Check**:
    - Check for `docs/research/baseline_variance_*.md`.
    - If missing, AUTOMATICALLY run the baseline variance estimation (3 seeds) and save the report.
    - **CRITICAL**: Do not proceed without a variance baseline.

## Phase 2: Autonomous Discovery Loop

1.  **Ideation**:
    - Use `ai-research-innovator` to generate 3 novel, testable ideas.
    - Use `idea-reviewer` to select the best one based on:
        - Feasibility (can be implemented in <200 lines).
        - Novelty (not just a parameter tune).
        - Testability (clear metrics).

2.  **Refinement**:
    - Use `idea-revisor` to refine the selected idea into a concrete implementation plan.
    - Define the control and experimental groups strictly.

3.  **Implementation**:
    - **Step 3.1**: 
        - **Rule**: Keep changes minimal and focused.
        - **Rule**: Use a configuration flag to toggle the idea (e.g., `--use_my_idea`).
    - **Step 3.2**: Use `implementation-reviewer` to verify the code does not break baselines when the flag is off.

4.  **Experimentation (Small Scale)**:
    - Run the **Control** group (1 seed) at 1M tokens (or current dev scale).
    - Run the **Experimental** group (2 seeds) at 1M tokens.
    - Use `experiment-runner` for this to ensure logs are captured.

5.  **Analysis & Decision**:
    - Use `experiment-analyzer` to calculate Cohen's d and p-values.
    - **Decision Gate**:
        - If **Cohen's d < 0.2**: The idea is noise. **Revert changes** and GOTO Phase 2, Step 1 (Ideation) to try a new idea.
        - If **Cohen's d >= 0.5**: The idea has merit. Proceed to Phase 3.

## Phase 3: Publication

1.  **Drafting**:
    - Use `paper-drafter` to write the full paper.
    - Include: Introduction, Method, Experimental Setup, Results (Tables/Plots), Discussion.
    - **Mandatory**: Include the variance measurement and statistical significance in the results.

2.  **Review Loop**:
    - Use `paper-reviewer` to critique the draft.
    - Use `paper-revisor` to fix issues (overclaiming, clarity, typos).

3.  **Final Polish**:
    - Use `abstract-reviewer` for the final check of the abstract.
    - Use `latex-paper-generator` to create the PDF.

4.  **Cleanup**:
    - Use `repo-cleaner` to organize the artifacts (logs, checkpoints) into `docs/research/archive`.
    - Add the paper to `docs/papers/`.

## Phase : Completion

1.  **Notify User**:
    - Output a summary of the successful research.
    - Link to the PDF and the codebase changes.