---
description: A fully autonomous, end-to-end research workflow that goes from idea generation to final paper PDF.
---

# Autonomous Research Workflow

This workflow executes the entire research lifecycle without user intervention. It generates ideas, implements them, runs rigorous experiments, analyzes results, and if successful, drafts and compiles a research paper.

**ALL experiments are run at 1M tokens (1,000,000).** Do not run at 8M, 100M, or any other scale.

// turbo-all

## Phase 0: Environment Setup

1.  **Read Setup Instructions**:
    - Read `docs/SETUP_INSTRUCTIONS.md` for context.

2.  **Install Requirements**:
    - Run: `pip install -r requirements.txt`
    - If any install fails, troubleshoot and fix before continuing.

3.  **Download Dataset (if needed)**:
    - Check if `processed_data/pretrain_dataset/` exists and is non-empty.
    - If missing, run: `python3 data/download_hf_data.py`
    - Verify the dataset was downloaded successfully.

4.  **Smoke Test**:
    - Run a quick 50-step training to verify everything works:
      ```bash
      python train_llm.py --train_tokens 100000 --output_dir checkpoints/smoke_test --seed 42
      ```
    - If this fails, debug and fix before proceeding.
    - Delete the smoke test checkpoint after verification: `rm -rf checkpoints/smoke_test`

## Phase 1: Baseline Variance Measurement

1.  **Check for Existing Baselines**:
    - Look for `docs/research/baseline_variance_1M.md`.
    - If it exists and contains valid 5-seed data, skip to Phase 2.

2.  **Run Baseline (5 seeds)**:
    - Run the unmodified baseline 5 times at **1M tokens**:
      ```bash
      python train_llm.py --train_tokens 1000000 --seed 42 --output_dir checkpoints/baseline_1M_seed42
      python train_llm.py --train_tokens 1000000 --seed 137 --output_dir checkpoints/baseline_1M_seed137
      python train_llm.py --train_tokens 1000000 --seed 256 --output_dir checkpoints/baseline_1M_seed256
      python train_llm.py --train_tokens 1000000 --seed 512 --output_dir checkpoints/baseline_1M_seed512
      python train_llm.py --train_tokens 1000000 --seed 1024 --output_dir checkpoints/baseline_1M_seed1024
      ```
    - **IMPORTANT**: Run these sequentially (one at a time). Do NOT run them in parallel.
    - After each run, extract the final `val_loss` from the terminal output or metrics JSON.

3.  **Compute Baseline Statistics**:
    - Compute mean (μ), standard deviation (σ), min, max for `val_loss` and `wall_time`.
    - Save to `docs/research/baseline_variance_1M.md` using this format:
      ```markdown
      # Baseline Variance Report: 1M Tokens
      | Metric | Mean | Std | Min | Max |
      |--------|------|-----|-----|-----|
      | val_loss | X.XXXX | X.XXXX | X.XXXX | X.XXXX |
      | wall_time (s) | X.XX | X.XX | X.XX | X.XX |
      
      Seeds used: 42, 137, 256, 512, 1024
      ```

4.  **Clean Up Baseline Checkpoints**:
    - Remove checkpoint directories to save space: `rm -rf checkpoints/baseline_1M_seed*`

## Phase 2: Autonomous Discovery Loop

1.  **Ideation**:
    - Use `ai-research-innovator` skill to analyze the codebase and generate 3 novel, testable ideas.
    - Use `idea-reviewer` skill to select the best one based on:
        - Feasibility (can be implemented in <200 lines).
        - Novelty (not just a parameter tune).
        - Testability (clear metrics at 1M tokens).

2.  **Refinement**:
    - Use `idea-revisor` skill to refine the selected idea into a concrete implementation plan.
    - Define the control and experimental groups strictly.
    - Log the idea to `docs/research/idea_log.md` using the `research-journal` skill format.

3.  **Implementation**:
    - **Step 3.1**: Implement the idea in the codebase.
        - **Rule**: Keep changes minimal and focused.
        - **Rule**: Use a configuration flag to toggle the idea (e.g., `--use_my_idea`).
        - **Rule**: When the flag is OFF, the code path must be IDENTICAL to the original baseline.
    - **Step 3.2**: Verify the implementation didn't break the baseline:
        - Run: `python train_llm.py --train_tokens 1000000 --seed 42 --output_dir checkpoints/verify_baseline`
        - Compare val_loss to the known baseline mean. It must be within 1σ.
        - Clean up: `rm -rf checkpoints/verify_baseline`

4.  **Experimentation (1M tokens)**:
    - Run the **Control** group (3 seeds) at **1M tokens**:
      ```bash
      python train_llm.py --train_tokens 1000000 --seed 42 --output_dir checkpoints/control_1M_seed42
      python train_llm.py --train_tokens 1000000 --seed 137 --output_dir checkpoints/control_1M_seed137
      python train_llm.py --train_tokens 1000000 --seed 256 --output_dir checkpoints/control_1M_seed256
      ```
    - Run the **Experimental** group (3 seeds) at **1M tokens** with the feature flag ON:
      ```bash
      python train_llm.py --train_tokens 1000000 --seed 42 --use_my_idea true --output_dir checkpoints/exp_1M_seed42
      python train_llm.py --train_tokens 1000000 --seed 137 --use_my_idea true --output_dir checkpoints/exp_1M_seed137
      python train_llm.py --train_tokens 1000000 --seed 256 --use_my_idea true --output_dir checkpoints/exp_1M_seed256
      ```
    - **IMPORTANT**: Run ALL runs sequentially (one at a time). Replace `--use_my_idea` with the actual flag name.

5.  **Analysis & Decision**:
    - Use `experiment-analyzer` skill to calculate Cohen's d and significance.
    - **Decision Gate**:
        - If **Cohen's d < 0.2**: The idea is noise. **Revert all code changes** and GOTO Phase 2, Step 1 (Ideation) to try a new idea. Maximum 3 attempts total before declaring failure.
        - If **0.2 ≤ Cohen's d < 0.5**: Suggestive. Run 2 more seeds (512, 1024) for both control and experimental to increase confidence. Re-analyze.
        - If **Cohen's d ≥ 0.5**: The idea has merit. Proceed to Phase 3.
    - Save experiment results to `docs/research/experiment_<name>_1M.md`.
    - Update `docs/research/idea_log.md` with the verdict.
    - Clean up checkpoints: `rm -rf checkpoints/control_1M_seed* checkpoints/exp_1M_seed*`

## Phase 3: Paper Writing

1.  **Drafting**:
    - Use `paper-drafter` skill to write the full paper.
    - Include: Introduction, Related Work, Method, Experimental Setup, Results (with tables), Discussion, Conclusion.
    - **Mandatory**: Include the variance measurement, per-seed data, effect sizes, and wall-clock comparisons in the results.
    - Save to `docs/papers/<topic>.md`.

2.  **Review & Revision**:
    - Use `paper-reviewer` skill to critique the draft.
    - Use `abstract-reviewer` skill to check the abstract.
    - Use `paper-revisor` skill to fix ALL issues found by both reviewers.
    - Save the revised paper to `docs/papers/<topic>.md` (overwrite).

3.  **LaTeX & PDF**:
    - Use `latex-paper-generator` skill to create `docs/papers/paper.tex` and compile to `docs/papers/paper.pdf`.

## Phase 4: Cleanup & Completion

1.  **Archive**:
    - Move intermediate artifacts (old plots, old proposals) to `archive/`.
    - Keep: variance reports, final experiment report, final paper, idea log.

2.  **Notify User**:
    - Output a summary of the successful research:
        - The research question and hypothesis
        - The key result (effect size, significance)
        - Link to the PDF and the experiment report
    - If ALL ideas failed (3 attempts exhausted), report that honestly with lessons learned.
