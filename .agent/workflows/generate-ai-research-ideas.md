---
description: A complete automated pipeline for generating, reviewing, and refining original AI research ideas in a single pass.
---

# AI Research Idea Pipeline

Follow these steps to generate and refine a high-quality, testable AI research idea:

1.  **Idea Generation**:
    - Use the `ai-research-innovator` skill to analyze the current codebase.
    - Generate 3 original ideas across different architectural aspects.
    - Each idea must include: math, pro/con analysis, failure predictions, and a minimum viable experiment at **1M tokens**.

2.  **Autonomous Selection & Review**:
    - Select the most promising idea (highest novelty × feasibility).
    - Use `idea-reviewer` skill to brutally critique it: novelty, falsifiability, trivial baselines, scale appropriateness, math rigor.

3.  **Autonomous Refinement**:
    - Use `idea-revisor` skill to address ALL reviewer feedback.
    - Simplify overcomplicated proposals.
    - Add trivial baseline comparisons.
    - Constrain scope to **1M tokens**.

4.  **Journal Entry**:
    - Log the idea to `docs/research/idea_log.md` with status "Ready to Test" using the `research-journal` format.

5.  **Final Output**:
    - Present the finalized V2 proposal.
    - Include the experiment plan with success criteria.
    - Do not wait for user input between steps.