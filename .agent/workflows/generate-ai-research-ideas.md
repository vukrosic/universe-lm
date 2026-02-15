---
description: A complete automated pipeline for generating, reviewing, and refining original AI research ideas in a single pass.
---

# AI Research Idea Pipeline

Follow these steps to generate and refine a high-quality, testable AI research idea:

1.  **Idea Generation**:
    - Use the `ai-research-innovator` skill to analyze the current codebase.
    - Generate 3-5 original ideas across different architectural aspects.
    - Each idea must include: math, pro/con analysis, failure predictions, and a minimum viable experiment at ≤8M tokens.

2.  **Autonomous Selection & Review**:
    - Select the most promising idea (highest novelty × feasibility).
    - Use `idea-reviewer` to brutally critique it: novelty, falsifiability, trivial baselines, scale appropriateness, math rigor.

3.  **Autonomous Refinement**:
    - Use `idea-revisor` to address ALL reviewer feedback.
    - Simplify overcomplicated proposals.
    - Add trivial baseline comparisons.
    - Constrain scope to ≤1B tokens.

4.  **Journal Entry**:
    - Use `research-journal` to log the idea with status "Ready to Test".

5.  **Final Output**:
    - Present the finalized V2 proposal.
    - Include the experiment plan with ablations and success criteria.
    - Do not wait for user input between steps.