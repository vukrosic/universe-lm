---
description: A workflow to automatically draft a professional research paper in markdown based on a research idea.
---

# Research Paper Drafting Workflow

Follow these steps to turn a research idea into a structured paper from scratch:

1.  **Idea Generation**: 
    - Use the `ai-research-innovator` skill to analyze the code/file at hand.
    - Generate 3-5 original, mathematically-grounded ideas with accessible explanations.

2.  **Autonomous Idea Selection & Review**: 
    - Identify the most promising idea from the list (highest novelty and impact potential).
    - Immediately use the `idea-reviewer` skill to critique this specific idea based on Novelty, Feasibility, Impact, Math Rigor, and Alignment.

3.  **Autonomous Idea Refinement**: 
    - Take the reviewer's critique and use the `idea-revisor` skill to generate a "V2" of the proposal.
    - Ensure the final math is robust and the explanation is social-media ready.

4.  **Drafting**:
    - Use the `paper-drafter` skill to expand the V2 idea into a comprehensive markdown paper.
    - Target sections: Abstract, Introduction, Methodology (with full math), and Proposed Experiments.
    - Ensure authors are listed as **Vuk Rosić and Gemini**.

5. **Autonomous Paper Review**:
    - Use the `paper-reviewer` skill to critique the overall draft.
    - Specifically check for undergraduate accessibility and mathematical depth.

6. **Abstract Deep-Dive**:
    - Use the `abstract-reviewer` skill to find anything "weird", "unclear", or "iffy" in the abstract.
    - Focus heavily on jargon explanation and intuitive flow.

7. **Autonomous Revision**:
    - Use the `paper-revisor` skill to integrate BOTH the paper review and the abstract review.
    - Ensure the abstract is now 100% crystal clear and jargon-explained.

8. **Explanation Expansion**:
    - Use the `explanation-expander` skill to significantly expand the paper's length.
    - Deep-dive into jargon, add extended analogies, and explain the "soul" of every mathematical equation.
    - Ensure the paper feels "thick" with helpful, accessible information.

9. **File Creation**:
    - Save the final improved paper to the `docs/papers/` directory.
    - Name the file based on the research idea (e.g., `curvature_aware_muon.md`).

10. **Verification & Approval**:
    - Confirm the file has been created.
    - Provide a summary of the final improvements.
    - **MANDATORY**: Explicitly ask the user for approval of the paper draft.
    - **CRITICAL**: Do NOT generate any code or implement the ideas until the human user has reviewed the paper and provided explicit approval to proceed with coding.
