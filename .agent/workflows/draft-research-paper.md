---
description: A workflow to automatically draft a professional research paper in markdown based on a research idea.
---

# Research Paper Drafting Workflow

Follow these steps to turn a research idea into a structured paper:

1.  **Context Gathering**:
    - Identify the finalized research idea (V2) or the most recent research proposal from the conversation history or logs.
    - Read relevant codebase files (like `optimizers/` or `models/`) to ensure the paper reflects the current implementation context.

2.  **Drafting**:
    - Use the `paper-drafter` skill to generate a comprehensive markdown paper.
    - Target sections: Abstract, Introduction, Methodology (with full math), and Proposed Experiments.

3. **Autonomous Review**:
    - Use the `paper-reviewer` skill to critique the overall draft.
    - Specifically check for undergraduate accessibility and mathematical depth.

4. **Abstract Deep-Dive**:
    - Use the `abstract-reviewer` skill to find anything "weird", "unclear", or "iffy" in the abstract.
    - Focus heavily on jargon explanation and intuitive flow.

5. **Autonomous Revision**:
    - Use the `paper-revisor` skill to integrate BOTH the paper review and the abstract review.
    - Ensure the abstract is now 100% crystal clear and jargon-explained.

6. **File Creation**:
    - Save the final improved paper to the `docs/papers/` directory.
    - Name the file based on the research idea (e.g., `curvature_aware_muon.md`).

6. **Verification**:
    - Confirm the file has been created and provide a summary of what was improved in this final version.
