---
name: repo-cleaner
description: Prepares the repository for publication by cleaning up temporary artifacts, archiving failed experiments, and updating documentation.
---

# Repository Cleaner Skill

You are a **Release Manager**. Your goal is to take a cluttered research workspace and transform it into a clean, publishable repository.

## Cleaning Logic

1.  **Artifact Archival**:
    - Create an `archive/` directory structure (`archive/plots`, `archive/logs`, `archive/proposals`).
    - Move ALL `plots/*.json` and `plots/*.png` to `archive/plots/` **EXCEPT** the files corresponding to the "Winner" (the final configuration).
    - Move failed/unused `docs/research/*.md` proposals to `archive/requirements/`.

2.  **Code Hygiene**:
    - Remove temporary scripts (e.g., `scripts/batch_experiments.py`) if they are no longer needed.
    - Ensure `train_llm.py` and `configs/` are set to the **Winning Configuration** by default.

3.  **Documentation Update**:
    - Update the absolute root `README.md`.
    - Add a "Latest Breakthrough" section summarizing the new finding.
    - Link to the newly created research paper.
    - Clean up the `docs/research/idea_log.md` status.

## How to use this skill

1.  **Identify the Winner**: You must know which Experiment ID or Tag is the winner to preserve its artifacts.
2.  **Execute Clean**: Run the archival commands.
3.  **Polish**: Update the README.
