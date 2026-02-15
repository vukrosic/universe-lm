---
name: repo-cleaner
description: Prepares the repository for publication by cleaning up temporary artifacts, archiving failed experiments, and updating documentation.
---

# Repository Cleaner Skill

You clean up the research workspace for clarity and organization.

## Cleaning Logic

### 1. Artifact Archival
- Create `archive/` directory structure (`archive/plots`, `archive/logs`, `archive/proposals`).
- Move ALL `plots/*.json` and `plots/*.png` to `archive/plots/` **EXCEPT** the files corresponding to the current best result.
- Move superseded `docs/research/*.md` proposals to `archive/research/`.
- Keep ALL variance reports (`baseline_variance_*.md`) — never archive these.

### 2. Code Hygiene
- Ensure `configs/` defaults match the current best configuration.
- Remove temporary scripts no longer needed.
- Ensure experiment flags default to OFF (baseline behavior) in committed code.

### 3. Documentation Update
- Update the root `README.md` with current status.
- Update `docs/research/idea_log.md` statuses to reflect final verdicts.
- Ensure failed experiments are documented (not deleted).

### 4. Never Delete
- Never delete variance reports
- Never delete the idea log
- Never delete failed experiment records (archive them, don't destroy them)

## How to use this skill

1.  **Identify the current best result** (from the latest experiment report).
2.  **Archive** intermediate artifacts.
3.  **Update** documentation.
4.  **Verify** baseline config is default.
