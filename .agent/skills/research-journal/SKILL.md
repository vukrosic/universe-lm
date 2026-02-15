---
name: research-journal
description: Manages the documentation and lifecycle of research ideas. Processes ideas into a persistent log and categorizes them by status (Backlog, Ongoing Review, Validated, Archived).
---

# Research Journal Skill

Your job is to be the "librarian" of the research process. You ensure that no idea is lost and that every critique is documented.

## Journal Organization

Maintain a file at `docs/research/idea_log.md`. The folder and file should be created if they don't exist.

### Status Categories:
1. **The Idea Spark (Backlog)**: Raw ideas from the `ai-research-innovator` that haven't been reviewed yet.
2. **Peer Review (Under Evaluation)**: Ideas currently being critiqued by the `idea-reviewer`.
3. **The Forge (Revision)**: Ideas being iterated on by the `idea-revisor`.
4. **The Vault (Accepted)**: High-quality, mathematically sound ideas ready for implementation.
5. **The Archive (Discarded)**: Ideas that were proven redundant or unfeasible, with a short "Lessons Learned" note.

## How to use this skill

1. **Log Entry**: Whenever a new idea or feedback is generated, immediately record it in the `idea_log.md`. Do NOT wait for permission or offer to do so.
2. **Metadata**: Each entry should include:
   - **ID**: (e.g., `RID-01`)
   - **Timestamp**: Current date.
   - **Status**: Current lifecycle stage.
   - **Core Concept**: 1-sentence summary.
3. **Maintenance**: Regularly clean up the log and move items between categories based on the automated research progress.

## Goal
To create a "Paper Trail" of intelligence that shows the evolution of thought from a raw snippet of code to a publishable mathematical idea.
