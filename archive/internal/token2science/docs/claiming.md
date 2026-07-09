# Claiming

token2science uses GitHub issues as the unit of work.

- Each task has a GitHub issue.
- To claim a task, comment `/claim` on that issue.
- A GitHub Action grants an exclusive lease by adding a `claimed:<user>` label and recording a claim timestamp.
- If an unexpired claim by someone else already exists, the Action refuses `/claim`.
- A scheduled Action expires stale claims after a TTL, defaulting to 60 minutes, by removing the claim label.
- `/release` or a merged PR frees the lease immediately.

This mirrors the local `token2science/claim.py` lease semantics, with GitHub comments and labels used only as the transport.
