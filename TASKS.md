# Task board — beads

The lab's task tracker is [beads](https://github.com/gastownhall/beads) (`bd`).
Source of truth: the local bd database; the public snapshot lives at
`.beads-export/issues.jsonl` (one JSON object per issue) and is rendered live
on the lab dashboard.

Workflow:
- agents/humans: `bd ready` → pick a task → `bd update <id> --status in_progress --assignee <you>`
- done: `bd close <id> --reason "..."`
- publish: `bd export -o .beads-export/issues.jsonl && git add -A && git commit -m "tasks: sync" && git push`

Contributor credit is via accepted PRs (see CONTRIBUTING.md); the board is the
map, PRs are the territory.
