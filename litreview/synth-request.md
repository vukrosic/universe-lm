---
status: idle
updated: 2026-06-09T00:00:00Z
theme: ""
min_digests: 5
---

# Synthesis request

Flip `status` to `needs-synth` and set `theme` when you want the synthesizer to
run. It reads all `done` papers with matching `theme:` in frontmatter and writes
`litreview/synthesis.md`.

```bash
# example — request attention-theme synthesis
# (edit frontmatter below, then orchestrate will pick it up on next manual invoke)
```

Idle — no synthesis queued.
