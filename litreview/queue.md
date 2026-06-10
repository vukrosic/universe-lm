# Literature-review queue (index)

**Each paper lives in `litreview/papers/<NNN-slug>/paper.md`.** This file is a
human-readable index only — `status:` in each `paper.md` is the routing truth.

Regenerate status view:

```bash
grep -H "status:" litreview/papers/*/paper.md 2>/dev/null | grep -v _closed
```

## Pipeline health

```bash
upstream=$(grep -L "status: \(done\|rejected\)" litreview/papers/*/paper.md 2>/dev/null | grep -v _closed | wc -l | tr -d ' ')
screen=$(grep -l "status: needs-screen" litreview/papers/*/paper.md 2>/dev/null | wc -l | tr -d ' ')
digest=$(grep -l "status: needs-digest" litreview/papers/*/paper.md 2>/dev/null | wc -l | tr -d ' ')
done=$(grep -l "status: done" litreview/papers/*/paper.md 2>/dev/null | wc -l | tr -d ' ')
echo "upstream=$upstream needs-screen=$screen needs-digest=$digest done=$done"
```

Target: **≥3 papers at `needs-screen` or `needs-digest`** so gates stay busy.

## Papers board

| # | Folder | Theme | One-liner |
|---|---|---|---|
| — | *(none filed yet)* | — | run scout to populate |

## Synthesis

Last run: *(none)* — see `synthesis.md` when synthesizer has run.

Trigger: ≥5 `done` digests in one `theme`, or edit `synth-request.md`.
