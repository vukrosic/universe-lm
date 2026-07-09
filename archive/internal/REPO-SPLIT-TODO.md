# TODO: split universe-lm (public) from autoresearch (private lab)

**Status:** planned, not started. Decided 2026-06-28.

## Why
This working directory is really two projects tangled together:
- **universe-lm** — the public model + train + eval kit the README pitches to compute collaborators (~40 files: `train_llm.py`, `configs/`, `models/`, `optimizers/`, `training/`, `data/`, `utils/`, `benchmarks/`, `evals/`, `tests/`, `docs/`).
- **autoresearch** — the autonomous-experiment daemon + lab notebook that *produced* the four levers (`autoresearch/` = 1,383 tracked files, plus `_arq_*.py`, `runs/`, `baselines/`, `plans/`, `releases/`, `results/`, `remote-results/`, `token2science/`, `kaggle_job/`, `release.sh`, `LEADERBOARD.md`, the `CONTRIBUTING.md` template workflow).

A compute provider who clones this currently wades through ~1,383 files of daemon state to find a ~40-file training kit.

## Key facts (verified 2026-06-28)
- Core path has **zero runtime dependency** on `autoresearch/` — the only references in `train_llm.py`/`models/`/`optimizers/` are comment pointers (`# see autoresearch/ideas/NNN/idea.md`). Code runs identically without it.
- `champion.json` points at `_arq_323-champion-mom90-lr2x.py`, but the `_arq_*.py` scripts are **untracked** (0 in git) — so that pointer is already dangling on a fresh clone. The coupling only exists locally.
- Daemon last active ~Jun 20 2026. Real and recent → **separate, do not delete.**

## Decision
**Split, non-destructive. Do NOT `rm -rf autoresearch/`** (1,383 files of genuine research provenance — the trail behind value-embeddings / per-head Q-gain / RoPE-base / sliding-window).

Two acceptable shapes (pick when we do this):
1. **Untrack in place** — `git rm --cached -r autoresearch/ runs/ baselines/ ...` + add to `.gitignore`. Files stay on disk, daemon keeps working, GitHub shows only the clean kit. Lowest effort.
2. **Move to its own repo** — physically relocate autoresearch to a separate private repo. Cleaner, but the daemon's relative paths may need fixup.

## When we do it, also
- Decide what (if anything) stays as a curated `lab/` or `ideas/` summary in the public repo for provenance.
- Fix or drop the dangling `champion.json` stub reference if autoresearch leaves.
- Re-point `CONTRIBUTING.md` / `LEADERBOARD.md` if they stay public.
