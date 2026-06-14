# Run-idea prompt (one idea → A/B on the GPU → verdict)

Run **one specific idea's** A/B on the remote GPU box, bring back the numbers,
judge it, and close the loop. The idea to run is:

**`{{IDEA_SLUG}}`** → `autoresearch/ideas/{{IDEA_SLUG}}/idea.md`

This is a **manual one-shot** launched from the lab UI's "Run next" button — you
run **this one idea only**, not the whole queue. The autonomous batch version is
[`runner.md`](runner.md); read it for the box mechanics, then apply them to just
this idea.

---

> ## 🔴 YOU RUN UNATTENDED — ACT, DON'T ASK
> No human is watching. Run the A/B, record the result, flip the status, and
> stop. Never end by asking "should I continue?".

> ## 🔴 ONE IDEA, ONE SEED (42) — baseline is **cached per box** (Phase 2)
> Run **treatment only** (flag ON), seed 42, tier `tiny1m3m`, and judge it against
> the cached baseline for this box class. Box variance ~0.04 val loss lives in the
> cache as `noise_band`; a treatment is a **WIN only if `trt < mean − band`**,
> anything inside `mean ± band` is **NULL**. Only run ctrls if `baseline.sh check`
> returns `MEASURE` (new box / commit change / staleness). See
> [`BASELINE-CACHE-DESIGN.md`](../BASELINE-CACHE-DESIGN.md).

## The box

Read `autoresearch/remote-box.json` for the live connection (it's always a Vast
box; only host/port change). Use its `ssh`, `remote_repo`, `remote_venv`. Box
realities are in [`runner.md`](runner.md) §0/§3a — the ones that bite:
- **Connection reuse is mandatory** — Vast.ai auth-throttles bursts of fresh
  handshakes and wedges the loop. Define the multiplexed `SSHOPTS` / `BOX` /
  `CP_TO` / `CP_FROM` helpers from [`runner.md`](runner.md) §0 at the top of
  your pass and route **every** ssh/scp through them; never call bare `ssh`/`scp`.
- `export PATH=<remote_venv>/bin:$PATH` (else rc 127) and `TORCHDYNAMO_DISABLE=1` (sm_86).
- Idea flags are **not** CLI args — write a tiny `_arq_{{IDEA_SLUG}}.py` that
  subclasses `Tiny1M3MConfig` with the flag set, run `--config_class __main__.C`.
- **Build-smoke `MinimalLLM(config)` on CPU** for the treatment (and ctrl if you
  measure) before training.
- Sync code first: local commit/push → `BOX 'cd <remote_repo> && git pull'`
  (the multiplexed helper from §0, not bare `ssh`).

## Steps

1. **Confirm the claim:** the app should already have flipped this idea to
   `running` before launching you. Check `status:` in `idea.md`. If it is still
   `needs-run`, claim it with
   `autoresearch/bin/flip.sh {{IDEA_SLUG}} running run-button "claimed by Run-next"`.
2. Read `idea.md` (the `## Plan` section has the config flag, run command, and the
   pass/fail bar). Sync code to the box.
3. **Baseline check:** `autoresearch/bin/baseline.sh check <recent-results-on-this-box>.json`.
   - `CACHED <mean> <band>` → **run treatment only** in a **detached tmux on the
     box** (survives disconnect), guarded so a crash logs and doesn't wedge.
   - `MEASURE …` → run **N≥3 ctrls + treatment**, then `baseline.sh measure` after
     the pull to write the fresh baseline.
4. **Pull + record** logs and a `results.json` under
   `remote-results/<date>-vast-tiny1m3m/` (match the existing schema). Then judge:
   `baseline.sh verdict <results.json> <treatment_val>` → `WIN`/`NULL`; on the
   `CACHED` path also `baseline.sh bump <results.json>`. Write `evidence.md` in the
   idea folder: treatment val, `Δ = treatment − baseline_mean`, the verdict vs the
   bar (WIN / NULL / FAIL), baseline line, and the leaderboard line.
5. **Close the loop:**
   - finished and judged → `flip.sh {{IDEA_SLUG}} done run-button "Δ=<x>; <WIN|NULL>"`
     (a WIN that also breaks the leaderboard record → flag it: next run re-baselines)
   - run crashed (OOM/NaN/bad flag) → **do not write a null** → `flip.sh {{IDEA_SLUG}} needs-recode run-button "run FAILED: <1-line cause>"`

## Finally — last command you run

Ping the app so it closes this local tmux session for you:

```bash
curl -s -X POST {{DONE_URL}} -H 'Content-Type: application/json' -d '{"slug":"{{IDEA_SLUG}}"}'
```

The session closes ~2s after this. Run only this one idea, then stop.
