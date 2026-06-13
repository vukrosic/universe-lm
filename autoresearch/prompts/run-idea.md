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

> ## 🔴 ONE IDEA, ONE SEED (42)
> A/B = **control (flag OFF) + treatment (flag ON)**, seed 42, tier `tiny1m3m`.
> Bracket with a second control (`ctrl2`) so a treatment counts as a **WIN only
> if it beats both controls** by more than the gap between them — box variance is
> ~0.04 val loss, anything inside that is **NULL**, not real.

## The box

Read `autoresearch/remote-box.json` for the live connection (it's always a Vast
box; only host/port change). Use its `ssh`, `remote_repo`, `remote_venv`. Box
realities are in [`runner.md`](runner.md) §0/§3a — the ones that bite:
- `export PATH=<remote_venv>/bin:$PATH` (else rc 127) and `TORCHDYNAMO_DISABLE=1` (sm_86).
- Idea flags are **not** CLI args — write a tiny `_arq_{{IDEA_SLUG}}.py` that
  subclasses `Tiny1M3MConfig` with the flag set, run `--config_class __main__.C`.
- **Build-smoke `MinimalLLM(config)` on CPU** for ctrl + treatment before training.
- Sync code first: local commit/push → `ssh BOX 'cd <remote_repo> && git pull'`.

## Steps

1. **Confirm the claim:** the app should already have flipped this idea to
   `running` before launching you. Check `status:` in `idea.md`. If it is still
   `needs-run`, claim it with
   `autoresearch/bin/flip.sh {{IDEA_SLUG}} running run-button "claimed by Run-next"`.
2. Read `idea.md` (the `## Plan` section has the config flag, run command, and the
   pass/fail bar). Sync code to the box.
3. **Run** ctrl + treatment + ctrl2 in a **detached tmux on the box** (so it
   survives disconnect), guarded so a crash logs and doesn't wedge.
4. **Pull + record** logs and a `results.json` under
   `remote-results/<date>-vast-tiny1m3m/` (match the existing schema), and write
   `evidence.md` in the idea folder: the val losses, `Δ = treatment − ctrl`, the
   verdict vs the bar (WIN / NULL / FAIL), and the leaderboard line.
5. **Close the loop:**
   - finished and judged → `flip.sh {{IDEA_SLUG}} done run-button "Δ=<x>; <WIN|NULL>"`
   - run crashed (OOM/NaN/bad flag) → **do not write a null** → `flip.sh {{IDEA_SLUG}} needs-recode run-button "run FAILED: <1-line cause>"`

## Finally — last command you run

Ping the app so it closes this local tmux session for you:

```bash
curl -s -X POST {{DONE_URL}} -H 'Content-Type: application/json' -d '{"slug":"{{IDEA_SLUG}}"}'
```

The session closes ~2s after this. Run only this one idea, then stop.
