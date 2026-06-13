# Implement-idea prompt (one idea → working experiment code)

Take **one specific idea** and turn it into ready-to-run experiment code in this
repo. The idea to implement is:

**`{{IDEA_SLUG}}`** → `autoresearch/ideas/{{IDEA_SLUG}}/idea.md`

---

> ## 🔴 YOU RUN UNATTENDED — ACT, DON'T ASK
> This fires from a button with no human watching. Implement the idea and write
> the code down. Never end by asking "should I continue?" — just do it, then stop.

> ## 🔴 THE FIXED TEST — never change it
> The experiment runs at **tiny1m3m (0.94M params · 3M tokens), one seed: 42.**
> No seed sweeps, no other scale. The change must be **< 200 LoC** behind a config
> flag, **off by default**, and **byte-identical to the baseline at step 0**
> (zero/identity init).

**Repo:** `/Users/vukrosic/my-life/llm-research-kit-scaling`

Before you start: `git status` and `git diff` (another agent may be editing the
same files). Never rebase, never push.

## Step 1 — Claim it (tracking)

Mark the idea as being implemented so it's tracked:

```bash
autoresearch/bin/flip.sh {{IDEA_SLUG}} implementing implement-button "claimed by implement button"
```

Then read the whole `autoresearch/ideas/{{IDEA_SLUG}}/idea.md` — the mechanism,
design sketch, and the bet are already written. Follow the design sketch.

## Step 2 — Plan

Write `autoresearch/ideas/{{IDEA_SLUG}}/plan.md`: the exact files and functions
you'll change (usually `models/layers.py` and/or `configs/llm_config.py`), the
config flag name (`use_<feature>`), how it stays zero-init at step 0, and how the
final val loss is read. Keep it tight.

## Step 3 — Implement

Make the change behind the `use_<feature>` flag, **off by default** so the
baseline path is untouched. Keep it minimal and < 200 LoC. Then confirm:
- it imports cleanly,
- the flag toggles the behavior,
- with the flag **off**, step-0 output matches the baseline.

Follow the existing run convention in `prompts/runner.md` / `PIPELINE.md` so the
experiment is ready to launch (don't necessarily run the full training here —
just make it runnable and note the exact command in `plan.md`).

## Step 4 — Mark done (tracking) and stop

When the code is in place and verified to import/toggle:

```bash
autoresearch/bin/flip.sh {{IDEA_SLUG}} needs-run implement-button "code ready; runnable at tiny1m3m seed 42"
```

If you get genuinely blocked and can't implement it, bounce it back instead of
leaving it stuck:

```bash
autoresearch/bin/flip.sh {{IDEA_SLUG}} needs-review implement-button "blocked: <one-line reason>"
```

Print a short log: idea slug, files changed, the flag name, the run command, and
the final status. Then stop — implement only this one idea.
