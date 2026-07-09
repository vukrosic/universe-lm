# Baseline cache — Phase 2 design

> **Status:** design / not yet implemented. Phase 1 (record ledger + UI) shipped
> in `voidspark` (`records.jsonl`, `/api/research-records`, the Record-history &
> Closed-experiments sections). This doc specs Phase 2: the GPU-side rework that
> stops retraining the baseline from scratch on every run.
>
> Read [`PIPELINE.md`](PIPELINE.md) (status vocab, prime directive) and
> [`prompts/runner.md`](prompts/runner.md) §2–§5 (the two-ctrl bracket, the run
> queue, the close-the-loop rules) first — this doc only changes those.

---

## 1. The problem

Every A/B run trains **three models from scratch**: `ctrl`, the treatment, and
`ctrl2` (the variance bracket, `runner.md` §2/§3a). On a serial GPU that means
**~2/3 of every run re-establishes a baseline we already know.** With seed fixed
at 42 and tier fixed at `tiny1m3m`, the control is the *same training run* every
time — modulo box variance. We are paying full price for a number we could cache.

`runner.md`'s prime directive is "the GPU must never be idle / test more ideas."
Dropping two of three jobs per run is a direct ~3× throughput win against that
directive.

## 2. Why we can't just cache blindly (the constraint)

The two controls are **not redundant**. They measure **box variance at run time**:
identical-seed `ctrl` re-runs swing **~0.04 val loss** (`runner.md` §2, measured).
A treatment counts as a WIN **only if it beats both `ctrl` and `ctrl2` by more
than the gap between them.** That noise band is what makes a WIN trustworthy.

And the box genuinely changes. From `remote-results/*/results.json`:

```json
"instance": { "gpu": "NVIDIA GeForce RTX 3060", "compute_cap": "8.6",
              "driver": "580.159.03", "host": "1.208.108.242:52674" }
```

`host` changes on **every Vast rental**; `gpu`/`compute_cap`/`driver` define the
*class* of box, which is what the ~0.04 band is tied to. A baseline measured on an
RTX 3060 box is **not** comparable to a treatment run on a different GPU. A naive
wall-clock cache would silently produce **false wins** the moment the box class
changes. The 015→025 record ledger already shows how fragile cross-session
comparison is — half those "wins" carry a *"shared ctrl was buggy / re-test for
isolation"* caveat because the control wasn't the right control.

**Conclusion: cache the baseline keyed to the box class, not to time.** Re-measure
when the key changes or the record moves. This keeps the bracket's rigor while
reclaiming the compute.

## 3. Design

### 3.1 Box fingerprint (the cache key)

```
box_key = sha1(f"{gpu}|{compute_cap}|{driver}")[:12]
```

Derived from `results.json.instance`. **Excludes `host`** (changes per rental) and
`id` (derived from host). Same GPU model + compute cap + driver ⇒ same key ⇒ the
~0.04 band holds and a cached baseline is reusable. If driver upgrades, the key
changes and we re-measure — cheap insurance against a driver-induced shift.

### 3.2 Baseline cache file — `autoresearch/baseline-cache.json`

One entry per `box_key`. The baseline is measured with **N≥3 same-seed (42)
re-runs once** — *not* multiple seeds. The pipeline is seed-42-only by rule
(`runner.md`: "the fix for noise is the two-ctrl bracket, **never more seeds**"),
and the treatment-vs-baseline comparison must stay **paired at seed 42** or the
delta is confounded. The N re-runs measure **box/run variance at the fixed seed**
— exactly what the two-ctrl bracket measures today, just N times once instead of
twice every run. N=5 is a better reference than today's n=2 fresh-every-run.

```json
{
  "tier": "tiny1m3m",
  "seed": 42,
  "boxes": {
    "5b8a7fea8963": {
      "box_key": "5b8a7fea8963",
      "gpu": "NVIDIA GeForce RTX 3060",
      "compute_cap": "8.6",
      "driver": "580.159.03",
      "commit": "<git short sha of repo at measure time>",
      "n_measurements": 5,
      "val_runs": [6.4272, 6.4419, 6.4409, 6.4225, 6.4184],
      "val_mean": 6.4302,
      "val_std": 0.0096,
      "noise_band": 0.04,
      "measured_at": "2026-06-14T09:00:00Z",
      "runs_since_measure": 0,
      "source_results": "remote-results/2026-06-14-vast-tiny1m3m/results.json"
    }
  }
}
```

`noise_band = max(0.04, 2·val_std)` — the documented ~0.04 box-variance floor
(`runner.md` §2, measured) OR 2σ of the re-runs, whichever is **wider**. Wider is
the safe direction: it makes a WIN *harder* to claim, so it suppresses false wins.

- `noise_band` replaces the live ctrl-to-ctrl gap. A treatment is a **WIN iff
  `trt < val_mean − noise_band`** (and clears the idea's plan bar); **NULL** if it
  lands inside `val_mean ± noise_band`.
- `commit` guards correctness: if the baseline-relevant code (train loop, tier
  config) changed since the cache was measured, the cache is stale → re-measure.

### 3.3 Re-baseline triggers (the two you chose)

Re-measure the baseline for a `box_key` when **any** holds:

1. **Box change** — no cache entry for the current `box_key` (new GPU class /
   driver), **or** `commit` mismatch (baseline-relevant code changed).
2. **Record break** — a treatment beats the record by ≥ the WIN threshold. The
   winning config *becomes the candidate baseline*; confirm it with a fresh N≥3
   same-seed measurement on the current box, then it's the new reference. (This is
   the "until we break the record by a certain amount" you asked for, encoded.)
3. **Staleness guard** — `runs_since_measure ≥ K` (default **K=25**). Bounds
   accumulated drift even if the box key is stable. Tunable; box-change +
   record-break are the primary triggers, this is the safety net.

When none fire: skip `ctrl` and `ctrl2` entirely, run **treatment-only**, judge
against the cached `val_mean ± noise_band`.

### 3.4 Optional cheap drift sentinel

Every run still has *some* risk the box silently drifted within-key. Cheap
mitigation: ~1 in `K` runs (or when wall-clock since last ctrl > 24h), prepend a
**single** `ctrl` (not the full N≥3) and check it lands within `val_mean ±
noise_band`. If it doesn't → flag `BOX DRIFT`, distrust that run's treatment, and
force a full re-measure. This is ~1 extra job amortized over `K`, vs 2 every run.

## 4. Changes to the run path

| File | Today | Phase 2 |
|---|---|---|
| `prompts/runner.md` §2 | "queue runs ctrl twice (first + last)" | Check `baseline-cache.json` for `box_key`. **Triggers fired** → measure N≥3 ctrls, write cache. **Else** → no ctrl in queue. |
| `prompts/runner.md` §3a | `run ctrl … run <trt> … run ctrl2` | Trigger path: `run ctrl ×3` then treatments. Cached path: treatments only. |
| `prompts/runner.md` §5 | WIN = beats both ctrls by > gap | WIN = `trt < val_mean − noise_band` **and** clears plan bar. NULL = inside band. On WIN that breaks record → set re-baseline trigger #2. |
| `prompts/run-idea.md` | "Run ctrl + treatment + ctrl2" | Same cache check; cached path runs treatment only. |
| `autoresearch/baseline-cache.json` | — | **New.** Box-keyed baseline store (§3.2). |
| `autoresearch/records.jsonl` | Phase-1 ledger | Now also the **record-break trigger source** (§3.3.2). Already append-only; add nothing. |
| `bin/` | `flip.sh` | **New** `baseline.sh {check,measure,bump,verdict}` so agents never hand-edit the JSON (mirrors the flip.sh discipline). |

> **Status: IMPLEMENTED 2026-06-14.** All four rows above are done —
> `baseline-cache.json` is backfilled (box `5b8a7fea8963`, N=5, mean 6.4302, band
> 0.04), `bin/baseline.sh` is written + smoke-tested, and `runner.md` §2/§3a/§5 +
> `run-idea.md` are cut over to the cache. The only step not yet run is the live
> validation (§5.4) — it needs the next real GPU queue.

`results.json` schema is unchanged — cached-baseline runs just have fewer
`runs[]` entries. `evidence.md` gains one line: `baseline: cached val_mean=<x>
±<band> (box <key>, measured <date>)` so a reader sees it wasn't a fresh ctrl.

## 5. Rollout

1. **Backfill** `baseline-cache.json` from the most recent full-bracket run per
   box class already in `remote-results/` — no new GPU time to seed the cache.
   **(done — N=5 same-seed ctrls from 06-13/06-14 on the RTX 3060 box, stronger
   than the N≥3 target.)**
2. ~~**Shadow check (1 cycle)**~~ — **skipped by decision (2026-06-14).** The
   backfill is N=5 with tight σ=0.0096 (band floored at the documented 0.04), so
   we cut over directly rather than burn a confirmation cycle. The first live
   queue *is* the validation (the Watch step below).
3. **Cut over: done.** Treatment-only is the default; ctrl runs only on a fired
   trigger or the drift sentinel.
4. **Watch:** first record-break under the new scheme must re-measure and confirm
   — that's the highest-risk moment (a cached band could manufacture the "record").

## 6. Risks

| Risk | Mitigation |
|---|---|
| Cached baseline from box A judges a treatment on box B → **false win** | `box_key` excludes host; mismatch forces re-measure (§3.1, trigger #1). |
| Box silently drifts *within* the same key | Drift sentinel (§3.4) + staleness guard K (§3.3.3). |
| Baseline-relevant code changes, cache goes stale | `commit` field mismatch = re-measure (§3.2). |
| Band set too tight → false wins; too loose → real wins logged null | Seed the band from N≥3 `val_std`, not a 2-point gap; shadow-check cycle (§5.2) validates before cutover. |
| Agent hand-edits the JSON and corrupts it | All writes via `bin/baseline.sh` (§4), never direct. |

## 7. What does **not** change

- One tier (`tiny1m3m`), one seed (42) for treatments. Prime directive. Status
  vocabulary and the flip protocol. The doer↔critic gates. `results.json` as raw
  source of truth. The two-ctrl *logic* — it's preserved as `val_mean ±
  noise_band`, just measured once per box instead of twice per run.

## 8. Expected payoff

- **~3× idea throughput** on the same GPU budget (1 job/run vs 3), directly
  serving the prime directive — more ideas tested, box never idle waiting on
  redundant baselines.
- **Better baseline** (N≥5 same-seed re-runs vs n=2) → a *more* trustworthy noise
  band, not a weaker one.
- The record ledger becomes load-bearing, not just display: it drives
  re-baselining.

## Open question for Vuk

`K` (staleness guard) default = **25 runs**. Lower = safer/more compute, higher =
cheaper/more drift risk. Box-change + record-break are the real triggers; K is
just the backstop. Fine to tune after the shadow-check cycle shows real drift
behavior.
