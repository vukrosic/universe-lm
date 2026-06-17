# Champion promotion protocol — screen at 1 seed, confirm at 3

> **Status: policy (2026-06-15).** Amends the historical "🔴 ONE SEED ONLY" rule
> in [`PIPELINE.md`](PIPELINE.md). One seed is still the law for **screening**
> (the cheap funnel that runs every idea). **Promotion** — crowning a new
> champion, a rare and *irreversible* event that every future experiment then
> stacks on — is the one place that earns three seeds.

## Why a confirm gate exists

A champion is not just a result; it is the **baseline pinned in
`baseline-cache.json` that every later experiment is judged against and built on
top of** (see [`BASELINE-CACHE-DESIGN.md`](BASELINE-CACHE-DESIGN.md) and
`champion.json`). Promote a noise fluctuation and you don't just record a wrong
number — you poison the bar for the entire lineage after it. This has already
happened twice (`champion.json → rejected_promotions`):

- **180-qk-logit-conv** — a causal-mask leak (val 0.984) auto-promoted before the
  leak guard existed.
- **209-canon-conv** — a single-seed 6.2519 judged against a *base* control mean
  (6.3988) instead of the alibi champion (6.2403); a NULL promoted as a WIN.

Screening can be cheap and permissive because a NULL costs nothing. Promotion
must be strict because it is sticky. Different stakes ⇒ different seed budgets.

## The two stages

### Stage 1 — SCREEN (1 seed, band 0.015) — LIVE
Every idea runs once at **seed 42**, treatment-only, judged against the pinned
champion val. The daemon's live gate (`finalize_one`, env `SCREEN_BAND`):

```
candidate WIN  ⟺  trt_val < champion_val − 0.015
```

> **Band history:** 0.04 (cross-box drift, far too deaf) → **0.02** (2026-06-16,
> ~1σ within-session) → **0.015** (2026-06-17). The tier is saturated enough that
> real gains are now ~0.01–0.015, so the screen band was tightened to surface
> near-miss wins (347 stack-gmlp-mish at Δ−0.0195 read NULL under 0.02) rather than
> swallow them. A lower screen band can only *offer* a candidate to the confirm — it
> can never promote a fluke (Stage 2 does that), so sensitivity here is cheap.

A result inside the band is **NULL / inconclusive** — logged to `closed.md`,
never promoted, never "confirmed with more seeds." Most ideas end here. **Do not
promote off a Stage-1 result.**

> ⚠️ **The 0.04 band is wrong for screening — it is cross-box drift, not paired
> noise.** Measured 2026-06-15 from all 21 ctrl runs in `remote-results/`:
> - **within-session** (same box, same day, fixed seed/data): 1σ ≈ **0.017**, 2σ ≈ 0.033
> - **cross-day / cross-box drift**: 1σ ≈ **0.039**
>
> 0.04 is 2σ of the *worst* (cross-box) noise. The whole 208–216 alibi+X batch
> landed at Δ 0.005–0.025 — all swallowed by the band, all NULL. If any was a real
> +0.01–0.02 stacking win, this screen **cannot see it** (this is exactly the
> regime modded-nanogpt / parameter-golf resolve, by pairing treatment vs control
> in the *same* session and averaging seeds, never letting drift in). **Fix:** judge
> each treatment paired against a same-session/same-box control + ≥3-seed median,
> which collapses drift → a real ~0.01–0.015 band. Not yet wired into `finalize_one`.

### Stage 2 — CONFIRM (3 seeds, band 0.001 + sign guard) — required before any promotion
Only a Stage-1 candidate WIN enters Stage 2. Run **both** the challenger and the
current champion at **3 seeds — `42, 123, 7`**, back-to-back in ONE session on ONE
box (so only within-session noise is in play — the champion is RE-RUN fresh each
confirm, never compared to its stale pinned val; the bar is drift-free).

```
PROMOTE  ⟺  mean3(challenger) < mean3(champion) − 0.001     (any real improvement)
       AND  every one of the 3 seeds individually favors the challenger   (sign guard)
```

- **Operator policy (2026-06-17): promote on *any* real 3-seed-mean improvement.**
  The band is a tiny epsilon (0.001), NOT a noise window — the noise guard is
  **sign-consistency**, not band width. Because the confirm is paired (same box,
  same session, same seeds, champion re-run live), a wide band is no longer the
  right instrument; the failure mode is a fluky-negative *mean*, which 3/3-seeds-
  agree kills. For a true null, `mean3 < champ − 0.001` alone false-promotes ~37%
  (≈0.3·SEM); requiring all 3 seeds right-sign drops that to (½)³ = **12.5%** while
  still passing any genuine small gain. Enforced in `bin/confirm_paired.py`
  (`all_negative` + `--band 0.001`).
- *(historical)* The old **0.02 ≈ 2·SEM** band shrank the single-seed 0.04 by √3,
  back when the confirm was treated as a wide-window test — superseded above.
  zero.
- **Like-for-like only.** Never compare a 3-seed challenger mean to a 1-seed
  champion val. Both sides are 3-seed means at the same tier/box.
- **Fails the sign guard ⇒ do not promote.** `mean3(challenger) < mean3(champion)`
  by >0.001 but with a seed disagreeing (one seed favors the champion) is
  "promising, inconclusive": keep the champion, log it, move on. The sign guard —
  not a wide band — is now the noise floor (see Stage 2 above).

On promotion: pin `mean3(challenger)` as the new champion val + its 3-seed std,
append to `champion.json.lineage`, and the new champion becomes the Stage-1 bar
for the next batch. (The pinned val only gates the 1-seed screen; every confirm
re-runs the champion fresh, so an optimistically-low pin self-corrects.)

## What is NOT in scope (still hard rules)
- Still **one tier** — `tiny1m3m`. 3 seeds means 3 runs of the *same* tier, not a
  multi-tier ladder.
- Screening is still **strictly one seed**. No seed sweeps in Stage 1, no "add a
  seed to break a tie." A sub-noise Stage-1 effect is inconclusive, full stop.
- Three seeds is the **ceiling**, only for the promotion confirm. Not 5, not 10.

## Implementation (LIVE — wired 2026-06)
The two-stage gate is built and running:
1. **Stage-1 SCREEN** — `finalize_one` in `queue-daemon.sh` judges the 1-seed run
   vs `champion.json` val at `SCREEN_BAND` (0.015). The lucky-seed guard flips a
   SCREEN-WIN to **`needs-confirm`** instead of auto-promoting — nothing promotes
   off Stage 1.
2. **Stage-2 CONFIRM** — `bin/confirm_paired.py <idea> <flags>` runs both arms at
   seeds `42,123,7` back-to-back on one box, judges
   `mean3(chal) < mean3(champ) − 0.001` **AND** all-3-seeds-agree, writes
   `confirm-paired.md`, and (with `--promote`) re-pins `champion.json`.

Champion config_class re-pin is a deliberate step (env-driven champions like 296/323
can't be expressed by a flag list alone — see the hand-rolled `_arq_confirm_*.py`).
