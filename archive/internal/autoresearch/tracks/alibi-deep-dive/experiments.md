# ALiBi Deep-Dive — experiments log

Mechanism under study: the lab's flagship record **175-alibi** — a learnable
per-head linear distance bias on attention scores, `scores − m_h·(j−i)`, with
`m_h` a trained parameter **initialized at 0** (NOT classic fixed-geometric
ALiBi). All runs: tiny1m3m (0.94M params, ~3M tokens, 732 steps), seed-paired,
on the dedicated box `1.208.108.242:55010` (RTX 3060). Baseline measured fresh on
this box to remove cross-box drift.

Empirical note: the learned slopes converge **negative** (mean ≈ −0.064, range
≈ −0.22…+0.01 across 12 layers × 4 heads). Their magnitudes are well *below* the
classic geometric slopes (2^−8h/H ranges 0.25…0.004 for H=4) — motivating RQ2.

---

## RQ1 — kernel shape: linear vs polynomial vs log-distance

Does a different *shape* of monotone distance penalty beat linear ALiBi? Three
mutually-exclusive kernels, each per-head and **identity at step 0** (fair):

| arm | config | kernel |
|---|---|---|
| `alibi` (baseline = the record) | `Tiny1M3MAlibiConfig` | linear `−m_h·d` |
| `poly` | `Tiny1M3MPolyAlibiConfig` | `−(m_h·d + c_h·d²/L)` (adds curvature) |
| `kerple` | `Tiny1M3MKerpleLogConfig` | `−m_h·log(1+r_h·d)` (concave) |

3 seeds (42/123/7) × 3 arms = 9 runs. Verdict = trt 3-seed mean vs `alibi`
3-seed mean, paired within seed.

**Results (val loss; lower = better) — DONE 2026-06-16:**

| seed | alibi | poly | kerple |
|---|---|---|---|
| 42 | 6.2403 | 6.2428 | 6.2988 |
| 123 | 6.2641 | 6.2663 | 6.3197 |
| 7 | 6.2709 | 6.2578 | 6.3250 |
| **mean** | **6.2584** | **6.2556** | **6.3145** |

> alibi seed-42 = 6.2403 matches the historical 175-alibi seed-42 value (6.2403)
> → this box reproduces the record; baseline is trustworthy.

**Verdict — linear ALiBi is the right shape.**
- **poly** (adds convex `c_h·d²/L` curvature): paired Δ **−0.0028** vs alibi — **NULL**, deep inside the 0.02 band. The same marginal result the main track saw (`230-poly-alibi` standalone was null; it only bound stacked on DeepNet-α). Curvature buys nothing on plain alibi.
- **kerple** (concave log-distance): paired Δ **+0.0561** — a clear, consistent **LOSS** on all 3 seeds. A gentler far-token penalty is the wrong direction; at 0.94M the model wants a *harder* locality prior, not a softer one.
- **Conclusion:** the *shape* of the distance penalty is settled — linear is right, concave loses, convex ties. The headroom is not in the kernel. → RQ2 asks where it actually is.

---

## RQ2 — slope init & learnability (the hyperparameter sweep)

The slopes are learned from 0 and stay small. Does seeding them at the classic
geometric ALiBi magnitudes (correct negative sign), and/or freezing them, help?
Implemented as an env-driven knob in `models/layers.py` on the box only
(`ALIBI_SLOPE_INIT`/`ALIBI_SLOPE_SCALE`/`ALIBI_SLOPE_LEARNABLE`; default = zeros,
learnable = byte-identical to the record). All arms `Tiny1M3MAlibiConfig`.

| arm | init | scale | learnable | meaning |
|---|---|---|---|---|
| `zero` (baseline) | 0 | – | yes | the record (learn from 0) |
| `geo1lrn` | geometric | 1× | yes | warm-start at classic slopes, then learn |
| `geo1frz` | geometric | 1× | **no** | true fixed classic ALiBi |
| `geo2lrn` | geometric | 2× | yes | stronger locality prior, then learn |

3 seeds × 4 arms = 12 runs. Verdict vs the `zero` arm, paired within seed.

**Results (val loss; lower = better) — DONE 2026-06-16.** First 3 seeds (42/123/7)
from exp2; a tie-breaker batch (exp4) added 3 fresh seeds (11/22/33) on the
decisive arms `zero` / `geo1frz` / `geo2lrn` for a 6-seed paired test.

| seed | zero (record) | geo1lrn | geo1frz | geo2lrn |
|---|---|---|---|---|
| 42  | 6.2403 | 6.2531 | 6.2438 | 6.2453 |
| 123 | 6.2641 | 6.2453 | 6.2366 | 6.2344 |
| 7   | 6.2709 | 6.2297 | 6.2359 | 6.2263 |
| 11  | 6.2359 | –      | 6.2412 | 6.2541 |
| 22  | 6.2884 | –      | 6.2466 | 6.2341 |
| 33  | 6.2659 | –      | 6.2525 | 6.2263 |
| **mean (6-seed)** | **6.2609** | _6.2427 (3s)_ | **6.2428** | **6.2368** |

**Verdict — the slope INIT matters more than the kernel shape (the real finding).**
- **geo2lrn** (geometric slopes × 2, then learn): paired Δ **−0.0242 ± 0.0119 SEM**
  vs zero (t≈−2.0, 6 seeds). Beats the zero-init record — and beats even the lucky
  single-seed record 6.2403. **New thread best: 6.2368.**
- **geo1frz** (true fixed classic ALiBi, no learning): paired Δ **−0.0181 ± 0.0081**.
  Just freezing the slopes at the classic magnitudes already beats learning from 0.
- **Trend is monotone in locality strength:** zero 6.2609 → 1× 6.2428 → 2× 6.2368.
  Learning the slopes from 0 *underfits the locality prior* in 92 steps; warm-starting
  at the classic geometric magnitudes (and scaling them up) supplies the prior the
  optimizer can't discover in time. This is the headroom RQ1 said wasn't in the shape.

---

## RQ2-followup — slope-SCALE sweep (exp5, in flight)

The zero→1×→2× trend says push the scale past 2×. Arms (init=geometric, seeds
42/123/7), judged vs zero (record) and vs geo2lrn (current best):

| arm | scale | learnable | question |
|---|---|---|---|
| `geo3lrn` | 3× | yes | does the locality prior keep helping past 2×? |
| `geo4lrn` | 4× | yes | … or turn over? |
| `geo2frz` | 2× | **no** | is learnability needed at the winning scale, or is fixed enough? |

(planned `geo6lrn` 6× dropped — 4× already turned over, so the optimum is bracketed
between 2× and 4×, no need to probe further out.)

**Results (val loss; lower = better) — DONE 2026-06-16 ~10:25 UTC.** New arms on
seeds 42/123/7, paired against the existing `zero` and `geo2lrn` runs (same seeds):

| seed | zero (record) | geo2lrn (2×) | geo3lrn (3×) | geo4lrn (4×) | geo2frz (2× frz) |
|---|---|---|---|---|---|
| 42  | 6.2403 | 6.2453 | 6.2328 | 6.2462 | 6.2519 |
| 123 | 6.2641 | 6.2344 | 6.2294 | 6.2213 | 6.2256 |
| 7   | 6.2709 | 6.2263 | 6.2281 | 6.2269 | 6.2375 |
| **mean (3-seed)** | 6.2584 | 6.2353 | **6.2301** | 6.2315 | 6.2383 |

**Verdict — the locality-strength hill peaks at ~3×; geo3lrn is the new thread best.**
- **geo3lrn (3× geometric, learnable): 6.2301**, lower than every prior arm and below
  all three seeds of the `zero` record (paired Δ vs zero = **−0.028**, but ±0.046 at
  3 seeds — all-negative yet not band-clearing because seed 42 is stubborn). New leader.
- **geo4lrn (4×) = 6.2315** ≈ geo3lrn (Δ +0.0014) — the curve flattens then dips: the
  optimum is a broad plateau around 3–4×, with the single best run anywhere being
  geo4lrn seed 123 = **6.2213**.
- **geo2frz (2× frozen) = 6.2383** — freezing the strong prior is *worse* than learning
  from it (geo2lrn 6.2353): at the winning scale, learnability still buys a little.
- **Updated trend:** zero 6.2584 → 1× 6.2428 → 2× 6.2353 → **3× 6.2301** → 4× 6.2315.

**Confirm in flight (exp6):** geo3lrn + geo4lrn on 3 fresh seeds (11/22/33) → 6-seed
paired test vs zero, to settle whether the strong-slope record clears the noise band.

---

## Status

- RQ1 (kernel) + RQ2 (slope-init) + ablation (no-bias) + 6-seed tie-breaker + exp5
  slope-scale sweep: **DONE**.
- exp6 strength-confirm: **running** on :55010 (tmux `exp6`) — geo3lrn/geo4lrn × 11/22/33.
- Current thread best: **geo3lrn 6.2301** (3× geometric init, learnable) — beats the
  zero-init ALiBi record (6.2584 honest 3-seed mean; 6.2403 lucky single seed) on all
  three seeds. 6-seed confirm in flight to clear the band.
- Findings → basis for the post
  *"The shape of distance-decay: tuning ALiBi for a 1M-parameter transformer."*
