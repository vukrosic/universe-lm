# ALiBi Deep-Dive — closed experiments ledger

Independent research track studying the **175-alibi record** (learnable per-head
linear distance bias, slopes init 0). Baseline = plain ALiBi measured fresh on
this thread's own box (vast :55010), val **6.2403** (seed 42; → 3-seed mean when
RQ1 completes). Full design + live results: `experiments.md`.

This board's record-to-beat is the ALiBi baseline above; experiment arms
(kernel shape RQ1, slope-init sweep RQ2) are appended below as they finish.

## Closed by the loop

RQ1 (kernel shape) and RQ2 (slope-init sweep) verdicts are appended here as
one-line records once the runs complete (see experiments.md for the live tables).

- RQ1-poly — null: trt 3-seed mean 6.2556 vs alibi 6.2584 (paired Δ-0.0028, deep inside 0.02 band) at tiny1m3m; convex `c_h·d²/L` curvature buys nothing on plain alibi (mirrors main-track 230-poly-alibi standalone null) — 2026-06-16
- RQ1-kerple — null/LOSS: trt 3-seed mean 6.3145 vs alibi 6.2584 (paired Δ+0.0561, ~2.8× band, wrong sign, all 3 seeds) at tiny1m3m; concave log-distance penalty is the wrong direction — at 0.94M the model wants a HARDER locality prior, not softer; closes the concave-kernel axis — 2026-06-16
- RQ2-geo1frz — WIN(weak): trt 6-seed mean 6.2428 vs zero (record) 6.2609 (paired Δ-0.0181±0.0081 SEM, t≈-2.24) at tiny1m3m; true fixed classic ALiBi (geometric slopes 2^-8h/H, frozen) beats learning slopes from 0 — 2026-06-16
- RQ2-geo2lrn — WIN: trt 6-seed mean 6.2368 vs zero (record) 6.2609 (paired Δ-0.0242±0.0119 SEM, t≈-2.03; seeds 42/123/7/11/22/33) at tiny1m3m; geometric slope init ×2 then learnable. Beats the zero-init ALiBi record AND the lucky single-seed 6.2403. Finding: the slope INIT/magnitude is a bigger lever than the kernel SHAPE — learning from 0 underfits the locality prior in 92 steps. [SUPERSEDED by geo3lrn below; exp5 found a stronger scale] — 2026-06-16
- exp5-geo3lrn — WIN(new leader): trt 3-seed mean 6.2301 vs zero 6.2584 (paired Δ-0.028, all 3 seeds below record) at tiny1m3m; geometric slope init ×3, learnable. Beats geo2lrn (6.2353). The locality-strength hill peaks at ~3×: trend zero 6.2584 → 1× 6.2428 → 2× 6.2353 → 3× 6.2301 → 4× 6.2315. NEW THREAD BEST 6.2301 (pending exp6 6-seed confirm on seeds 11/22/33) — 2026-06-16
- exp5-geo4lrn — WIN(plateau): trt 3-seed mean 6.2315 ≈ geo3lrn (Δ+0.0014) at tiny1m3m; 4× geometric, learnable. Optimum is a broad 3–4× plateau; single best run anywhere = geo4lrn seed 123 = 6.2213 — 2026-06-16
- exp5-geo2frz — null vs geo2lrn: trt 3-seed mean 6.2383 vs geo2lrn 6.2353 (Δ+0.0030) at tiny1m3m; freezing the strong 2× prior is slightly worse than learning from it — at the winning scale, learnability still buys a little (but freezing still beats zero-init: 6.2383 < 6.2584) — 2026-06-16
