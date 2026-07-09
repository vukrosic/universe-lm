# Evidence — 168 av-output-carry

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674
- baseline: cached mean=6.4455 ±0.0524 (box 5b8a7fea8963, RTX 3060 sm_86 driver 580.159.03, measured 2026-06-14T09:00:17Z, n=11)
- treatment val: 6.4228   Δ vs baseline: -0.0227
- bpb: n/a (pending harness)
- pass/fail bar: WIN Δ ≤ -0.01; NULL |Δ| < 0.01; DRIFT Δ > +0.01  → plan-bar would say WIN (Δ=-0.0227 < -0.01), but cache verdict (Δ=-0.0227 inside ±0.0524) is NULL. Per §5 of runner.md both must clear for WIN → NULL (cache-authoritative).
- box check: baseline mean 6.4455 vs leaderboard 6.4216 — within noise (~0.024, < band 0.0524)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (log: 168-av-output-carry_52674.log)
- date: 2026-06-14

## Transfer note
Cross-block attention-output carry (post-AV pathway with learnable scalar α starting at 0) measured Δ=-0.0227 at tiny1m3m seed 42. Mechanically clean: α=0 init ⇒ bit-identical at step 0; α is the only new learnable scalar per block (12 blocks ⇒ +12 params, negligible). The negative trend is the largest of the three V/Q/AV-output attribution axes so far: 021-value-residual Δ=-0.034 (WIN), 168-av-output-carry Δ=-0.0227 (NULL, inside band), 164-q-carry was closed NULL. The hierarchy V > AV-output > Q is preserved — V-side wins at this tier, AV-output is borderline-null, Q-side is null. The AV-output pathway is more expensive than V-residual (extra scalar mix per block on every forward) without a clean tier win; it would not survive a Phase-2 promotion decision. The lever itself is sound but is not the right place to spend the param/FLOP budget at 0.94M.
