# Evidence — 165 k-only-norm

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060 sm_86, box_key 5b8a7fea8963)
- baseline: cached mean=6.4455 ± 0.0524 (n=11 ctrl re-runs same seed; commit 2368d6c; measured 2026-06-14T09:00:17Z)
- treatment val: 6.4162   Δ vs cached mean: -0.0293   Δ vs latest ctrl 6.4247: -0.0085
- train_loss 6.3787 (vs ctrls 6.3866/6.4692/6.4128 etc., right-sign by ~0.03)   val_accuracy 0.1449
- bpb: n/a (pending harness)
- pass/fail bar: PASS ≤ 016-qk-norm 6.3906 − 0.005 = 6.3851; NULL |trt−ctrl| < 0.005; DRIFT > +0.005
  → 165 trt 6.4162 missed PASS bar by 0.0311 and missed the strict |Δ|<0.005 NULL band by 0.0035, BUT |Δ| < cache band (0.0524) so verdict is NULL per §2 of runner.md
- box check: latest ctrl 6.4247 vs leaderboard 6.4306 (Δ −0.0059, within noise); box healthy
- raw: remote-results/2026-06-14-vast-tiny1m3m/{165-k-only-norm_52674.log, results.json}
- date: 2026-06-14

## Transfer note
The 3-way orthogonal axis test (016 / 162 / 165) at 0.94M is now complete:
- 016-qk-norm (symmetric QK RMSNorm): **WIN** at 6.3906 (Δ −0.014 vs ctrl, plan bar −0.005 cleared 3×) — 2026-06-09
- 162-q-only-norm (Q-side only): NULL at 6.4303 (Δ −0.0043 vs fresh baseline 6.4346, inside |Δ|<0.005) — 2026-06-14
- 165-k-only-norm (K-side only): NULL at 6.4162 (Δ −0.0293 vs cached baseline 6.4455, inside 0.0524 cache band) — 2026-06-14

**Attribution conclusion:** 016's WIN was carried by the **joint QK symmetry / interaction effect**, not by the Q-side or K-side individually. Neither single-side RMSNorm clears the plan's PASS bar against 016's recorded val. K-only's negative Δ (−0.0293) is 7× larger in magnitude than Q-only's (−0.0043), suggesting a weak K-side preference, but the inter-arm difference 0.0250 is well inside the 0.0524 noise band and the plan's strict |Δ|<0.005 NULL window for each arm is missed by both.

**Why this is the right null.** The K-side lever did *not* replicate 016's symmetric gain. The mechanism (rescaling K vectors to unit RMS per head-dim) was correctly wired (build-smoke confirmed 949,248 = 949,056 + 192 params, `nn.RMSNorm(d_k=16)` × 12 blocks, no bias) and the run completed cleanly (final_train 6.3787, final_val 6.4162, no NaN/OOM). The lever is plausible and well-formed but at 0.94M/3M tokens the optimizer cannot find a non-trivial K-side specialization pattern that the standard `1/√d_k` inverse-temperature doesn't already span. The closure of the symmetric vs asymmetric attribution axis (with 016 WIN + 162 NULL + 165 NULL) suggests QK-norm's value at 0.94M comes from the *joint* normalization constraint — when both Q and K are forced to unit RMS, the QK inner product's scale is bounded, which is what gives the optimizer a more controlled logit landscape.

**Transfer to 135M Phase-2: low yield.** Cohere Command-R/R+ validates asymmetric QK normalization at 35B+, but our 0.94M/12L/4H stack is too small for head specialization to differentiate sides. The K-only axis is closed at 0.94M alongside the Q-only axis. Future K-side levers (per-layer K temperature, K gain) are plausible at 135M but the current data point argues against investing in K-only normalization specifically. Re-evaluate 016's *symmetric* form at 135M first (the WIN is the actual mechanism); the asymmetric attributions are a 0.94M noise-level effect.
