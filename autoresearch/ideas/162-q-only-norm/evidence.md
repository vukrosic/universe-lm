# Evidence — 162-q-only-norm

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86, driver 580.159.03)
- baseline: fresh N=3 ctrls this queue (MEASURE path — commit changed d161390→0dd4e7d) — mean=6.4346, std=0.0229, noise_band=0.0458 (max(0.04, 2·std)), range=0.0534
- treatment val: 6.4303   Δ vs fresh baseline: −0.0043   Δ vs leaderboard ctrl (6.4306): −0.0003
- bpb: n/a (pending harness)
- pass/fail bar: −0.005   → **not met** (Δ=−0.0043 misses by 0.0007, well inside the 0.0458 noise band)
- box check: fresh ctrl mean 6.4346 vs leaderboard 6.4306 (Δ=+0.0040, well inside 0.04 band) — no drift
- raw: autoresearch/remote-results/2026-06-14-vast-tiny1m3m-5/results.json (logs alongside)
- date: 2026-06-14

## Transfer note
RMSNorm family is well-validated at 1B+ (LLaMA 3, Qwen 2.5, Mistral) and asymmetric QK normalization at 35B+ (Cohere Command-R). Transfer risk is **low**. The treatment stayed within seed-42 box variance at 0.94M (Δ=−0.0043 vs fresh mean; Δ=−0.0003 vs leaderboard ctrl), so we cannot conclude the lever is harmful at 0.94M — it just didn't help either. **Attribution insight** (the whole point of this axis): 162 (Q-only) NULL alongside the 016-qk-norm (QK both) WIN at tiny1m3m means the 016 WIN was carried by **K-side normalization or the QK symmetry**, NOT by Q-side specifically. With 165-k-only-norm filed as the orthogonal K-side ablation, the three-way attribution test for the 016 WIN is now: 162 NULL (Q-only, this run) + 165 = ? (K-only) + 016 WIN (QK both). If 165 also NULLs, the 016 WIN came from the **symmetry** of normalizing both. If 165 WINS, the K-side was carrying it. Either closure is the point — the 162-Q arm is now closed. At 1B+ transfer the lever likely has the same ±0.04 null band on absolute val loss, so we wouldn't recommend this for a production 1B model without re-running the 3-way at the target scale.
