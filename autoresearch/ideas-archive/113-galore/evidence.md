# Evidence — 113 GaLore (Gradient Low-Rank Projection)

## Verdict: NULL (DRIFT)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (5b8a7fea8963, RTX 3060, sm_86, driver 580.159.03, commit 42ed363)
- baseline: fresh 3 ctrls this queue (6.4259, 6.4416, 6.4453 → mean=6.4376 ±0.0084, band=0.04); MEASURE path triggered by commit-changed (cache 7f7fe90 → box 42ed363)
- treatment val: **7.1609**   train=7.0982   acc=0.0936   Δ vs fresh ctrl mean: **+0.7233**
- bpb: n/a (pending harness — never omit)
- pass/fail bar (plan.md): PASS Δ≤-0.005; NULL |Δ|<0.005; DRIFT >+0.005  → **Δ=+0.7233 ≫ +0.005 → DRIFT (degenerate)**
- box check: fresh ctrls 6.4259/6.4416/6.4453 vs leaderboard 6.4306 — max |Δ|=0.0147 (within 0.04 noise, **NO DRIFT**)
- raw: autoresearch/remote-results/2026-06-14-vast-tiny1m3m-3/results.json (log 113-galore.log alongside)
- date: 2026-06-14

## Run trace
- step 0: val_loss=10.8125, val_acc=0.0000 (bit-identity preserved at init ✓)
- step 100: ~7.7 (typical ctrl is ~7.4 at this point — already drifting)
- step 400: ~7.0 (typical ctrl is ~6.6 — clearly off-track)
- final (step 732): val_loss=**7.1609**, train=7.0982, val_acc=0.0936 (vs ctrl val_acc 0.1443)

The GaLore QR-projection path learned but under-converged severely at 0.94M/12L. The train_loss gap (+0.70 vs ctrl) is the dominant signal — the optimizer's projection-every-k steps + AdamW-on-rank-r is not converging at this update-step horizon, even though the implementation itself runs without exception (the prior round-2 BFloat16 crash on `torch.linalg.qr` is fixed in this round; the lever itself just doesn't work at this tier).

## Transfer note
GaLore is a memory-reduction tool at its source-paper headline — parity val loss at ÷2 memory on Llama 1B/7B. The *quality* axis the paper also reports (modest val-loss gains at rank-r=4-256 on the same val loss) requires a longer horizon + larger 2-D bucket to amortize the QR/SVD projection cost. At 0.94M/12L the 2-D matrices are so small (d_model=64, d_ff=256) that the projection overhead is essentially the entire update — the AdamW-on-rank-r fallback fires on every step and the effective learning rate is starved. The lever is closed at tiny1m3m on this pass; re-evaluate at any future mid-scale tier where GaLore's memory story actually applies. Quality at scale remains an open question — picoGPT/nanoGPT-speedrun re-implementations give similar parity val_loss, not gains.