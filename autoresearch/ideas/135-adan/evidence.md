# Evidence — 135 adan

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   ctrl2 val: 6.4419   treatment val: 6.4819   Δ vs ctrl: +0.0547
- train_loss: ctrl 6.3966, treatment 6.4285 (Δ +0.0319)
- val_accuracy: ctrl 0.1459, treatment 0.1485
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (≈ 6.4222); null band |Δ|<0.01, DRIFT > +0.01
  → Δ=+0.0547 wrong-sign, ~5× the 0.01 null band → **NULL**
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise);
  ctrl2 6.4419 vs ctrl 6.4272 = +0.0147 (ctrl-to-ctrl gap, within ~0.04 measured variance)
- raw: remote-results/2026-06-13-vast-tiny1m3m/arq-110/135-adan_52674.log, ctrl2_52674.log
- date: 2026-06-13

## Transfer note
Adan (Xie et al. 2022, arXiv:2208.06677) replaces AdamW's 1-step second-moment
EMA with an N-step lookback variance `v_t = β2·v_{t-1} + (1−β2)·mean(g_{t-i}²)
for i=0..N-1`, plus a Nesterov-style lookahead gradient
`g_la = g_t + β_la·(g_t − g_{t-1})`. Validated at ≥100M (ViT-L 307M, CogVLM 7B);
transfer-risk tag in idea.md is **low** (mechanism is scale-free). At 0.94M
with only 92 update steps, the N=4 lookback queue is only ~4% full at end of
training — the variance estimate barely has time to integrate. Combined with
`adan_lr=0.006` (~4× smaller than the AdamW peak 0.024 used by the baseline
because the Nesterov lookahead can over-shoot early), the lever under-shoots.
Same horizon-scaling null pattern as 110-weight-ema, 122-tiger, 124-radam,
134-mega-ema — all adaptive / multi-step EMA-style optimizers that need
3-4k update steps to develop their distinctive behaviour. Re-evaluate at
≥135M Phase-2 (3-4k steps) where the N=4 lookback has time to integrate and
Adan's mechanism-level claim (longer-range signal) can actually be tested.
Closed; not promoted.
