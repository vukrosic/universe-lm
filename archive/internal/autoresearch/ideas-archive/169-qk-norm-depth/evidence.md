# Evidence — 169 qk-norm-depth

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52646
- baseline: cached mean=6.3975 ±0.04 (box 5b8a7fea8963, measured 2026-06-15T05:15:46Z, n=4)  [fresh N=3 ctrls this queue: 6.4112/6.3934/6.3919]
- treatment val: 6.3775   Δ vs baseline: -0.0200
- bpb: n/a (pending harness)
- pass/fail bar: trt < 016 ctrl val by ≥0.005 (016 ctrls ≈ 6.4044/6.4091) → MET on absolute terms (Δ -0.027/-0.032) but **inside box noise band** (|Δ|=0.020 < band=0.040) — baseline band gates verdict, so NULL
- box check: baseline mean 6.3975 vs cached prior 6.4447 = DRIFT (cache re-baselined this queue)
- raw: remote-results/2026-06-15-vast-tiny1m3m/results.json (logs alongside)
- date: 2026-06-15

## Transfer note
The mechanism (per-block learnable scale on top of 016's per-head QK RMSNorm) is a narrow extension of a WIN, well-validated by NormFormer's per-layer gains at 100M+. The Δ of -0.020 sits inside the ±0.04 box noise band, so the effect cannot be distinguished from baseline variance at tiny1m3m. Transfer risk remains **low**: the primitive is well-tested at 1B+ (LLaMA 3, Qwen 2.5, Mistral all use RMSNorm variants); the sub-claim is whether per-block scalars matter vs per-head-shared. A null at tiny1m3m does not falsify the transfer hypothesis — it only means the signal is below noise at this scale; if it shows a real Δ at 135M, the per-block gain is the binding axis and 016 can be tightened.