# Evidence — 170 swiglu-ffn

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_8.6, driver 580.159.03)
- baseline: MEASURE-pass (commit changed 2368d6c → f791d0c) — 3 fresh ctrls + 170 in same queue (`arq-170-measure-pass`); cache re-written by `baseline.sh measure` to `mean=6.4447, std=0.0244, band=0.0488, n=14, commit=f791d0c`
  - ctrl  (ts 10:07:47Z): val=6.4188, train=6.3953, val_acc=0.1437
  - ctrl2 (ts 10:09:53Z): val=6.4547, train=6.4399, val_acc=0.1403
  - ctrl3 (ts 10:12:00Z): val=6.4516, train=6.4104, val_acc=0.1382
  - session ctrl mean = 6.4417
- treatment val: 6.4247 (train=6.3893, val_acc=0.1418)   Δ vs session-ctrl-mean = -0.0170
- baseline.sh verdict: **NULL -0.020** (inside cache band 6.4447 ± 0.0488; trt not below mean-band = 6.3959)
- plan bar: PASS = trt ≤ 6.4344 (Δ ≤ −0.005); NULL = |Δ| < 0.01; DRIFT = Δ > +0.01
- Δ = -0.020 is below the −0.005 PASS threshold *numerically* but the cache-authoritative verdict is NULL because trt sits inside the noise band — `baseline.sh verdict` is the final word.
- bpb: n/a (pending harness)
- box check: ctrl cluster (6.4188/6.4547/6.4516) sits inside cache band ⇒ baseline in-range, no DRIFT signal
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (170 + ctrl3 entries appended; logs alongside)
- date: 2026-06-14

## Transfer note
SwiGLU is the standard FFN choice in LLaMA 1/2/3, Mistral, Qwen, Gemma, OLMo, Falcon — direct validation at 7B-540B with the 2/3-trick param parity. Shazeer 2020 (arXiv:2002.05202) original paper validates at T5 1.1B/1.6B/3B. **Transfer risk: low** (≥100M direct). At 0.94M / d_model=64 / 12L / 4H, the 2/3-trick FFN has 32,640 params vs 32,768 baseline (≈0.4% smaller) — gating has ~64×64 = 4096 maskable weights per block to learn from over 92 update steps. Same null pattern as 153-relu2-ffn (Δ=-0.0053, inside null band): activation-/gating-shape levers on the FFN do not bind at this tier. 153 closed FFN-activation axis; 170 closes FFN-gating axis. **Re-evaluate at ≥135M Phase-2** where FFN capacity is the binding bottleneck and gating has enough update steps + capacity for the gate to develop non-trivial statistics. Mechanism is well-validated at scale; the 0.94M null is a budget null, not a mechanism null.
