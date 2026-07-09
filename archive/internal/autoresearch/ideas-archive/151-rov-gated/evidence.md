# Evidence — 151 Gated Rotary Value Embeddings (RoV)

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674
- baseline: cached mean=6.4302 ±0.04 (box 5b8a7fea8963, measured 2026-06-14)
- treatment val: 6.4416   Δ vs baseline: +0.0114
- bpb: n/a (pending harness)
- pass/fail bar: PASS trt < cached_mean − 0.04; NULL inside band  → **met (NULL band)**
- box check: cached mean 6.4302 vs leaderboard ctrl 6.4306 (within noise)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (log: 151-rov-gated_52674.log)
- date: 2026-06-14

## Transfer note
Intra-layer rotary on V (per-block scalar `rov_gate=0` at init, gated
mix `V + g·V_rot`). At 0.94M / 92 update steps the gate stays pinned
near 0 across the run — the model has no measurable reason to rotate
V by position because Q and K already carry that signal into the
attention weights, and the additional V-rotation does not buy
anything the softmax routing can't already express. Train_loss trt
6.4291 sits in the ctrl cluster (6.42–6.43), confirming the lever is
forward-pass-active but never mixes the rotated V. Mechanism
hypothesis (output position should know *which* input position it
came from) is sound in principle but the *binding constraint* at our
tier is not V-position-blindness — it's the body rank, not the value
side. Orthogonal to 021-value-residual (cross-layer V reuse, won with
caveat) and 009-FIRE-PE (Q,K rotary, won); closes the V-rotation
axis at 0.94M. Re-test at 135M where the FFN is wider and V may carry
more independent per-position info is a future play, but the null is
informative — V-position-coupling is not the lever at this tier.
