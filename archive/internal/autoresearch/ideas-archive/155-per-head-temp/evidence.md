# Evidence — 155 Per-Head Learnable Attention Temperature

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060, sm_86, 12GB)
- baseline: cached mean=6.4394 ±0.04 (box 5b8a7fea8963, 3 ctrls measured 2026-06-14T00:13:56Z)
- treatment val: 6.4331   Δ vs baseline: −0.0063
- train: 6.3954   val_acc: 0.1427   step-0 val: 10.8125 (bit-identical to ctrl)
- bpb: n/a (pending harness)
- pass/fail bar (plan.md): PASS trt < cached_mean - 0.04; NULL inside band; DRIFT > +0.01 → **not met** (Δ inside band)
- box check: cached mean 6.4394 vs leaderboard 6.4306 (within 0.009 of noise band, healthy)
- raw: remote-results/2026-06-14-vast-tiny1m3m/results.json (log: 155-per-head-temp_52674.log)
- date: 2026-06-14

## Transfer note
The per-head temperature lever adds H=4 learnable scalars per attention layer (one per head), initialized to `τ_h = 1/√d_k` so step-0 attention scaling is bit-identical to the standard `QK^T/√d_k` path. As training proceeds each head can sharpen or broaden its own softmax. At 0.94M/12L/4H the optimizer absorbs the 48 per-head scalars into the Q/K gradient updates — train_loss is 0.03 below the ctrl cluster (right sign) but val_loss Δ is inside the 0.04 noise band, so the small generalization improvement is at noise. This closes the per-head-attention-shape axis at 0.94M alongside the closed 016-qk_norm and 152-attn-logit_bias axes — three different per-head-attention-shape levers (norm on Q/K, bias on QK^T logits, scalar on logits temperature) all close null at this tier. The mechanism is qualitatively validated upstream at PaLM 2/OLMo 2/Gemma 2 (≥7B) where head specialization is meaningful: at 0.94M/4H the heads are too few and the residual stream too narrow for any per-head-axis lever to find a non-trivial specialization pattern the Q/K weights don't already span. Re-evaluate at >=135M Phase-2 with deeper stacks (L=24+) and more heads (H=12+) where each head has a distinct prior to develop.
