# Evidence — 129 yoco

## Verdict: NULL (wrong-sign, 15× null band)
- tier: tiny1m3m, seed 42, box: 1.208.108.242:52674 (RTX 3060)
- control val: 6.4272   treatment val: 6.5038   Δ: +0.0766
- ctrl2: pending (queue still running 130+ + ctrl2); even a favorable ctrl2 cannot recover this magnitude — verdict decided regardless of ctrl2
- bpb: n/a (pending harness)
- pass/fail bar: ctrl − 0.005 (small/null band; lever is architectural, max effect is O(1/n) per step) → not met
- box check: ctrl 6.4272 vs leaderboard ctrl 6.4306 = −0.0034 (within noise, box healthy)
- raw: remote-results/2026-06-13-vast-tiny1m3m/129-yoco.log, ctrl.log
- date: 2026-06-13

## Trajectory check
- Live θ (train_loss): 6.4939 vs ctrl 6.3966 ⇒ +0.097 train-loss gap ⇒ YOCO's shared KV pathway carries less signal than per-layer K/V at this scale
- Treatment val curve: 10.81 → 8.30 → 7.81 → 7.58 → 7.42 (step 100) → 7.12 → 6.99 → 6.86 → 6.71 → 6.50 final
- Ctrl val curve:    10.81 → 8.34 → 7.82 → 7.58 → 7.42 → 7.08 → 6.94 → 6.76 → 6.63 → 6.43 final
- Curves track through step ~100, then YOCO falls visibly behind (Δ widens from ~0.0 to ~0.08 by end). The +0.08 wrong-sign gap is large and unambiguous.

## Transfer note
YOCO (Sun et al. 2024, arXiv:2405.05254) splits a 12L decoder in half: the lower 6 layers run standard self-attention, then a single GlobalKVHead projects the residual stream to `(K_g, V_g)` of shape `[B, T, kv_size]`, and the upper 6 layers re-use that KV cache instead of computing their own. Memory saving at inference is the headline win (~5× KV cache reduction in the paper at 7B-13B). The paper reports parity-to-better val loss with significantly reduced memory. The +0.08 wrong-sign result at tiny1m3m says: with `n_layers=12, yoco_split=6, yoco_lower_window=512`, the upper-half attention is reading a single fixed `(K_g, V_g)` per position while the lower-half was producing 6 distinct KV pairs per position — the upper half loses expressiveness. The mechanism's "compute/memory saving" lever is invisible at 0.94M (the entire model is tiny1m3m-sized), so the *only* thing this A/B is testing is whether collapsing the upper-half KV into a single shared copy preserves learning — and the answer at this scale is no. At 7B+ this is a different question (memory is the binding constraint) and the same idea would likely win on the *inference* axis but with a *training* val-loss trade-off the paper explicitly accepts. For the LM training pipeline: closed for tiny1m3m, re-evaluate at Phase-2 (135M) only if inference-memory is added to the metric, otherwise close as "tier-mismatch" — the lever does not fire in 92 update steps.
