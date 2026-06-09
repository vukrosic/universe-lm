---
id: 020-forgetting-attn
status: planning
round: 1
updated: 2026-06-09T13:48:40Z
---

# 020 — Forgetting Transformer (per-head learnable decay multiplier)

## Source
"Forgetting Transformer: Softmax Attention with a Forget Gate" (Lin,
Wang, Yang et al.; March 2025 — arXiv:2503.02130). Adds a per-head,
per-token learnable forget gate to standard softmax attention; reports
consistent wins over RoPE/SWA baselines at GPT-2 and Llama scales.

## Mechanism
Standard softmax attention computes `A = softmax(QK^T / √d)`. Forgetting
Transformer (FoX) multiplies the attention matrix element-wise by a
causal decay kernel built from a per-head, per-token sigmoid forget
gate:

```
f_h,t = sigmoid(W_f^h · x_t + b_f^h)            # scalar in (0,1) per head h, token t
log_f_h = log(f_h,t)                            # shape (B, H, T)
cf_h    = cumsum(log_f_h, dim=T)                # cumulative log-gate up to t
D_h[i,j] = exp(cf_h[i] - cf_h[j])  (j ≤ i)      # = prod_{k=j+1..i} f_h,k, lower-tri
A_h     = softmax(Q_h K_h^T / √d) ⊙ D_h         # then renormalize rows of A_h
y_h     = A_h V_h
```

`D_h[i,j]` is a *learned* exponential-style decay between query position
`i` and key position `j` — the model can choose to forget far tokens
fast (small `f_h`) or keep them (`f_h → 1`). Implemented in < 90 LoC on
top of `MHA.forward`: one extra `nn.Linear(d_model, H)` (per-head gate
logit projection), one `cumsum` of `log f` along the time axis to build
the kernel in O(T) memory, one elementwise multiply on the attention
matrix followed by a row-renorm. At a 6-layer model with `T = 2048`
this adds ~5% to the attention forward.

### Identity-init (corrected for T = 2048)
The r1 init (`b_f^h = +5`) is **not** near-identity at our
`max_seq_len = 2048` (`configs/llm_config.py:25`): `f = sigmoid(5) ≈
0.9933`, `log f ≈ -0.00672`, so `D_h[0, 2047] = exp(-0.00672 · 2047) =
exp(-13.76) ≈ 1e-6` — a token at position 0 cannot attend to the last
token at all under that init. To get true near-identity at T=2048 we
need `log f · 2047 ≈ -0.1` (≤ 10% decay over the full context), i.e.
`b_f^h = +10` (sigmoid(10) ≈ 0.99995, log f ≈ -5e-5,
`D_h[0, 2047] = exp(-0.1) ≈ 0.90`).

Use `b_f^h = +10` and `W_f^h = 0`. At step 0 the gate is within ~10% of
1 over the full context, so row-renorm recovers a baseline-equivalent
attention pattern to ~2 decimals; the model still has to *learn* to
forget from scratch (gates start nearly 1.0, can only go down). The
test in `LoC budget` (e) pins this as an assertion: with
`use_fox=True` and the `W_f=0, b_f=+10` init, the output is within
`1e-5` of the `use_fox=False` output.

## Why it's worth a slot
The bet: 009-fire-pe won big and is an **additive** positional bias on
attention *logits*; FoX is a **multiplicative** decay on attention
*probabilities* — a strictly orthogonal mechanism. Additive bias
changes *which* key wins the softmax; multiplicative decay changes
*how much mass* even the winners keep. If FoX wins on top of FIRE, the
missing axis is mass control; if it nulls on top of FIRE, FIRE's
additive bias already saturates the relative-position benefit at our
scale. Distinct from every closed lever: NSA/diff-attn modify the
*score function*, SWA/dilated attention modify the *mask*, RoPE/NoPE
modify the *position encoding*, logit-softcap clamps the *logit
range* — none introduce a *content-conditional, per-head, per-token
learned decay multiplier on A*. Distinct from the closed delta-net /
linear-attention family (008/012) because the softmax stays — FoX is a
*conservative extension* of softmax attention, not a replacement of it.
< 90 LoC, single boolean flag, identity-init clean (with the corrected
`b_f = +10`).

## Definition (gate 2)

### Ctrl vs trt
- **Ctrl**: `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`
  (`configs/llm_config.py:695` + 009 FIRE flag). The 009 WIN signature
  in `closed.md:40` is `trt=6.3234 vs ctrls 6.3875/6.4050` — that
  FIRE-equipped config *is* the 009 trt. Pinned to the FIRE-equipped
  baseline so the A/B partitions the orthogonal-axis question, not the
  "does FIRE win?" question.
- **Trt**: same config + `use_fox=True`.

### Pass bar (tiny1m3m noise floor)
Run-to-run val-loss variance at this tier is ≈ ±0.01
(`closed.md:33-40` ctrls span 6.3875–6.4050 = 0.018 spread). With a
single seed the pass bar must clear the ctrl-gap and not just sit
inside it:
- **Win**: `trt_val < ctrl_val − 0.02`.
- **Null**: `|trt_val − ctrl_val| < 0.02` (sub-noise; the lever does
  not fire on top of FIRE at this scale).
- **Fail**: `trt_val > ctrl_val + 0.01` (worse than baseline by more
  than half the ctrl-gap — the multiplicative gate is hurting).

### Seed
**Seed 42 only.** Single fixed seed, no multi-seed sweep, no per-seed
mean. A sub-noise delta is *inconclusive, not real*; never add "run
more seeds to confirm" — log null and move on.

### LoC budget (≤ 50 LoC, well under the 200 ceiling)
- (a) per-head gate `nn.Linear(d_model, H)` + buffer for `b_f^h = +10`
  init: ≈ 12 LoC
- (b) cumsum path (`log f` → `cumsum` → broadcast `D_h`) +
  elementwise multiply on `A_h`: ≈ 8 LoC
- (c) row-renorm of `A_h` (divide by per-row sum): ≈ 4 LoC
- (d) flag wiring (`use_fox: bool = False` in MHA + pass-through from
  `TransformerBlock` + `LLMConfig` plumbing): ≈ 8 LoC
- (e) one test asserting `use_fox=False` ≡ baseline at step 0 *and*
  `use_fox=True, W_f=0, b_f=+10` ≡ baseline at step 0 within `1e-5`:
  ≈ 15 LoC

Total ≈ 47 LoC.
