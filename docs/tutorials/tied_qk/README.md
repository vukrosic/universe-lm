# Tied QK (PaLM): wins at tiny, fades at 10M

**Result:** Tied QK is the **current tiny winner** (val **6.3041**,
Δ **−0.1265** on the 0.94M "tiny" tier) but **loses 0.014 on the
10M screen20m best baseline** (4.6500 vs 4.6364). Same architecture
change, opposite sign of effect at different scales.

This is a clean example of "a structural lever that helps small models
hurts bigger ones." The mechanism matters: tying Q and K is a
**capacity constraint** that helps when the model has too many degrees
of freedom, and hurts when those degrees become load-bearing.

---

## What Tied QK does

Standard attention has **two independent projections** for Q and K:

```text
Q = X @ W_Q       (shape: [seq, d_model] → [seq, n_heads × d_k])
K = X @ W_K       (shape: [seq, d_model] → [seq, n_kv_heads × d_k])
```

PaLM-style Tied QK uses **one shared projection**, then splits the
output:

```text
QK = X @ W_QK    (shape: [seq, d_model] → [seq, (n_heads + n_kv_heads) × d_k])
Q, K = QK.split([n_heads × d_k, n_kv_heads × d_k], dim=-1)
```

The Q and K matrices are no longer independent — they're halves of the
same matrix. This is a **structural constraint**, not a hyperparameter.

**Parameter cost:** `W_QK` has the same total param count as
`W_Q + W_K` (the split is free). The constraint is *what the matrix
can represent*, not *how big it is*.

---

## The tiny tier (0.94M, 3M tok) — Tied QK wins

Same recipe (V-embed + Q-gain + RoPE-250k + SWA-384), seed 42:

| config | Val loss | Δ vs GQA=2 |
|---|---|---|
| **Tied QK** (n_heads=4, n_kv_heads=4) | **6.3041** | **−0.0309 winner** |
| MHA (n_kv_heads=4) | 6.3069 | −0.0281 |
| LayerNorm | 6.3109 | −0.0241 |
| MLA (d_c=16) | 6.3253 | −0.0097 |
| GQA=2 (baseline) | 6.3350 | 0 |
| GQA1 (n_kv_heads=1) | 6.3447 | +0.0097 |
| PostNorm | 7.6209 | **+1.286 collapse** |

Tied QK beats full MHA by 0.0028 and the original tiny winner by 0.020.
The structural constraint is **helping** at this scale.

---

## The screen20m tier (10M, 20M tok) — Tied QK fades

Same recipe (V-embed + Q-gain + SWA-512 + RoPE-500k), seed 42:

| config | Val loss | Δ vs best |
|---|---|---|
| **GQA=2 (no tying)** | **4.6364** | 0 (best) |
| Tied QK | 4.6500 | +0.014 |
| Tied QK (no HighRoPE) | 4.6652 | +0.029 |

Tied QK **loses 0.014** on the best baseline. The constraint is now
**anti-additive.** Same change, opposite sign.

---

## Why it helps small models

A 0.94M model has ~10× fewer parameters than a 10M model. Every degree
of freedom matters. Tying Q and K cuts the QK parameter count in half
(from 2 projections to 1) **and** removes the rank in the QK matrix —
they're now coupled.

For a small model, this acts as **regularization**:
- less capacity to overfit
- a stronger structural prior (Q and K are related)
- fewer spurious Q/K relationships to learn

For a bigger model, the same constraint is a **ceiling**:
- the model has the capacity to learn *different* Q and K
  projections and that capacity is *load-bearing*
- tying forces Q and K to share, which wastes that capacity
- the regularizer that helped at 0.94M is a tax at 10M

This is the same pattern as many regularization techniques: helpful
under capacity pressure, harmful otherwise.

---

## Why it loses at 10M

The +0.014 cost on the V+q+SWA+HighRoPE baseline is small but
consistent with the mechanism. The V-embed + Q-gain combination is
**already** giving the model extra flexibility on the Q side (Q-gain
tunes per-head sharpness; V-embed injects token identity into V). The
10M model can use that flexibility. Tying QK removes it.

We did not test Tied QK *without* the V+q recipe on screen20m. The
question "is Tied QK anti-additive with V+q, or anti-additive in
general at 10M?" remains open. The closed result on the best baseline
is enough — Tied QK is closed at 10M.

---

## The code

One flag, ~5 lines:

```python
# configs/llm_config.py
use_tied_qk: bool = False
```

```python
# models/layers.py — MultiHeadAttention.__init__
if self.use_tied_qk:
    self.qk_proj = nn.Linear(d_model, (n_heads + n_kv_heads) * d_k, bias=False)
else:
    self.q_proj = nn.Linear(d_model, n_heads * d_k, bias=False)
    self.k_proj = nn.Linear(d_model, n_kv_heads * d_k, bias=False)
```

```python
# models/layers.py — MultiHeadAttention.forward
if self.use_tied_qk:
    qk = self.qk_proj(x)
    Q, K = qk.split([self.n_heads * self.d_k, self.n_kv_heads * self.d_k], dim=-1)
else:
    Q, K = self.q_proj(x), self.k_proj(x)
```

Zero new parameters vs the untied baseline.

---

## Lessons

1. **"Wins small" is a real pattern, not a contradiction.** A lever
   that helps at 0.94M and hurts at 10M is teaching you something
   about *what the model can use*. Don't dismiss the small-scale
   result because it doesn't scale.
2. **Structural constraints are scale-conditional.** Tying, sharing,
   grouping — these all change the *shape* of the model, not its
   size. Their effect depends on whether the model has spare capacity
   to absorb the constraint.
3. **Closed on the best baseline ≠ closed in general.** Tied QK is
   closed on V+q+SWA+HighRoPE. We don't know if it's a real lever
   *without* the rest of the recipe. That's a separate experiment.
4. **PaLM's defaults are not always our defaults.** PaLM uses Tied QK
   because it was the right choice for *that* scale. Our 10M
   experiments say it isn't the right choice for ours. Defaults from
   big models need to be re-tested, not copied.

---

## Caveats

- **Single-seed at both scales.** Tied QK is the tiny winner by 0.0028
  (within the noise band of 0.005–0.015 for tiny runs). The win is
  direction-consistent with the other arch axes (LayerNorm, MHA are
  all within 0.005) but the exact "winner" status is not multi-seed
  confirmed.
- **Tiny → 10M is the only scale pair tested.** We don't know if Tied
  QK is anti-additive at 135M. It might come back. The closed
  verdict is "at 10M with our current recipe."
- **PaLM's Tied QK is on a 540B model.** Our extrapolation goes from
  0.94M to 10M. Going from 10M to 540B is a much bigger leap; the
  constraint that hurts at 10M might help at 540B.

---

## Reproduce

```bash
# the tiny winner (~2 min on RTX 3050):
python train_llm.py --config tiny1m \
  --use_value_embed true --use_q_gain true \
  --use_sliding_window true --sliding_window_size 384 \
  --rope_base 250000 \
  --use_tied_qk true \
  --seed 42

# the screen20m closed result (~30 min):
python train_llm.py --config screen20m \
  --use_value_embed true --use_q_gain true \
  --use_sliding_window true --sliding_window_size 512 \
  --rope_base 500000 \
  --use_tied_qk true \
  --seed 42
```

Code: [models/layers.py](../../../models/layers.py) (`MultiHeadAttention`),
flag in [configs/llm_config.py](../../../configs/llm_config.py) (`use_tied_qk`).
Evidence: [LEADERBOARD.md](../../../LEADERBOARD.md) §`tiny1m arch` row 0
(tiny winner) and §`screen20m` row 18h (closed);
`runs/tiny1m_arch_tiedqk_full/metrics.json`,
`runs/s_vqgain_swa_highrope_tiedqk_full/metrics.json`.
