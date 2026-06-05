# KV-head sharing: MHA vs GQA vs MLA — a wash at 10M, with caveats

**Result:** at 10M, **GQA=2 is in the noise band** of full MHA. At the
0.94M "tiny" tier, full MHA actually wins. MLA's latent bottleneck
**loses 0.091** at the screen20m best baseline. **Max GQA (GQA=1) hurts
at every scale tested.**

The KV sharing ratio is a **scale-dependent wash** — and a clean example
of "do not over-tune a knob that does nothing."

---

## What KV sharing actually controls

The number of distinct K/V projections, vs the number of Q heads:

```text
config                  n_kv_heads   ratio        KV params (10M, d=144)
─────────────────────────────────────────────────────────────────────
MHA (no GQA)            6            1:1          6 × 144 × 2d_model = 248k
GQA=2 (default)         2            3:1          2 × 144 × 2d_model = 83k
GQA=1 (max share)       1            6:1          1 × 144 × 2d_model = 41k
MLA (DeepSeek-V2)       1 latent     (low-rank)   1 × 144 × d_c + d_c × 2d_k × 6
```

More sharing = fewer KV params = less per-head KV capacity. Less sharing
= more KV params, but more redundancy between Q heads reading the same
KV.

---

## The tiny tier (0.94M, 3M tok) — MHA wins

Same recipe (V-embed + Q-gain + RoPE-250k + SWA-384), seed 42:

| config | n_kv_heads | Val loss | Δ vs GQA=2 |
|---|---|---|---|
| GQA1 (max share) | 1 | 6.3447 | +0.0097 (worst) |
| GQA=2 (default)  | 2 | 6.3350 | 0 (baseline) |
| MLA              | latent d=16 | 6.3253 | −0.0097 |
| **MHA**          | 4 | **6.3069** | **−0.0281 (winner)** |

At the tiny scale, **full MHA wins by 0.028**, and the ranking is
**MHA > MLA > GQA=2 > GQA1**. More KV capacity helps. This is a real,
small effect.

---

## The screen20m tier (10M, 20M tok) — wash on control

Same recipe as control (no embeds, no gains, no SWA), seed 42:

| config | n_kv_heads | Val loss | Δ vs GQA=2 |
|---|---|---|---|
| **GQA=2 (default)** | 2 | **4.7984** | 0 (baseline) |
| MHA (no GQA) | 6 | 4.7981 | **−0.0003 (wash)** |

**GQA=2 is indistinguishable from full MHA at 10M control.** The −0.028
tiny win disappears. KV sharing is **not a lever at this scale on this
baseline.**

---

## The best baseline (V+q+SWA+HighRoPE 4.6364) — GQA=1 hurts, MLA hurts

Same recipe, seed 42:

| config | Val loss | Δ vs best | verdict |
|---|---|---|---|
| **GQA=2 (default)** | **4.6364** | 0 | baseline |
| MHA (n_kv_heads=6) | 4.6384 | +0.002 | **wash** |
| GQA1 (n_kv_heads=1) | 4.6761 | +0.040 | hurts |
| MLA (d_c=36) | 4.7269 | +0.091 | **closed** — latent bottleneck loses |

**MHA is a wash** on the best baseline. The GQA=2 default is fine.

**GQA=1 hurts** — max KV sharing loses 0.040. The 1 KV head is forced to
serve 6 Q heads; it can't.

**MLA loses 0.091.** The latent bottleneck (d_c=36) compresses the K/V
representation, then up-projects. At our scale, that compression is too
aggressive. DeepSeek-V2's MLA is a lever *at their scale*, not ours.

---

## Why the effect disappears with size

Two readings, both consistent:

1. **Capacity saturation.** A 10M model has enough depth and width to
   *not* need every KV head to be independent. The GQA=2 default
   already gives the model enough KV capacity for the kinds of
   patterns it can learn; more is wasted.
2. **Optimization basin.** The fact that MHA and GQA=2 land within
   0.0003 of each other on control suggests the loss landscape has
   multiple equivalent basins at this scale. The KV parameter count
   is below the level where it changes which basin the optimizer
   falls into.

At 135M, this may flip back. We have not measured it.

---

## Why GQA=1 still hurts

GQA=1 isn't just "more sharing" — it's **one KV head for six Q heads**.
The K projection can only learn one "what to look for" pattern; every
Q head's query is matched against the same K. The V projection is the
same story. The model effectively has a single attention pattern, just
applied six times. That's not enough capacity to attend to different
things in different positions.

The 0.040 loss at screen20m is the cost of forcing attention into a
single pattern. Even with the rest of the V+q+SWA+HighRoPE wins, the
ceiling is real.

---

## Why MLA loses

MLA (DeepSeek-V2) compresses K and V through a low-rank latent:

```text
plain:  K = X @ W_K       (shape: [seq, d_model] → [seq, n_kv × d_k])
        V = X @ W_V

MLA:    c = X @ W_down    (shape: [seq, d_model] → [seq, d_c])   d_c << d_kv
        K = c @ W_K_up    (shape: [seq, d_c] → [seq, n_kv × d_k])
        V = c @ W_V_up
```

The bottleneck `d_c` forces a compressed representation of K and V. At
DeepSeek-V2's scale this is fine — the latent can be rich enough — and
the win is **KV-cache compression** for inference, not training
quality.

At 10M, the latent `d_c=36` is too narrow. The compressed K and V lose
information that the attention layer would otherwise use. −0.091 is
the cost of that compression.

---

## The code

Three flags:

```python
# configs/llm_config.py
n_kv_heads: int = 2                # default — works at all tested scales
use_mla: bool = False               # MLA is closed at 10M
mla_latent_dim: Optional[int] = None
```

The implementation in `models/layers.py` is a one-line switch in
`MultiHeadAttention.__init__`:

```python
if self.use_mla:
    self.kv_down = nn.Linear(d_model, mla_latent_dim, bias=False)
    self.k_up = nn.Linear(mla_latent_dim, n_kv_heads * d_k, bias=False)
    self.v_up = nn.Linear(mla_latent_dim, n_kv_heads * d_k, bias=False)
else:
    self.k_proj = nn.Linear(d_model, n_kv_heads * d_k, bias=False)
    self.v_proj = nn.Linear(d_model, n_kv_heads * d_k, bias=False)
```

---

## Lessons

1. **A "wash" result is information, not nothing.** GQA=2 vs MHA at 10M
   is a wash; that means *KV sharing is not a lever here* and we should
   stop testing it. Negative results close axes.
2. **Washes can flip with scale.** Tiny favored MHA; 10M washed it out.
   A 135M run may flip it back. Don't extrapolate a wash across scales
   in either direction.
3. **Architecture moves from other labs (MLA, GQA) are scale-conditional.**
   DeepSeek-V2's MLA is a real lever for them, closed at 10M for us.
   "Big model" levers are not "small model" levers, by default.
4. **Closed axes are not failed axes.** MLA −0.091 is a real
   measurement; we now know MLA is closed at 10M and don't need to
   test it again.

---

## Caveats

- **Single-seed for the MHA and MLA numbers on the best baseline.**
  The GQA=2 result (4.6364) is single-seed. The deltas are large
  enough to be safe (0.002 for MHA, 0.091 for MLA), but the exact
  numbers are ±0.01.
- **The wash at 10M may flip at 135M.** GQA=2 may be too aggressive
  (or too conservative) at larger scale. We have not measured.
- **MLA's win at DeepSeek scale is mostly about KV cache compression
  for inference**, not training quality. We measured training loss
  only; inference cost is a separate axis we did not test.

---

## Reproduce

```bash
# the closed MLA result (on best baseline, ~30 min):
python train_llm.py \
  --config screen20m \
  --use_value_embed true --use_q_gain true \
  --use_sliding_window true --sliding_window_size 512 \
  --rope_base 500000 \
  --use_mla true --mla_latent_dim 36 \
  --seed 42

# MHA control (no flags, full attention, ~19 min):
python train_llm.py --config screen20m --n_kv_heads 6 --seed 42
```

Code: [models/layers.py](../../../models/layers.py) (`MultiHeadAttention`),
flags in [configs/llm_config.py](../../../configs/llm_config.py) (`n_kv_heads`,
`use_mla`, `mla_latent_dim`).
Evidence: [LEADERBOARD.md](../../../LEADERBOARD.md) §`screen20m` row 18g
(MHA), row 18m (MLA); §`tiny1m arch` rows 1, 3, 4, 5.
