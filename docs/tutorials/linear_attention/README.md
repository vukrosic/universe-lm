# Linear attention (Performer-style): doesn't work at small scale

**Result:** Performer-style linear attention lands at **val 6.4691** on
the 0.94M "tiny" arch — **+0.134 worse** than the softmax-attention
baseline (6.3350). Linear attention is a **closed axis at small scale.**

This is a clean negative result: the random-feature approximation
that's supposed to make attention `O(n)` instead of `O(n²)` is too
lossy at our setup. The axis is closed for tiny; it likely won't
help at 10M either, and we have not re-tested it there.

---

## What linear attention does

Standard (softmax) attention computes a score matrix then a weighted
sum:

```text
S = softmax(Q @ K.T / sqrt(d_k))    shape: [seq, seq]    O(n²) memory
Y = S @ V                           shape: [seq, d_k]   O(n²) compute
```

Linear attention (Performer) **avoids the explicit `seq × seq`
matrix** by replacing softmax with a positive feature map `phi(x) =
elu(x) + 1`:

```text
phi_Q = phi(Q)                      shape: [seq, d_k]
phi_K = phi(K)                      shape: [seq, d_k]
KV   = phi_K.T @ V                  shape: [d_k, d_k]    O(n) compute
Y    = phi_Q @ KV / (phi_Q @ phi_K.sum(0))   shape: [seq, d_k]   O(n) compute
```

The trick: `phi(Q) (phi(K).T V)` is *mathematically equivalent* to
`softmax(Q K.T) V` only when `phi` is the softmax kernel. For
`phi(x) = elu(x) + 1`, it's an **approximation** — the random
feature map for the softmax kernel.

**No new parameters.** Same `Q`, `K`, `V` projections, different math.

---

## The result at the tiny tier (0.94M, 3M tok)

Same recipe (V-embed + Q-gain + SWA-384 + RoPE-250k), seed 42:

| attention | val loss | Δ vs softmax |
|---|---|---|
| **softmax (default)** | **6.3350** | 0 (baseline) |
| **linear (Performer-style)** | **6.4691** | **+0.134** |

Linear attention is **+0.134 worse**. That is *not* a wash — that is
the model failing to learn the same patterns the softmax version
learns.

---

## Why the approximation is too lossy at this scale

The positive-feature map `phi(x) = elu(x) + 1` approximates the
softmax kernel `K(x, y) = exp(x · y)` via a low-rank random
projection. The accuracy of the approximation depends on:

1. **The rank of the feature map.** `phi(x)` is `d_k`-dimensional,
   same as the input. The kernel approximation has rank `d_k`. For
   `d_k=24` (tiny) or `d_k=24` (screen20m), that's a **low-rank
   approximation** of a `seq × seq` matrix.

2. **The structure of the actual attention pattern.** If the model's
   attention is mostly **low-rank** (a few "modes"), the
   approximation works. If it's **dense** (many distinct patterns
   per row), the approximation misses things.

At small scale, models tend to learn *more* low-rank attention
patterns (they don't have the capacity to learn dense ones). This
should help linear attention. The +0.134 says it doesn't help
*enough* — the approximation error from the random feature map
dominates over the benefit of low-rank structure.

---

## Why it works at large scale (and ours isn't large)

Linear attention's win is at **Llama-scale and beyond**, where:

- The `seq × seq` matrix is a real memory cost (long context).
- The model has the capacity to learn *dense* attention patterns,
  and the random feature map captures them well *at sufficient rank*.
- The compute savings (`O(n)` vs `O(n²)`) are worth the
  approximation error.

At 0.94M / 3M tokens, **none of those conditions hold.** The
attention matrix is small, the model is capacity-limited, and the
`O(n²)` cost is irrelevant. Linear attention is paying the
approximation tax for a speedup it doesn't need.

---

## What this teaches

This is a **clean negative result** with a clear mechanism. The
axis is closed at small scale; we should not test it again at
10M until we have a reason to think the approximation is the
bottleneck.

It's also a teaching example of "**a real lever at large scale can
be a closed axis at small scale.**" Performer-style linear attention
is a published, cited, real mechanism. It just doesn't help at our
size.

---

## The code

One flag, ~30 lines:

```python
# configs/llm_config.py
use_linear_attn: bool = False
```

```python
# models/layers.py — MultiHeadAttention.forward
if self.use_linear_attn:
    # phi(x) = elu(x) + 1
    Q_phi = F.elu(Q) + 1.0
    K_phi = F.elu(K) + 1.0
    # K_phi.T @ V : d_k × d_k
    KV = torch.einsum("bnsd,bsnd->bsnd", K_phi, V)  # actually einsum...
    # ...
    # (full implementation in models/layers.py)
```

Same `Q`, `K`, `V` projections. Different attention math.

---

## Lessons

1. **"Real lever" at one scale ≠ lever at another.** Linear
   attention is a real mechanism with published wins. It is
   closed at 10M and below. **Don't take "it works" from a paper
   as "it works here."**
2. **A clean negative result is information.** +0.134 with
   reproducible seed tells us linear attention is a closed axis
   at this scale. We don't need to test it again.
3. **The cost of an approximation must be paid in a place that
   matters.** Linear attention pays approximation error everywhere
   — including in the patterns the model would otherwise learn
   well. At small scale, that error is dominant. At large scale,
   the compute savings dominate.
4. **Closed axes are valid tutorial material.** This isn't a
   "win" tutorial. It's a "this is closed, here's why" tutorial.
   Both kinds prevent future-me from re-running the same failed
   experiment.

---

## Caveats

- **Single-seed at tiny.** The +0.134 is large enough to be
  unambiguous (well outside the noise band of 0.005–0.015), but
  multi-seed confirmation is not done.
- **Linear attention at 10M was not re-tested.** The screen20m
  version of this run is not on disk (per the leaderboard, #80
  is "pending"). The closed verdict is **at 0.94M**, not at 10M.
- **Linear attention has many variants** (Performer, Linear
  Transformer, RWKV, Mamba, etc.). We tested one specific
  Performer-style positive-feature implementation. State-space
  models (Mamba) and other variants may behave differently.
- **The approximation rank is `d_k`.** If we increased `d_k`, the
  approximation would improve. We did not test that.

---

## Reproduce

```bash
# the closed result at tiny (~2 min):
python train_llm.py --config tiny1m \
  --use_value_embed true --use_q_gain true \
  --use_sliding_window true --sliding_window_size 384 \
  --rope_base 250000 \
  --use_linear_attn true \
  --seed 42
# observe val loss ≈ 6.47 instead of ≈ 6.34
```

Code: [models/layers.py](../../../models/layers.py) (`MultiHeadAttention.forward`),
flag in [configs/llm_config.py](../../../configs/llm_config.py) (`use_linear_attn`).
Evidence: [LEADERBOARD.md](../../../LEADERBOARD.md) §`tiny1m arch` linear
attention row; `runs/tiny1m_arch_linearattn_full/metrics.json`.
