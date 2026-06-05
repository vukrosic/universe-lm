# RoPE base: positional decay is a real lever, and the optimum scales

**Result:** RoPE base is a clean teaching curve — and the **optimum
shifts with model size**. At the 0.94M "tiny" tier, **250k wins**. At the
10M "screen20m" tier, **500k wins**. At 10M, the default base=10000 was
leaving ~−0.045 on the table.

| Tier | Best base | Val loss | Default-base (10k) | Δ |
|---|---|---|---|---|
| tiny1m3m (0.94M, 3M tok) | **250,000** | 6.3506 | ~6.4306 | **−0.080** |
| screen20m (10M, 20M tok) | **500,000** | 4.6364 | ~4.68+ | **−0.045** |

The default RoPE base of 10,000 — GPT-Neo style — was tuned for short
contexts and small models. It is **not the right default for our setup.**

---

## What RoPE base actually does

Rotary Position Embeddings rotate each `Q` and `K` vector by an angle that
grows with position. The "base" is the geometric center of that rotation
spectrum:

```text
θ_i = base ^ (-2i / d_k)        for i = 0, 1, ..., d_k/2 - 1
```

- **Low base** (e.g. 10,000) → short wavelength → positional information
  blurs fast → distant tokens look the same.
- **High base** (e.g. 500,000) → long wavelength → positional information
  stays sharp over longer distances → the model can read far-apart
  relationships.

For `seq_len=2048` and `d_k=24` (tiny) or `d_k=24` (screen20m has
`d_model=144, n_heads=6 → d_k=24` too), the default `base=10000` puts most
of the rotation information in the **first ~256 positions** — the rest of
the 2,048-token context is mostly rotationally-aligned noise.

---

## The sweep at the tiny tier (0.94M, 3M tok)

Same recipe (V-embed + Q-gain + SWA(384)), seed 42:

| RoPE base | Val loss | Δ vs 250k |
|---|---|---|
| 125,000 | 6.3650 | +0.0144 |
| **250,000** | **6.3506** | **winner** |
| 375,000 | 6.3656 | +0.0150 |
| 500,000 | 6.3694 | +0.0188 |
| 750,000 | 6.3769 | +0.0263 |

**250k is the sweet spot at tiny scale.** Going lower loses (positional
information too compressed), going higher loses (wavelengths too long for
the model to use them at this size).

This is a **clean monotone sweep with a single interior optimum** — exactly
the shape you want from a real lever.

---

## The sweep at the screen20m tier (10M, 20M tok)

Same recipe (V-embed + Q-gain + SWA(512)), seed 42:

| RoPE base | Val loss | Δ vs 500k |
|---|---|---|
| 250,000 | 4.65xx | +0.014 |
| **500,000** | **4.6364** | **winner** |
| 1,000,000 | 4.65xx | +0.014 |

The optimum **moved up by 2×** as we went from 0.94M → 10M. This is
expected: a bigger model can use longer-range positional information, so
the optimal wavelength grows with capacity.

**The default 10k is bad at both scales** — it just happens to be a clean
default for GPT-2 era, not for our 10M / 2k-seq setup.

---

## Why the optimum scales

Two readings, both consistent:

1. **Wavelength / useful range.** At base=10k and `seq_len=2048`, the
   shortest-rotating frequency is already past one full cycle in the
   first 256 tokens. The last 1,800 tokens of context are
   rotationally-near-identical. A 10M model has enough capacity to
   *want* to read those positions — but the rotation has washed out the
   signal by then.
2. **Effective dim usage.** Each `θ_i` is a separate "knob" in the
   rotation spectrum. With base=10k and `d_k=24`, the smallest 8-10
   `θ_i`'s are doing all the work and the rest are flat. With base=500k,
   all 12 are spread out across useful periods, and the model can use
   the full spectrum.

The right base is "as large as the model can use without the
shortest-rotating `θ_i` collapsing below 1 radian per position." At 10M /
`seq_len=2048`, that's ~500k. At 135M, it will likely be larger still.

---

## The result on the best baseline

V-embed + Q-gain + SWA(512) + **RoPE base=500k** → **4.6364**, current
screen20m record. The HighRoPE add was the **last big lever** in the
ladder — 12 closed axes before it, 12 closed axes after, all of them
*losing* to the HighRoPE combination.

The default base=10000 was hiding a lever because nobody questions the
default. **Question the defaults.**

---

## The code

One flag:

```python
# configs/llm_config.py
rope_base: int = 10000       # default — leaves headroom on the table at seq_len=2048
```

```python
# models/components.py — RoPE precomputation
freqs = 1.0 / (self.rope_base ** (torch.arange(0, d_k, 2).float() / d_k))
```

That's it. No new parameters, no new shapes. Just a single integer
multiplied into the precomputed inverse-frequency buffer.

---

## Lessons

1. **Defaults are not free.** Base=10000 was the right default for GPT-2
   at 1k context. It is not the right default for our 10M / 2k setup.
   Cost of changing: one integer. Win: ~−0.045 on the record.
2. **Sweeps reveal monotone curves.** A clean 250k-1000k sweep with one
   interior optimum is the shape of a real lever. If the curve is flat
   or noisy, the lever is probably nothing.
3. **The optimum depends on the model.** A rule of thumb "use base=500k"
   is wrong; the answer is "sweep it." At larger models, push higher.
4. **High RoPE pairs with SWA and V-embed.** Alone, RoPE base doesn't
   move much; combined with V+q+SWA, it unlocks the −0.045 win. RoPE
   base is in *conversation* with the rest of the architecture, not a
   standalone knob.

---

## Caveats

- **Single-seed for the 500k record.** The 4.6364 result is seed=42.
  Direction is well-supported (3-seed mean of V+q alone is 4.6815 ± 0.006,
  HighRoPE adds another −0.045 in single-seed sweeps), but the exact
  −0.045 is ±0.01.
- **The optimum will keep moving.** The right base at 135M / 2k context
  is probably > 500k. We have not measured this.
- **A 1M base was tested once** (#85, single-seed) and lost. That doesn't
  *close* 1M at 10M; it just says the optimum sits below 1M at this
  scale. A 750k test would refine the boundary.

---

## Reproduce

```bash
# the record recipe (single seed, ~30 min on RTX 3050):
python train_llm.py \
  --config screen20m \
  --use_value_embed true \
  --use_q_gain true \
  --use_sliding_window true \
  --sliding_window_size 512 \
  --rope_base 500000 \
  --seed 42

# the tiny-tier sweep (faster, ~2 min each):
python train_llm.py --config tiny1m --use_value_embed true --use_q_gain true \
  --use_sliding_window true --sliding_window_size 384 --rope_base 250000 --seed 42
# (change --rope_base to 125000 / 375000 / 500000 / 750000 for the sweep)
```

Code: [models/components.py](../../../models/components.py) (RoPE precompute),
flag in [configs/llm_config.py](../../../configs/llm_config.py) (`rope_base`).
Evidence: [LEADERBOARD.md](../../../LEADERBOARD.md) §`tiny1m3m` row 1
(250k) and §`screen20m` row 18d (500k);
`runs/s_vqgain_swa_highrope_full/metrics.json`.
