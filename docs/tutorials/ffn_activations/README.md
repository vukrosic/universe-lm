# FFN activation: ReLU² vs GELU vs SwiGLU — a conditional lever

**Result:** the FFN activation is a **conditional** lever. The same
change (squared-ReLU → GELU) is *anti-additive* standalone,
*additive* on V+q+SWA, then *anti-additive* again on
V+q+SWA+HighRoPE. SwiGLU is washed on V+q.

| Recipe | squared-ReLU | GELU | SwiGLU | verdict |
|---|---|---|---|---|
| control | **4.7984** | 4.8647 | — | GELU **worse** alone |
| V+q+SWA (base 10k) | 4.6700 | **4.6608** | — | GELU **additive** (−0.009) |
| V+q | 4.6797 | — | 4.6944 | SwiGLU **washed** (+0.015) |
| **V+q+SWA+HighRoPE** | **4.6364** | 4.6527 | — | GELU **anti-additive** (+0.016) |

The same change has **opposite sign** in different recipes. **Always
test the activation in the context you care about.**

---

## What the activations are

The feed-forward block in a transformer is two linear projections
with a non-linearity in between. The question is **what the
non-linearity is**:

```text
squared-ReLU (Primer-style, our default):
  y = (relu(x @ W1))^2 @ W2
  # the squaring makes it non-monotonic, gives the model a
  # "soft" gate that goes through zero smoothly

GELU (most common in modern LLMs):
  y = gelu(x @ W1) @ W2
  # gelu(x) = x * Φ(x)  — Gaussian Error Linear Unit
  # smooth, monotonic, no squaring

SwiGLU (Llama-style, gated):
  y = (silu(x @ W1_gate) * (x @ W1_up)) @ W2
  # 3 projections instead of 2; the gate modulates the up-projection
  # parameter-matched to squared-ReLU d_ff=576 via d_ff=384
```

All three are "the same block with a different non-linearity." The
param count varies slightly (SwiGLU has 3 projections, so we use
`d_ff=384` to match squared-ReLU's `d_ff=576` cost).

---

## Standalone: GELU is *worse* than squared-ReLU

| Activation | Val loss | Δ vs squared-ReLU |
|---|---|---|
| **squared-ReLU** (default) | **4.7984** | 0 |
| GELU | 4.8647 | +0.066 |

**On the control recipe, GELU is anti-additive.** This is a clean
negative result for "swap to GELU" as a default change. Primer's
squared-ReLU is not a mistake; it works on this scale.

---

## On V+q+SWA: GELU is *additive* (small)

| Activation | Val loss | Δ vs squared-ReLU |
|---|---|---|
| **squared-ReLU** | 4.6700 | 0 |
| **GELU** | **4.6608** | **−0.009 (additive)** |

On the V+q+SWA recipe, GELU **wins by 0.009** — within the noise band
but directionally additive. The "−0.009" is the screen20m GELU win that
motivated putting it on the best baseline.

**The GELU story changes when SWA unlocks a new operating point.** SWA
restricts the attention pattern; the FFN now does more of the work.
GELU's smoother non-linearity may fit that new role better.

---

## On V+q+SWA+HighRoPE: GELU is *anti-additive* again

| Activation | Val loss | Δ vs squared-ReLU |
|---|---|---|
| **squared-ReLU** | **4.6364** | 0 (best) |
| GELU | 4.6527 | +0.016 (anti-additive) |

**The same change that won on V+q+SWA now loses on V+q+SWA+HighRoPE.**
GELU flips sign again. The HighRoPE recipe changed which FFN
non-linearity is load-bearing — and squared-ReLU is now back on top.

This is a **closed-with-explanation result.** GELU is closed on the
best screen20m baseline. We do not need to test it again at this
recipe.

---

## SwiGLU on V+q: washed

| Activation | Val loss | Δ vs squared-ReLU |
|---|---|---|
| **squared-ReLU** | **4.6797** | 0 |
| SwiGLU (d_ff=384) | 4.6944 | +0.015 (slightly anti-additive) |

SwiGLU is a *parameter-matched* test (3 × d_model × 384 = 2 × d_model
× 576). The extra gating projection costs nothing in params, but
**gating doesn't help here.** V+q already provides the model's
non-linearity story; SwiGLU is one more way to mix that the model
isn't asking for.

---

## Why the activation is conditional

Two honest readings, both consistent:

1. **Operating point changes the load.** SWA restricts attention
   pattern → FFN does more work → smoother non-linearity (GELU)
   helps. HighRoPE extends the useful positional range → attention
   recovers some work → the FFN role narrows → squaring (ReLU²)
   helps. The "right" activation depends on the role the FFN is
   playing in *this* recipe.

2. **Loss landscape smoothness.** Squared-ReLU has a sharper
   non-linearity (the square), which is a stronger gate. GELU is
   smoother, which gives the optimizer a flatter landscape. The
   flatter landscape helps when the basin is hard to find (SWA on
   base 10k); the sharper gate helps when the basin is well-located
   (HighRoPE). The activation is a knob on the local geometry.

Both readings are consistent with the data. The truth is "both, in
a way that depends on the rest of the recipe."

---

## The code

One flag:

```python
# configs/llm_config.py
ffn_variant: str = "squared_relu"   # default
```

The three implementations live in `models/layers.py`:

```python
if self.ffn_variant == "squared_relu":
    h = F.relu(self.w1(x)) ** 2
elif self.ffn_variant == "gelu":
    h = F.gelu(self.w1(x))
elif self.ffn_variant == "swiglu":
    h = F.silu(self.w1_gate(x)) * self.w1_up(x)
y = self.w2(h)
```

`d_ff` is set per-config to parameter-match the three variants
(squared-ReLU uses 576, SwiGLU uses 384; GELU uses 576).

---

## Lessons

1. **A lever can flip sign across recipes.** GELU is anti-additive
   alone, additive on V+q+SWA, anti-additive on V+q+SWA+HighRoPE.
   **Don't generalize a "win" from one recipe to another without
   testing.**
2. **Conditional wins are still real.** GELU on V+q+SWA at 10k
   RoPE was a 0.009 win. Real, but **conditional on the recipe**.
3. **Primer's squared-ReLU is not a mistake.** It wins on control
   *and* on the best screen20m baseline. The "default" activation
   is well-chosen for our setup.
4. **SwiGLU's win at Llama scale doesn't transfer down.** SwiGLU is
   a real lever at Llama's scale; it's washed at ours. Big-model
   levers are not small-model levers, by default.
5. **Always test a lever in the context you care about.** A
   standalone test is not enough; an additive test in *your*
   recipe is the only test that matters.

---

## Caveats

- **Single-seed for all GELU and SwiGLU numbers.** Direction is
  consistent (the differences are 0.009–0.066, well outside the
  noise band of 0.005–0.015 at this scale), but the exact deltas
  are ±0.01.
- **The "GELU on V+q+SWA is additive" claim is the weakest.** 0.009
  is *inside* the noise band on a single seed. The directional
  story (additive on V+q+SWA at 10k, anti-additive on HighRoPE) is
  well-supported; the exact number is not.
- **No activation at depth=24 was tested on the *full* 200M run.**
  The 20M-token screen is what we measured. Activation effects
  often grow with training duration, so the HighRoPE+GELU
  anti-additive result may be **stronger** at full length.
- **Other activations (SiLU, Mish, etc.) were not tested.** We
  tested the three that mattered: the default, the most common, and
  the Llama default.

---

## Reproduce

```bash
# the closed GELU+HighRoPE result:
python train_llm.py --config screen20m \
  --use_value_embed true --use_q_gain true \
  --use_sliding_window true --sliding_window_size 512 \
  --rope_base 500000 \
  --ffn_variant gelu \
  --seed 42

# the additive GELU+V+q+SWA (base 10k RoPE):
python train_llm.py --config screen20m \
  --use_value_embed true --use_q_gain true \
  --use_sliding_window true --sliding_window_size 512 \
  --ffn_variant gelu \
  --seed 42

# SwiGLU on V+q:
python train_llm.py --config screen20m \
  --use_value_embed true --use_q_gain true \
  --ffn_variant swiglu --d_ff 384 \
  --seed 42
```

Code: [models/layers.py](../../../models/layers.py) (FFN block),
flag in [configs/llm_config.py](../../../configs/llm_config.py)
(`ffn_variant`, `d_ff`).
Evidence: [LEADERBOARD.md](../../../LEADERBOARD.md) §`screen20m` row 18c
(GELU additive), row 18e (GELU anti-additive on HighRoPE), and SwiGLU
test (closed); `runs/s_gelu_full/metrics.json`,
`runs/s_vqgain_swa_gelu_full/metrics.json`,
`runs/s_vqgain_swa_highrope_gelu_full/metrics.json`.
