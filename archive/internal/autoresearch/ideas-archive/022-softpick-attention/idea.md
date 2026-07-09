---
id: 022-softpick-attention
status: running
round: 3
updated: 2026-06-10T12:28:10Z
---

# 022 — Softpick (rectified-softmax attention, sink-free normalization)

## Source
Zuhri, Fuadi, Aji, "Softpick: No Attention Sink, No Massive Activations
with Rectified Softmax" (arXiv:2504.20966), 29 Apr 2025.

## Mechanism
Replace the softmax in scaled-dot-product attention with a rectified
normalization that permits a *zero* total attention mass:

```
softpick(x_i) = relu(exp(x_i) − 1) / (Σ_j |exp(x_j) − 1| + ε)
```

Because the numerator can be zero for all keys, a head is no longer
forced to dump probability on a sink token, eliminating the
massive-activation / attention-sink pathology. **Pin `ε = 1e-6`**
(paper default). The `exp − 1` op is computed in **fp32** then cast
back to model dtype — large positive scores overflow in fp16/bf16.
Drop-in replacement for the softmax call inside attention; no extra
params, no schedule, no init-scale tuning.

### Mask interaction (correctness-critical)
Standard softmax handles masking by setting masked scores to `−∞` →
`exp(−∞) = 0` → zero attention. Naive softpick under the same regime
is **wrong**: `exp(−∞) − 1 = −1`, then `|−1| = 1` *adds to the
denominator* — masked positions silently pollute the normalizer even
though their numerator contribution is zero. The canonical fix: **set
masked scores to 0** so that `exp(0) − 1 = 0` (clean zero in both
numerator and denominator terms), then apply the same
`masked_fill(−1e9)` *before* the `exp − 1` op but **after** subtracting
`1` zero out the masked entry entirely. Equivalently, multiply both
numerator and denominator by a 0/1 mask after the `exp − 1` op. The
code must use one of these two forms and **must** have a test that
masked positions contribute zero to both numerator and denominator.
This interacts with both the causal mask and SWA (window 512); both
use the same line.

### Step-0 is NOT identity (acknowledge as known A/B asymmetry)
At init, `Q, K ≈ 𝒩(0, small)` → scores ≈ 0 → `exp(0) − 1 = 0` →
numerator = 0 → output = `0 / ε = 0`. The attention path returns
**zero**, not pass-through-V. The residual stream survives (`+ attn(x)`
adds 0), but the model starts with effectively no attention. Taste r1
flagged this as "carry, don't block" — it must be **stated in the
spec** as a known A/B asymmetry (the trt's step-0 attention output
distribution is not the same as the ctrl's) and gated by a smoke
test: build the trt model, run one fwd+bwd, assert loss is finite and
grads on Q/K/V projections are non-zero. If grads vanish (because
attention output is exactly zero ⇒ `∂L/∂Q = 0`), the lever is dead on
arrival and the A/B is malformed — caught before burning GPU.

### Swap site
`models/layers.py:1421` — the `torch.softmax(scores, dim=-1)` line
inside the **`use_fire_pe=True` branch** (the manual attention path).
The SDPA fast path and the FoX/manual-alternative branch at
`models/layers.py:1435+` are left untouched. Because the ctrl =
FIRE-equipped baseline (manual path), this single line swap is the
only softmax call that ever fires in the A/B. Add a corresponding
`use_softpick` entry to the `or self.use_fox`-style OR list at
`models/layers.py:1435-1445` so any non-FIRE path that somehow tries
to use softpick falls back to softmax (defensive — the
`use_fire_pe=True` branch is the only one reached for this trt).

## Why it's worth a slot
We expect a val-loss improvement (or at least equal loss with cleaner
activations) because tiny models with few heads are *most* hurt by
sink-induced wasted attention capacity — at 0.94M params every head
matters. This is a softmax *function* swap (per-step, parameter-free),
categorically distinct from the closed `attn-sink` lever (which
*added* a sink token) and from `sigmoid-loss` (an output-layer loss,
not an attention normalizer). Not identity at step 0, but it's a
one-line normalization change with no init-scale tuning, so a clean
A/B. A null tells us the sink is benign at this scale; a win is a free,
parameter-free architectural lever.

## Definition (gate 2)

### Ctrl vs trt
- **Ctrl**: `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`
  (the 009 WIN signature, val 6.3234 in `closed.md:40`). Pinned to
  the FIRE-equipped baseline so the A/B partitions the orthogonal-
  axis question, not the "does FIRE win?" question. This config uses
  the manual attention path because `use_fire_pe=True` forces it.
- **Trt**: same config + `use_softpick=True`. New config class
  `Tiny1M3MSoftpickOnFireConfig` in `configs/llm_config.py`:
  `use_fire_pe = True`, `use_softpick = True`.

### Pass bar (tiny1m3m noise floor)
Run-to-run val-loss variance at this tier is ≈ ±0.01
(`closed.md:33-40` ctrls span 6.3875–6.4050 = 0.018 spread). With a
single seed the pass bar must clear the ctrl-gap and not just sit
inside it:
- **Win**: `trt_val < ctrl_val − 0.005`. The −0.005 bar (not −0.02) is
  chosen because the bet is "small but real" — softpick is a
  parameter-free normalization tweak, not a structural change, so a
  modest gain is the realistic win shape and we don't want a real
  effect lost in the noise floor.
- **Null**: `|trt_val − ctrl_val| < 0.01` (sub-noise; the lever does
  not fire on top of FIRE at this scale).
- **Fail**: `trt_val > ctrl_val + 0.01` (worse than baseline by more
  than half the ctrl-gap — sink-removal is hurting).

### Seed
**Seed 42 only.** Single fixed seed, no multi-seed sweep, no per-seed
mean. A sub-noise delta is *inconclusive, not real*; never add "run
more seeds to confirm" — log null and move on.

### Step-0 smoke check
Build the trt model (`Tiny1M3MSoftpickOnFireConfig`), run one forward
+ backward pass on a tiny batch, assert (a) loss is finite, (b) grads
on `q_proj.weight`, `k_proj.weight`, `v_proj.weight` are non-zero, (c)
attn_w sums per row to ≤ 1 (sum ≤ 1, not == 1, because softpick permits
zero total mass). If (b) fails the lever is dead on arrival (zero
attn output ⇒ zero grad on Q/K/V) and the A/B is malformed — the
runner must NOT proceed to a full training run.

### Mask interaction test
Build the trt model with a constructed input that places a real key
inside the SWA window and several masked keys outside it. Assert
that the softpick output (a) is zero on masked positions, (b) sums
correctly on unmasked positions, (c) the denominator is *not* polluted
by masked positions (this is the bug class the spec calls out).

### LoC budget (≤ 50 LoC, well under the 200 ceiling)
- (a) `softpick` helper function: rectified `exp − 1` in fp32,
  `relu(·)` numerator, `|·|` denominator, `+ ε` guard, mask multiply
  applied to both numerator and denominator: ≈ 12 LoC
- (b) swap site in `models/layers.py:1421` — replace `torch.softmax`
  with `softpick` inside the `use_fire_pe` branch: ≈ 3 LoC
- (c) flag wiring — `use_softpick: bool = False` on `LLMConfig`
  (`configs/llm_config.py`, sits next to `use_fox: bool = False` at
  line 179), passed through `MultiHeadAttention.__init__`, stored on
  `self`, threaded through `TransformerBlock`: ≈ 10 LoC
- (d) OR-list entry at `models/layers.py:1435-1445` (defensive
  fallback) + new config class
  `Tiny1M3MSoftpickOnFireConfig` in `configs/llm_config.py`: ≈ 6 LoC
- (e) step-0 smoke test (loss finite, non-zero Q/K/V grads, attn_w
  row-sum ≤ 1): ≈ 8 LoC
- (f) mask-handling assertion test (zero on masked positions,
  denominator not polluted, unmasked sum correct): ≈ 6 LoC

Total ≈ 45 LoC.
