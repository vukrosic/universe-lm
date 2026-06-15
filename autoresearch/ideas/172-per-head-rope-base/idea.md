---
id: 172-per-head-rope-base
status: running
round: 1
updated: 2026-06-15T05:07:19Z
transfer-risk: med
plain: Give each attention head its own learnable RoPE base frequency (all starting at the global 500k base so the first step is byte-identical to the baseline), and let the model learn per-head frequency scales via backprop.
---

# 172 — Per-Head Learnable RoPE Base Frequency (Multiplicative Head Scale on Rotary Frequencies)

## Source
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
  (Neurocomputing 2024, arXiv:2104.09864) — the canonical RoPE paper. Uses a
  single global `base` hyperparameter (default 10000) shared across all heads
  and all layers. Different bases encode different frequency ranges:
  `inv_freq[i] = 1 / (base ** (2i / d_k))`. Smaller base → broader
  frequencies (better for short context); larger base → finer frequencies
  (better for long context).
- "RoPE base sweep" (closed axes line, `closed.md:22`): "500k winner" — we
  already swept fixed global bases and found 500k beats 10000 at tiny1m3m.
  172 takes the next step: instead of picking one fixed base, *learn* a
  per-head multiplicative scale on top of the base.
- Per-head frequency scaling is conceptually related to NTK-aware scaling
  (bloc97, 2023) and YaRN (Peng et al., arXiv:2309.00071) which adapt the
  base for *context-length extension*; 172 adapts the effective base for
  *head specialization* — different heads may want different frequency
  regimes.
- Code is already in place: `models/layers.py:1753-1754` declares
  `self.use_per_head_rope_base: bool` and
  `self.per_head_rope_log: nn.Parameter`. `models/layers.py:2031-2032` uses
  it: `head_scale = torch.exp(self.per_head_rope_log)` then
  `freqs *= head_scale[None, :, None, None]`. The mechanism is *built and
  wired but never tested* — there is no `172-per-head-rope-base` idea file
  and no entry in `closed.md` referencing it.

## Mechanism
Baseline RoPE (per head `h`): position `t` and dimension `i` give frequency
`f_{t,i} = t / base^{2i/d_k}`. With per-head learning:
```
head_scale[h] = exp(per_head_rope_log[h])    # one scalar per head, init 0
f_{t,i,h} = t · head_scale[h] / base^{2i/d_k}
```
The Q/K rotation for head `h` becomes `head_scale[h]` times the global-base
rotation. Init: `per_head_rope_log = 0` ⇒ `head_scale = exp(0) = 1.0` for
all heads ⇒ all heads use the global base ⇒ **byte-identical to the
baseline RoPE at step 0 (max-abs-diff = 0.0 across the entire forward)**.

The optimizer can then push each `head_scale[h]` away from 1.0: a value
of 1.2 means head `h` operates at a slightly higher-frequency band
(broader context), 0.8 means a lower-frequency band (more local). The
total parameter overhead is `n_heads = 4` scalars per block × `n_layers =
12` blocks = 48 scalars total (+0.005% of 0.94M params).

## Design sketch
- **Files**: the lever is already implemented in `models/layers.py` at
  lines 1753-1754 (init), 2031-2034 (use), 2828 and 2851 (forward-graph
  integration). The implementation work is purely the **config wiring**:
  - `configs/llm_config.py` — add `use_per_head_rope_base: bool = False`
    on `LLMConfig` (default off) and a `Tiny1M3MPerHeadRopeConfig`
    subclass with `use_per_head_rope_base: bool = True`. Set
    `rope_base: int = 500000` (the closed-axes winner) explicitly.
  - Verify that `use_per_head_rope_base` is read from config and threaded
    into the four `TransformerBlock(...)` / `MultiHeadAttention(...)`
    construction sites in `models/llm.py` (the sites that already thread
    `use_per_head_rope_base` per the existing `models/layers.py:3833`).
- **Config flag**: `use_per_head_rope_base: bool = False` (default off).
- **Step-0 byte-identical**: with `per_head_rope_log = 0` (the default
  init for a new `nn.Parameter`), `head_scale = exp(0) = 1.0` for all
  heads. The frequency spectrum `freqs * 1.0 = freqs` is unchanged. The
  cos/sin tables are unchanged. The Q/K rotation is unchanged. Forward
  output is unchanged ⇒ **byte-identical to the rope_base=500k baseline
  at step 0 (max-abs-diff = 0.0, no tolerance needed)**.
- **Intuition (why it might lower val loss)**: different attention heads
  specialize on different positional scales (e.g. one head attends to
  local context, another to mid-range, another to long-range). With a
  single global base, *all heads are forced into the same frequency
  regime* — they can still specialize by which subset of frequency
  components they use, but the per-head scaling freedom is locked.
  Per-head learnable base lets the model find a better specialization
  by adjusting the per-head frequency band. The 500k base was the
  *best fixed compromise* across heads; per-head learning may find
  that head 0 wants 700k, head 1 wants 350k, etc.
- **LoC**: ~10 lines of config wiring (mechanism is already in
  `models/layers.py`). Smallest implementation cost of the three
  filed levers.

## Scale evidence
- The original RoPE base sweep (closed axes: "500k winner") was at
  tiny1m3m — the lever's *target tier*.
- Per-head learnable RoPE base is principled (every modern LM uses
  per-head Q/K/V projections; giving per-head learnable position
  encoding is a natural extension) but not heavily validated at
  ≥100M in the literature. The closest analog is per-head RoPE
  partial-rotary (`partial_rotary_p`, already in the code), which
  has been validated at 1B+.
- **Transfer risk: med** (principled extension, not directly
  validated at ≥100M; the 500k-base-sweep evidence is at our target
  tier which is encouraging).

## Why it's worth a slot
The bet: per-head learnable RoPE base is a small, clean lever with
step-0 byte-identical behavior, and the 500k global base is a
compromise that per-head learning could improve. We expect Δval ≈
-0.003 to -0.012 (small wins, since the lever only adds 4 scalars
per block). A null would tell us the global 500k base is sufficient
at 0.94M and per-head frequency specialization doesn't bind. A win
would tell us per-head RoPE is a real axis and the lever is worth
re-evaluating at Phase-2 ≥135M where each head has more gradient
signal to develop a useful scale. Smallest implementation cost of
the three filed levers (~10 LoC of config wiring only).
