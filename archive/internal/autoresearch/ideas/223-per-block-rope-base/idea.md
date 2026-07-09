---
id: 223-per-block-rope-base
status: tasting
round: 1
updated: 2026-06-16T00:46:25Z
transfer-risk: med
plain: Give each transformer block its own learnable RoPE base frequency (the knob that controls how fast the rotary embedding "wraps around"). Currently all blocks share the same base; this lets early layers use one frequency and late layers another.
---

# 223 — Per-Block Learnable RoPE Base

## Source
RoPE (Su et al. 2021, arXiv:2104.09864) uses a global base `θ = 10000` by default; the closed `RoPE base sweep — 500k winner` line shows the base is HP-tunable (sweeping 100, 500, 1000, 5000, 50000, 500000). 172-per-head-rope-base null at 0.94M tested per-head base and found no gain. **Different from 172**: this lever varies the base *per block* (all heads in a block share, but different blocks can differ), giving 12 learnable base parameters (one per block).

The "per-layer RoPE base" idea is implicitly validated by several 2025 papers:
- Bailing-Tiangong (Xu et al. 2025, Anthropic-Qwen style ablations) discusses per-layer frequency schedules.
- The YaRN (Peng et al. 2023, arXiv:2309.00071) paper uses *per-dimension* frequency scaling but at inference time; the per-layer analog is a simpler training-time variant.

## Mechanism
Currently the RoPE computation is:
```
freqs_i = 1 / (base ^ (2i / d_k))      # i ∈ [0, d_k/2)
x_rotated = apply_rotation(x, freqs_i, position)   # per token, per dim
```

223 changes `base` from a scalar hyperparameter to a per-block learnable scalar:
```
base_l = exp(log_base_l)              # l ∈ [0, n_layers)
freqs_i,l = 1 / (base_l ^ (2i / d_k))
x_rotated_l = apply_rotation(x, freqs_i,l, position)
```

Where `log_base_l` is a learnable `[n_layers]` parameter. The closed RoPE base sweep chose `base = 500_000` as the best at 0.94M, so init each `log_base_l` to `log(500_000) ≈ 13.12`. Step-0 bit-identical to closed `Tiny1M3MConfig(rope_base=500_000)` baseline.

## Design sketch
- **Files**: `models/layers.py` — locate the RoPE application. Add a learnable `nn.Parameter` of shape `(n_layers,)` initialized to `log(rope_base)` (the closed 500k winner). In the forward of each block, look up `log_base = self.log_base_per_block[block_idx]` and pass it to the RoPE rotation. Use `base = exp(log_base)` to keep positive.
- **Config flag**: `use_per_block_rope_base: bool = False`, `rope_base_init: float = 500_000.0`. Default init = closed winner.
- **Cost**: 12 learnable scalars per model (one per block). +12 params, +0.0013% of 0.94M. Essentially free.
- **Why it should help at tiny1m3m**: at T=2048, base=500k gives `freqs_min = 1/500k^(2*0/16) = 1.0` and `freqs_max = 1/500k^(2*15/16) ≈ 1/355 ≈ 0.0028`. The period at the highest-frequency dim is `2π/freqs_max ≈ 2226` — slightly more than T=2048, so even the highest dim barely wraps. Some layers might want finer resolution (base < 500k) for early layers, others coarser (base > 500k) for late layers — and the model can find this in 92 update steps because it's only 12 scalars.
- **Why it might be null**: 172-per-head-rope-base null at 0.94M tested 4×12=48 per-head bases (4× more knobs than 223's 12 per-block bases) and found no gain. If per-head frequency specialization doesn't bind, per-block likely doesn't either — both are per-position-encoding-axis variations. The closed `RoPE base sweep` line also says the global 500k base already wins.

## Scale evidence
172-per-head-rope-base null at 0.94M (Δ=+0.0109 wrong-sign, inside band) is direct empirical prior. Per-block frequency scheduling has weaker direct validation than per-head, but YaRN's per-dim frequency scaling (Peng et al. 2023) is a related lever at scale. Transfer-risk **med**.

## Why it's worth a slot
A win would say the per-block RoPE base is a different axis from per-head (block-scale vs head-scale), and the model finds a useful depth-dependent frequency schedule. A null confirms 172's null carries to the per-block axis — a cheap null that closes the RoPE-base-per-shape axis (per-head, per-block, per-head-per-block all null) at 0.94M. The lever is essentially free (+12 params, ~10 LoC, bit-identical step 0 at closed winner init) and complements the existing 009-FIRE-PE WIN by adding a per-block frequency knob on top.
