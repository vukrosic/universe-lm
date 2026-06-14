---
id: 161-dyt-temp
status: running
round: 2
updated: 2026-06-14T05:29:21Z
transfer-risk: med
plain: Let each layer learn its own attention "sharpness" so different depths of the model can have different attention focus levels — start every layer at sharpness 1 (no change) so step-0 matches the baseline exactly.
---

# 161 — Per-Layer Learnable Attention Temperature

## Source
Standard transformer attention uses a fixed `1/sqrt(d_head)` scale. Per-layer variants appear in:
- Press et al. "ALiBi" (2022, closed) — fixed per-layer linear bias.
- Su Jianlin "RoPE-Xtended" — per-layer scaling discussion on kexue.fm (2024).
- Recent (2024-2025) per-layer attention-scaling literature.

Distinct from 155 (per-head) and from closed logit softcap (per-layer clamp, scalar).

## Mechanism
Replace the fixed `1/sqrt(d_head)` attention scale with a learnable per-layer scalar `τ_l`: `logits = QK^T * τ_l`. With `τ_l = 1/sqrt(d_head)` at init, the scale is identical to baseline. As training proceeds, each layer can adjust its own temperature — early layers might prefer broad attention, late layers might prefer sharp focus. ~10 LoC.

## Design sketch
- **File**: `models/layers.py` — add `self.layer_temperature = nn.Parameter(torch.full((n_layers,), 1.0 / math.sqrt(d_head)))` to the model; index it per layer in the forward.
- **Config flag**: `use_per_layer_temp: bool` (default False).
- **Step-0 identity**: `layer_temperature` is initialized to exactly `1/sqrt(d_head)`, so `QK^T * τ_l = QK^T / sqrt(d_head)` byte-for-byte at step 0.
- **Intuition**: gives the model a free parameter for "how peaky" each layer's attention should be. Different from 155 per-head temperature (which varies within a layer); this varies *across* layers. A null would tell us per-layer temperature is dominated by the existing normalization structure at 0.94M; a win would tell us layers want different attention temperatures (early broad, late sharp).
- **Important distinction**: per-head (155) = within-layer head variability; per-layer (161) = across-layer variability. Two orthogonal axes.

## Scale evidence
Standard mechanism; per-layer variants appear in RoPE-scaling ablations and recent layerwise-attention studies. Transfer risk is **med** (the mechanism is simple but per-layer scaling has less direct production validation than per-head).

## Why it's worth a slot
A win would tell us *layerwise* attention-temperature specialization is the binding axis at 0.94M (orthogonal to per-head temperature from 155); a null would close the per-layer temperature axis and confirm per-block normalization absorbs the variance.
