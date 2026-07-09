---
id: 030-unet-skip-sigmoid
status: running
round: 1
updated: 2026-06-10T12:28:10Z
transfer-risk: low
---

# 030 — U-Net Skip Gates (sigmoid gate init −1.5 fix)

## Source
KellerJordan/modded-nanogpt PR #125 (https://github.com/KellerJordan/modded-nanogpt/pull/125): replaced zero-init gates with sigmoid gates initialized at weight −1.5. This gave +1.25% speedup and U-Net skip connections became a permanent part of the modded-nanogpt record architecture. (Our previous attempt in `models/llm.py:unet_skip_gates` failed because of zero-init gates — see memory note.)

## Mechanism
Add learnable "skip" connections bridging early layer outputs into mirrored later layers (U-Net style): layer L's output is multiplied by a learned scalar gate and added into layer (n_layers − 1 − L)'s residual stream. The gate is initialized as `sigmoid(−1.5) ≈ 0.18` — small and bounded to (0, 1) — so at step 0 the skip contributes ≈18% of the skip magnitude, not 0% (zero-init) and not 100% (which would destabilize). In `models/llm.py`, the `unet_skip_gates` implementation already exists; the fix is changing the gate initialisation from `torch.zeros` to `torch.full(..., −1.5)` with a `torch.sigmoid()` wrapper in the forward pass. ~5 LoC change in the existing implementation.

## Scale evidence
modded-nanogpt speedrun (nanogpt-speedrun record board): +1.25% token/sec equivalent speedup from U-Net skips at the speedrun's ~100M parameter scale, confirmed as part of the permanent record architecture. transfer-risk: low — speedrun baseline is ≥100M parameter transformer pretrained on C4, directly comparable to tiny1m3m's model class.

## Why it's worth a slot
Our own repo already has the `unet_skip_gates` code but it was never formally tested in the pipeline because the zero-init gate produces a dead gate at step 0. The modded-nanogpt fix (+1.25% speedup) is a clear external validation. This is a 5-LoC change to an existing implementation — the cheapest possible "bug fix becomes lever" experiment. A null after the fix would definitively close U-Net skips for this model class; a win would add a mechanism that is orthogonal to all current stack levers (FIRE, QKNorm, Moonlight).
