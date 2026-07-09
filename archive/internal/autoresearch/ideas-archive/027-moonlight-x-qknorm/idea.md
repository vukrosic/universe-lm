---
id: 027-moonlight-x-qknorm
status: running
round: 1
updated: 2026-06-10T12:28:10Z
transfer-risk: low
---

# 027 — Moonlight Muon RMS × QK-Norm (optimizer scale alignment + per-head logit bounding)

## Source
Internal composition of two in-stack levers:
- `autoresearch/ideas/015-moonlight-muon-rms/evidence.md` — WIN, Δ −0.0138/−0.0185 at tiny1m3m (seed 42)
- `autoresearch/ideas/016-qk-norm/evidence.md` — WIN, Δ −0.0138/−0.0185 at tiny1m3m (seed 42)

## Mechanism
Enable both `use_moonlight_muon` (per-tensor RMS rescale `c·√max(d_in,d_out)` on Muon's Newton–Schulz orthogonalized update — controls how fast Q,K,V,O weight matrices grow during training) and `use_qk_norm` (LayerNorm on Q,K head-dim before the attention dot product — bounds runtime Q·K magnitude at inference). The two operate at entirely separate code paths: Moonlight is in the optimizer step (`optimizers/muon.py`), QK-Norm is in the model forward (`models/layers.py`). No shared state; enabling both is a two-flag change.

## Scale evidence
- Moonlight (015): demonstrated WIN at tiny1m3m; Moonshot AI (Muon is Scalable for LLMs, arXiv:2502.16982, 2025) applied at billion-parameter LLM scale. transfer-risk: low.
- QK-Norm (016): demonstrated WIN at tiny1m3m; adopted at 22B+ scale (Dehghani et al. arXiv:2302.05442, Qwen3, SmolLM3). transfer-risk: low.

## Why it's worth a slot
015 and 016 tied at exactly Δ −0.0138 in the same A/B session, raising the question of whether they both improve training by stabilising the same underlying signal (attention logit magnitude) from different directions — optimizer-side (update scale) vs inference-side (runtime magnitude). Stacking them tests orthogonality: if truly independent, the combined Δ should be additive (~−0.028 vs ctrl); if they are substitutes for the same stability mechanism, the stack yields diminishing returns. Either answer materially informs which two levers to carry into the 10M→135M ladder.
