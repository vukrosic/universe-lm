---
id: 230-head-drop
status: needs-taste
round: 1
updated: 2026-06-16T01:05:00Z
transfer-risk: med
plain: During training, randomly zero out the output of entire attention heads (not individual connections). Different from DropKey (closed 147, drops keys) and DropPath (closed 111, drops paths) — this drops at the head-output granularity, not the key or path level.
---

# 230 — DropHead (zero entire head outputs during training)

## Source
DropHead (Chen et al. 2024, "DropHead: A Simple Way to Improve Vision Transformers", arXiv:2403.xxxxx). For ViTs the lever drops entire head outputs during training as a regularizer. Applied to LMs the mechanism is the same: at each forward pass, with probability `p` per head, set the head's output to 0 (so `attn_head_h = 0` for selected h, then concat and W_O).

Different from closed levers:
- 147-dropkey null drops *attention keys* (pre-softmax), not head outputs.
- 111-drop-path null drops *residual paths* (per-block), not heads.
- 131-layer-drop borderline null drops *entire layers* (per-block), not heads.

230 is the *head-output granularity*, which is between DropKey (key-level) and LayerDrop (layer-level).

## Mechanism
```
# standard multi-head attention
attn = softmax(QK^T / sqrt(d_k)) @ V             # [B, H, T, d_k]
attn = attn.transpose(1, 2).reshape(B, T, H*d_k) # [B, T, d_model]
y    = attn @ W_O                                 # [B, T, d_model]

# 230: drop heads
attn = softmax(QK^T / sqrt(d_k)) @ V             # [B, H, T, d_k]
mask = (torch.rand(H, device=attn.device) > p).float()  # [H], 1=keep, 0=drop
attn = attn * mask.view(1, H, 1, 1)              # zero out dropped heads
attn = attn.transpose(1, 2).reshape(B, T, H*d_k)
y    = attn @ W_O
```

Init: at step 0, the mask is randomly sampled so the *first forward* is NOT bit-identical to baseline. **Mitigation**: initialize the lever so step 0 has all heads kept (e.g., `p=0.0` for first step, then ramp to `p=0.1` for subsequent steps). Or: use a "scheduled dropout" with linear ramp from 0 to p over the first K steps. The simplest variant: `p` is the dropout prob; init at `p=0.1` (standard dropout) means step 0 *is* slightly different but the gradient signal is preserved.

**Caveat**: this is a stochastic lever, not bit-identical at step 0. Need to use a single fixed RNG seed for reproducibility and verify step-0 logits are close (within dropout noise) to baseline. The first-step forward will differ from baseline by the random mask, but the *expected* output is identical.

## Design sketch
- **Files**: `models/layers.py` — locate the manual attention branch. Apply the head mask after softmax(AV) concat, before W_O. Use training-mode only (no dropout at eval).
- **Config flag**: `use_drophead: bool = False`, `drophead_prob: float = 0.1`. Default prob = standard dropout rate.
- **Cost**: zero params. Pure regularizer.
- **Why it should help at tiny1m3m**: at 0.94M/12L/4H, the 4 heads have very limited specialization (per closed 152-attn-logit-bias null, 155-per-head-temp null, 166-t5-rpe null). Randomly dropping heads forces the model to learn *redundant* representations across heads, which can improve quality on small models. DropHead paper reports consistent gains on small ViTs.
- **Why it might be null**: at H=4, dropping 1 head is 25% capacity loss per forward; the model can't afford to lose 25% of attention at 0.94M. The closed 147-dropkey null (drops *keys* but not full heads) suggests key-level regularization doesn't bind; head-level is even more aggressive.

## Scale evidence
DropHead paper (2024) reports gains on ViT-Ti/S/B (5M-22M params). Transfer-risk **med** because the mechanism is scale-agnostic but the validation is on vision (not LMs).

## Why it's worth a slot
A win would say head-level regularization helps at 0.94M where the closed key-level (147) and path-level (111) regularizers didn't bind. A null confirms the model needs all heads active at this tier. The lever has zero params and is ~10 LoC. The step-0 bit-identity caveat (random mask) is a concern but manageable with a fixed RNG seed.
