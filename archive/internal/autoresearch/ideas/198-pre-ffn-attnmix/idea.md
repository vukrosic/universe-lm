---
id: 198-pre-ffn-attnmix
status: done
round: 1
updated: 2026-06-16T00:39:02Z
transfer-risk: med
plain: Mix the attention output back into the FFN's input as a learned residual (init at 0 so step-0 is byte-identical), letting the FFN see what attention computed without disturbing the residual stream itself.
---

# 198 — Pre-FFN Attention Mixing (Learnable Cross-Stream Mixer Before FFN)

## Source
- 164-q-carry (closed null Δ=+0.0360 wrong-sign) — carries Q from previous block to current block in attention path. Different placement (cross-block, attention-side).
- 168-av-output-carry (closed null Δ=−0.0227 inside band) — carries attention output (AV) across blocks on residual stream. Different placement (cross-block).
- 021-value-residual (in-repo WIN Δ=−0.034) — V-side cross-block carry. Different axis.
- 186-v-carry-block (needs-run) — within-block V carry. Different axis (within-block).
- FiLM conditioning (Perez et al., 2018, arXiv:1709.07871) — feature-wise linear modulation of one stream by another.
- Shleifer et al., "NormFormer" (2021, arXiv:2110.09423) — extra LN at attention output, conditioning the residual stream. 198 is FFN-input-side, not residual-side.

## Mechanism
Standard FFN: `out = ffn(attn_residual)` where `attn_residual = x + attn_block(x)`. The FFN sees `attn_residual` as input.

Pre-FFN attention mixing: also feed the FFN the *raw attention output* (without the residual):
```
ffn_input = attn_residual + γ · attn_block(x).detach()    # γ learnable, init 0
out = ffn(ffn_input)
```
At γ=0, `ffn_input = attn_residual` exactly (bit-identical baseline). As γ grows, the FFN sees a growing fraction of the raw attention output as a conditioning signal.

This is **not** the same as 164-q-carry (Q-side cross-block carry) or 168-av-output-carry (post-attention cross-block carry on residual). 198 is *within-block*, pre-FFN, mixing the attention output *into the FFN input* (not into the residual).

## Design sketch
- **File**: `models/layers.py` — modify `ffn_block` to optionally receive a "raw attn output" tensor and add a learnable scalar γ to mix it in.
- **Config flag**: `use_pre_ffn_attn_mix: bool = False`, `pre_ffn_attn_mix_init: float = -10.0` (γ via `sigmoid(γ_raw)` init at `γ_raw = -10` so sigmoid ≈ 0).
- **Compute**: `ffn_input = attn_residual + sigmoid(γ_raw) · raw_attn_output.detach()`. The `.detach()` prevents gradient from flowing back through the raw attention output (cleaner attribution).
- **Bit-identical at step 0**: γ_raw = -10 ⇒ sigmoid ≈ 0 ⇒ `ffn_input ≈ attn_residual` (forward graph unchanged up to fp32 noise).
- **Params**: 1 scalar per block × 12 blocks = 12 γ scalars (+0.001% of 0.94M).
- **Intuition**: the FFN currently sees the residual stream (which includes the attention output added back). 198 lets the FFN *directly* see the attention output (separately from the residual) — a parallel side-channel that the optimizer can use to condition FFN behavior on the attention pattern. Different from 164/168 which are cross-block carries.

## Scale evidence
FiLM conditioning validated at visual reasoning (Perez et al. 2018). No published "pre-FFN attention mixing" win for LMs. Transfer-risk: med (lever is well-defined but novel).

## Why it's worth a slot
**Pattern**: cross-block carries (164 Q, 168 AV) closed null at 0.94M (the cross-block axis doesn't bind). 198 is a different axis: *within-block* attention-to-FFN mixing. The bet: at 0.94M/12L, the binding constraint is not cross-block information flow but *intra-block* information flow between attention and FFN. The raw attention output (without residual) carries signal the residual has not yet integrated; if the FFN can use this signal directly, it might do better than waiting for the residual to propagate. A 198 WIN would mean intra-block attention-FFN information flow is a binding lever; a 198 NULL would confirm the FFN's residual-stream input is sufficient.
