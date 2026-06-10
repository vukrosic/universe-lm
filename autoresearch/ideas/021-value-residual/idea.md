---
id: 021-value-residual
status: needs-codereview
round: 1
updated: 2026-06-10T02:00:31Z
---

# 021 — Value Residual Learning (cross-layer V shortcut)

## Source
Zhou et al., "Value Residual Learning for Alleviating Attention Concentration in Transformers" (arXiv:2410.17897), Oct 2024. Also adopted as `value embed`-adjacent trick in modded-nanogpt speedruns, but the cross-layer formulation here is the paper's ResFormer/SVA.

## Mechanism
Compute the value projection at the first layer once (`V_1`), then in every later layer mix it into that layer's own value: `V_l = (1-λ_l)·V_l + λ_l·V_1`, with a per-layer learnable scalar `λ_l` init 0 (identity at step 0). Gives later layers a direct shortcut to the first-layer value representation, reducing attention concentration. ~25 LoC: stash layer-0 V in the forward pass, add one learnable scalar per block, blend before attention output.

**V_1 shape and blend site (precise).** V_1 is the **projected V at layer 0**, stashed *post-`W_V@x`* in the same shape MHA uses at the blend point — `[B, n_heads, T, d_k]` (post-reshape at `models/layers.py:1257`, post-GQA `repeat_interleave` on `dim=2` at 1293-1294, post-`transpose(1,2)` at 1380). The actual order of operations in `models/layers.py` is: V is reshaped to `[B, T, n_kv_heads, d_k]` at 1257, GQA expansion on `dim=2` runs at 1293-1294 (giving `[B, T, n_heads, d_k]`), and only then does the `transpose(1, 2)` at 1380 produce `[B, n_heads, T, d_k]`. The blend runs **right after `models/layers.py:1380`**, before the optional `v_norm` at 1383 and well before the manual-attention branch at 1402. In every later layer `l > 0`, MHA.forward receives V_1 and the blend is:

```
# Right after models/layers.py:1380 (post-transpose, post-GQA, pre-v_norm, pre-attention):
if self.use_value_residual and v_residual is not None:
    V = (1.0 - self.lambda_v) * V + self.lambda_v * v_residual
```

The blend runs **before** `attn_weights @ V`, but **after** the projection / GQA expansion / per-head transpose. This is the paper's canonical site (ResFormer §3.1, Eq. 4): the projected V is the shortcut, not the post-attention output. Because the blend is post-GQA, the shape `[B, n_heads, T, d_k]` is identical across all layers regardless of GQA settings — no broadcast worries at the blend site.

**V_1 plumbing.** V_1 is a **forward-pass-local** value, not a `nn.Parameter`, not a persistent buffer. The MHA at layer 0 stashes it on `self` after computing the post-transpose V (`self._v_residual = V.detach()`), and the model reads it via `block.attention._v_residual` after the block-0 call. The model then passes it as a positional arg `v_residual` to blocks 1..N-1; each block forwards it to its MHA. The stash uses `.detach()` so gradients don't flow back into the layer-0 projection from the layer-l blend — each layer's W_V is trained on its own attention path. The shape `[B, n_heads, T, d_k]` is consistent across all layers because the stash and the blend both happen at the post-transpose site.

## Why it's worth a slot
We expect a small but real val-loss drop because a V-shortcut counters the attention-concentration collapse that hurts tiny models with few heads (tiny1m3m has narrow heads where a single sink token dominates). Distinct from the closed V/Q/K/O *embedding* axis — this is a cross-layer residual on the value *stream*, not an added token embedding. It fires every step (no EMA, no schedule dependence), so it cannot fail the AdEMAMix-style "too slow for a 92-step run" trap. A null still teaches us whether cross-layer value mixing transfers below 1M params.

## Definition (gate 2)

### Ctrl vs trt
- **Ctrl**: `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True` (`configs/llm_config.py:773` + the `use_fire_pe` flag = the 009 WIN config from `closed.md:44`; 009 trt = 6.3234 vs ctrls 6.3875/6.4050). The FIRE-equipped baseline is the correct control: 021 stacks on FIRE to test the orthogonality of cross-layer V mixing to additive positional bias (the same question 020 tests for multiplicative decay). **Do not** test against the un-FIRE'd baseline — that re-litigates the 009 question, not the 021 question.
- **Trt**: same config + `use_value_residual=True`. New config class `Tiny1M3MVResidualOnFireConfig(Tiny1M3MConfig)` with `use_fire_pe: bool = True, use_value_residual: bool = True` (mirroring the `Tiny1M3MFOXOnFireConfig` shape at `configs/llm_config.py:714`).

### Pass bar (tiny1m3m noise floor)
Run-to-run val-loss variance at this tier is ≈ ±0.01 (`closed.md:41-44` ctrl spread 6.3875–6.4050 = 0.0175). The FIRE-ctrl will re-bracket but assume the same ~±0.01 box noise. With a single seed the pass bar must clear the ctrl-gap and not just sit inside it:
- **Win**: `trt_val < ctrl_val − 0.005` (low-to-moderate bar because the bet is at the small end of the paper's reported effect; the taste r1 reviewer asked for exactly this band).
- **Null**: `|trt_val − ctrl_val| < 0.01` (sub-noise; the lever does not fire on top of FIRE at this scale).
- **Fail**: `trt_val > ctrl_val + 0.01` (worse than baseline by more than the ctrl-gap; the cross-layer mix is hurting attention concentration rather than helping it).

### Seed
**Seed 42 only.** Single fixed seed, no multi-seed sweep, no per-seed mean. A sub-noise delta is *inconclusive, not real*; never add "run more seeds to confirm" — log null and move on.

### LoC budget (≤ 50 LoC, well under the 200 ceiling)
- (a) per-block scalar `self.lambda_v = nn.Parameter(torch.zeros(()))` **on each `TransformerBlock`**, init 0 (0-dim scalar per block; no `layer_idx` plumbing needed — the model reads the vector via `[block.lambda_v.item() for block in model.transformer_blocks]`): ~3 LoC
- (b) stash V_1 in layer-0 MHA.forward after the post-transpose V is computed (`self._v_residual = V.detach()`); model reads `block.attention._v_residual` after the block-0 call, then passes `v_residual=V_1` as a positional arg to blocks 1..N-1; each block forwards it to its MHA: ~8 LoC
- (c) blend in `MHA.forward` of layers `l > 0`: `V = (1-self.lambda_v)·V + self.lambda_v·V_1` **right after `models/layers.py:1380`** (post-transpose, post-GQA, pre-v_norm, pre-attention): ~3 LoC
- (d) flag wiring (`use_value_residual: bool = False` in MHA + `TransformerBlock` + `LLMConfig`, plus new config class `Tiny1M3MVResidualOnFireConfig`, plus `lambda_v` as a sub-attribute initialized in `TransformerBlock.__init__`): ~12 LoC
- (e) one test asserting `use_value_residual=False` ≡ baseline at step 0 *and* `use_value_residual=True, lambda_v=0` ≡ baseline at step 0 within `1e-5` (identity-init at the blend point because λ=0 ⇒ V_l = V_l): ~10 LoC

Total ≈ 36 LoC. Well under the 50 cap and the 200 ceiling. The FIRE-equipped baseline already forces the manual-attention branch (line 1402 in `models/layers.py`), so 021 needs no additional manual-branch forcing — the blend runs in the manual path the FIRE baseline already takes.

### Evidence to capture
- Per-block λ_l values at the **end of training** (one scalar per block, one `lambda_v` per `TransformerBlock`) — collect via `[block.lambda_v.item() for block in model.transformer_blocks]` and append to `evidence.md`. A uniform `λ_l → 0` post-training is a **stronger null** than "inside variance": it means the model rejected the shortcut at every block, not just at the population level. Conversely, a non-monotonic λ profile (e.g. λ_0=0, λ_3≈0.1, λ_5=0) is a finding — it says deeper blocks want the shortcut more than shallow ones, or vice versa.
- `lambda_v.grad` snapshots at step 0 and step ~half (optional but cheap) — confirms the gradient is flowing through the blend and not dead.
- The A/B val-loss and step-time — the standard A/B output.
