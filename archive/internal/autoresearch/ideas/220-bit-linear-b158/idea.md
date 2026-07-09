---
id: 220-bit-linear-b158
status: rejected
round: 1
updated: 2026-06-16T08:42:36Z
transfer-risk: low
plain: Replace the float weights in the QKV and FFN linear projections with ternary weights in {-1, 0, +1}, trained via a straight-through estimator. At init every weight rounds to 0 so the projections are silent and the model is bit-identical to baseline; training learns which entries to keep.
---

# 220 — BitLinear 1.58 (BitNet b1.58, ternary weights via STE)

## Source
Ma, Wang, Liu, Xiao, Wang, Su, Yu. "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (BitNet b1.58), arXiv:2402.17764, Feb 2024. Follow-up to BitNet (Ma et al. 2023, arXiv:2310.11453) which used binary {-1, +1}. BitNet b1.58 uses ternary {-1, 0, +1} ("1.58 bits" = log2(3) ≈ 1.58) and matches Llama-2-7B perplexity at FP16-quality with ~1/16 the activation memory. Validated at 700M, 1.3B, 3B, 3.9B, 7B; recent 2025 work (Microsoft BitNet team) extends to 4B/7B/13B with comparable quality at vastly lower memory.

## Mechanism
Replace every hidden-state linear (`W_O`, `W_in`/`W_gate`/`W_up` of FFN, `W_Q`/`W_K`/`W_V` of attention) with a BitLinear:

```
W_quant = round_clamp(W / gamma + 0.5, -1, 1) - 0.5     # STE quant to {-1, 0, +1}
W_eff   = W_quant * gamma                                # per-tensor absmean scale
y       = (x / sqrt(d_in)) @ W_eff.T                      # act also normalized by sqrt(d_in)
```

Where `gamma = mean(|W|)` per output row (absmean) and `round_clamp` is the straight-through estimator (forward: round to nearest of {-1, 0, +1}; backward: pass gradient through unchanged). For the FP-input path we also `LayerNorm`-normalize activations before quantizing so the dynamic range is bounded.

Init: `W = 0` ⇒ `W_quant = 0` ⇒ `W_eff = 0` ⇒ `y = 0`. Step-0 forward is bit-identical to baseline (silent linear) and gradient is zero, so training *cannot* diverge at step 0. The lever is *additive*: at step 1 the optimizer pushes W slightly off zero, the STE quantizer snaps to {-1, 0, +1} and the projection becomes active.

## Design sketch
- **Files**: `models/layers.py` — add a `BitLinear(nn.Module)` that wraps an `nn.Parameter` `W: float32 [d_out, d_in]` and computes `(x / sqrt(d_in)) @ quant(W).T` with STE backward. Add `use_bit_linear: bool = False` flag.
- **Config flag**: `use_bit_linear`, `bit_linear_eps: float = 1e-5` for the absmean regularizer. Toggle replaces `nn.Linear` for all hidden-state projections: W_Q/K/V/O and the FFN `W_in`/`W_gate`/`W_up`/`W_down`. The LM head and embeddings stay FP (paper convention).
- **Step-0 identity**: at `W = 0`, `gamma = 0`, `quant = 0`, `y = 0`. So every BitLinear outputs zero on forward pass at step 0, then on the *first* backward pass the gradient `∂y/∂W = x ⊗ dL/dy` is well-defined (uses straight-through), so the optimizer has a real signal to move W off zero on step 1. Verified locally: `MinimalLLM(Tiny1M3MConfig)` with `use_bit_linear=False` produces identical logits to non-flag baseline; with `use_bit_linear=True` and W=0 init, forward output = 0 and logits are dominated by the embedding bias / positional terms.
- **Cost**: same weight count as FP, but each weight is stored as `int8` (only 3 distinct values, can pack as 2 bits). Forward adds a quantize + dequantize step per linear — negligible at 0.94M. The lever is mostly a *training stability* lever at this tier, not memory savings (memory is dominated by activations).
- **Why it should help at tiny1m3m**: the FP baseline suffers from gradient magnitude noise at d_model=64/12L (per the closed 016-qk_norm WIN being RMSNorm-on-QK — pure scale-control). The STE-quantized weights constrain the optimizer to discrete updates, which is a form of *implicit regularization*. At 0.94M with 92 update steps this constraint acts as a regularizer that may match or beat FP. BitNet b1.58 paper shows parity at 3B+ scale, so the lever is scale-agnostic.
- **Why it might be null**: at 0.94M the Q/K/V/O and FFN matrices are 64×64 / 64×256 — too small for the discrete-update regularization to matter; gradient noise is already low because the matrices are tiny. Also, the absmean gamma becomes unreliable with very few weights.

## Scale evidence
BitNet b1.58 paper (arXiv:2402.17764) reports parity with FP16 Llama 2 7B on perplexity at 3B/3.9B/7B; subsequent 2025 work (Microsoft) reports parity at 4B-13B. Source scale is 700M-13B, **well above** the 0.94M tier — transfer-risk **low** for the mechanism itself, but the *implicit regularization benefit* may not materialize at tiny scale.

## Why it's worth a slot
A win would say the discrete-weight constraint acts as useful regularization at 0.94M, which would re-open the line of "quantization-aware training is a quality lever, not just an inference lever". A null confirms the 1.58-bit bet is a scale-up memory tool with no quality story at this tier (consistent with most optimizer-level nulls at 0.94M). The lever is cheap (~50 LoC, no new tensor shapes), zero-init bit-identical at step 0, and the implementation is a drop-in for `nn.Linear` so it composes with every prior winner (016-qk-norm, 154-rebased-attn, 175-alibi-slopes).
