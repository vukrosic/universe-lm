---
id: 168-av-output-carry
status: needs-run
round: 1
updated: 2026-06-14T09:34:05Z
transfer-risk: med
plain: Have each attention block borrow the *post-attention output* of the previous block, mixed in by a learnable scalar starting at zero — a residual pathway at the attention output rather than the residual stream.
---

# 168 — Cross-Block Attention-Output Carry (Post-AV Residual)

## Source
- 021-value-residual (WIN, tiny1m3m, Δ -0.034) — V-side cross-block carry: `V_l = W_V x_l + α_l · W_V x_{l-1}`. Carries the *value* vector from block l-1 to block l, mixed in by a learnable scalar (init 0).
- 164-q-carry (in pipeline, round 2) — Q-side cross-block carry: `Q_l = W_Q x_l + α_l · W_Q x_{l-1}`. The dual axis to 021.
- 116-hyper-connections (null) — residual-stream splitting with a learnable mix; structurally different (operates on the residual stream, not the attention tensor).
- 150-xlayer-feedback (rejected, r3 cap) — full cross-block attention (Q/K/V from prev block); 3 rounds of divergence at d_model=64/12L.
- 168 is the *post-attention* axis: carry the *attention output* (post-AV product, pre-W_O) from block l-1 to block l.

Distinct from 021 (pre-AV V carry), 164 (pre-AV Q carry), 116 (residual stream mix), 150 (full cross-block attention).

## Mechanism
For each block l (l ≥ 1), augment the attention output with a learnable carry from block l-1's attention output:
```
av_l = softmax(Q_l K_l^T / sqrt(d)) V_l     # current block's AV product
out_l = W_O @ (av_l + α_l · av_{l-1})         # mix in previous block's AV
```
where `α_l` is a per-block learnable scalar (init to 0). At α=0, the carry is a no-op and `out_l` reduces to the baseline W_O @ av_l. The carry operates on the *attention output* (post-AV, pre-W_O), not on V (which is what 021 carries). This is the third axis in the cross-block carry family: V-side (021) / Q-side (164) / AV-output-side (168). ~10 LoC; 12 extra scalar params (one per block, +0.001% of 0.94M).

## Design sketch
- **File**: `models/layers.py` — `TransformerBlock.__init__` adds `use_av_output_carry: bool = False` kwarg. The flag is consulted by the parent `MinimalLLM` to wire `prev_av = av_{l-1}` (the *post-AV, pre-W_O* tensor from block l-1). In `MultiHeadAttention.forward`, when `use_av_output_carry=True`, after computing `av_l = SDPA(Q_l, K_l, V_l)`, apply `av_l = av_l + self.alpha_l * prev_av` *before* the W_O projection. The `prev_av` is passed in as an extra argument to the attention forward.
- **Config flag**: `use_av_output_carry: bool = False` (default off on `LLMConfig`).
- **Step-0 identity**: `alpha_l` is a `nn.Parameter(torch.zeros(1))` per block, so `0 · prev_av = 0` at step 0. The W_O projection of `av_l` is byte-identical to baseline (same input, same weights, same RNG state). Note: `prev_av` must be passed in via the call chain but does not affect the forward until α is nonzero.
- **Intuition**: 021 says V carry helps. 164 tests Q carry. 168 tests the *post-AV* axis — i.e. the previous block's "what to attend to" summary as a residual signal to the current block's attention output. This is structurally different from V carry (which operates on the *input* to AV) and from residual-stream mixing (which mixes after the *full* block). A 168-WIN would tell us the binding axis is the *attention output*, not V specifically. A 168-null would tell us 021's WIN was V-specific, and the post-AV axis is not the binding cross-block lever.
- **Why now**: 021 is the strongest cross-block win in the closed set. 164 (Q carry) is in the pipeline. 168 is the third axis — the only one that hasn't been tested. Once 164 and 168 both run, we have a complete 3-way cross-block carry test (V / Q / AV-output).

## Scale evidence
021 value-residual is the only published validation at this exact form, but the underlying mechanism (residual cross-block mixing) is well-validated: ResNets (He et al. 2015), Highway Networks (Srivastava et al. 2015), and 116-mHC (Zhu et al. 2024) all use residual-style cross-layer mixing at 100M+ scale. The narrow AV-output-specific subset is less directly validated — hence **med** transfer risk (the lever is structural but the AV-output-vs-V asymmetry is not a published production result).

## Why it's worth a slot
A win would tell us the *post-attention* axis (not V specifically, not Q specifically) is the binding cross-block carry, opening the door to richer post-attention cross-block mixing at deeper tiers. A null would tell us 021's WIN was V-specific, and the AV-output axis is not the binding one. Together with 164's result, this completes a 3-way attribution test for cross-block carry.
