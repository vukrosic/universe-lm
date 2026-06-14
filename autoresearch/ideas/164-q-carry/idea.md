---
id: 164-q-carry
status: needs-taste
round: 1
updated: 2026-06-14T05:20:00Z
transfer-risk: med
plain: Let each attention block "borrow" the queries computed by the previous block, with a learnable mix amount starting at zero — a residual-style information pathway that costs almost nothing.
---

# 164 — Cross-Block Q-Carry (Q Residual from Previous Block)

## Source
- 021-value-residual (WIN, tiny1m3m, Δ -0.034) — added V from block l-1 to V in block l, gated by a learnable scalar (init 0). The win says *cross-block V carry* is binding at 0.94M.
- 150-xlayer-feedback (reject, r3 cap) — attempted full cross-block attention. Failed at this tier; the cross-block mutation broke step-0 byte-identity.
- 116-hyper-connections (null) — residual-stream splitting. Different mechanism.
- mHC papers (Zhu et al. 2024) discuss cross-block residual mixing; QK-residual is a narrow subset.
- "Mamba-2 / RWKV-7" dual-stream mixing — QK is decoupled in some recent SSM hybrids.

Distinct from 021 (which carried *V*; 164 carries *Q*) and from 150 (which attempted full cross-block *attention*, not residual Q mix). 164 is the *lightest* cross-block lever: a single per-block scalar α_l gates Q carry.

## Mechanism
For each block l (l ≥ 1), augment the Q projection with a learnable cross-block carry from block l-1:
```
Q_l = W_Q x_l + α_l · (W_Q x_{l-1})
```
where `α_l` is a per-block learnable scalar (init to 0). At α=0, the carry is a no-op and `Q_l` reduces to the baseline projection. K and V are unchanged. This gives the model a *free* information pathway from the previous block's residual stream into the current block's query — a Q-side analog of value-residual (021). ~10 LoC; 12 extra scalar params (one per block, +0.001% of 0.94M).

## Design sketch
- **File**: `models/layers.py` — `TransformerBlock.__init__` adds `use_q_carry: bool = False` kwarg. The flag is consulted by the parent `MinimalLLM` to wire `prev_x = residual_after_block_{l-1}` (the *input* to block l, i.e. what block l sees *after* the residual sum). In `MultiHeadAttention.forward` (called from `TransformerBlock.forward`), when `use_q_carry=True`, take `q = self.W_Q(x_l) + self.alpha_l * self.W_Q(prev_x)`. The `prev_x` is passed in as an extra argument to the attention forward.
- **Config flag**: `use_q_carry: bool = False` (default off on `LLMConfig`).
- **Step-0 identity**: `alpha_l` is a `nn.Parameter(torch.zeros(1))` per block, so `0 · W_Q(prev_x) = 0` at step 0. The W_Q projection of `x_l` is byte-identical to baseline (same input, same weights, same RNG state).
- **Intuition**: 021 says V carry helps. The dual axis (Q carry) tests whether the previous block's information should also enter the *query* — i.e. whether the *content* of the previous block's residual stream is itself useful as a "what to ask for" signal at the current block. A Q-carry win would tell us 021's gain comes from the residual *information pathway* (V is one direction, Q is the dual); a null would tell us V is special and Q has its own information source.
- **Why now**: 021 was a strong WIN. The dual axis (Q) is unclosed. If V carry and Q carry both work, the mechanism is "residual stream cross-block mixing" (and we should consider 116-style mHC more carefully at deeper tiers); if only V works, the mechanism is V-specific (and 164 is null).

## Scale evidence
021 value-residual is the only published validation at this exact form, but the underlying mechanism (residual cross-block mixing) is well-validated: ResNets (He et al. 2015), Highway Networks (Srivastava et al. 2015), and 116-mHC (Zhu et al. 2024) all use residual-style cross-layer mixing at 100M+ scale. The narrow Q-specific subset is less directly validated — hence **med** transfer risk (the lever is structural, narrower than general mHC, and the Q-vs-V asymmetry is not a published production result).

## Why it's worth a slot
A win would tell us the *residual information pathway* (not just V specifically) is the binding cross-block axis at 0.94M, opening the door to richer cross-block mixing (generalized 116-mHC at deeper tiers). A null would tell us 021's WIN was V-specific, and Q has its own (independent) information source that doesn't benefit from cross-block carry.
