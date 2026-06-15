---
id: 197-output-residual-sqrt-2l
status: needs-taste
round: 1
updated: 2026-06-15T08:30:00Z
transfer-risk: low
plain: Initialize every block's residual contribution with a 1/sqrt(2L) scale (DeepNet's α rule) so the residual stream doesn't explode as depth grows — depth-aware but init-time only.
---

# 197 — DeepNet α Residual Init (Fixed 1/sqrt(2L) Scale on Sublayer Output, Init-Time Only)

## Source
- Wang et al., "DeepNet: Scaling Transformers to 1,000 Layers" (2022, arXiv:2203.00555) — the paper introduces a *fixed* (not learned) depth-conditional scale `α = (8N)^(-1/4) ≈ 1/sqrt(2L)` for residual scaling, where N is the total number of layers. DeepNet validates this at 200-1000 layers on machine translation and language modeling. The lever is the *fixed* `α` form (not learned, not per-layer-scalar — a single global scalar that's a function of depth).
- "Primer" (So et al. 2021) — also uses depth-conditional scaling, but in a *learned* form (multiplier per block).
- "ReZero" (Bachlechner et al. 2020) — uses a *learned* scalar per block, init at 0 (silent at step 0). Different mechanism (per-block learned, not global fixed).
- "LayerScale" (Touvron et al. 2021, arXiv:2103.17239) — per-channel *learned* gain, init at `1e-4`. Different mechanism (per-channel, not global).
- In-repo context: 017-sub-ln-sandwich (null) — closes the *depth-conditional LN-sandwich* axis at 0.94M (L=6). 197 is the *fixed* depth-conditional *scalar* form, not a per-block LN. Different mechanism.
- 130-rezero (null) — per-block *learned* scalar (init 0). 197 is a *fixed* global scalar (no learned params). Different.
- 142-layerscale (null) — per-channel *learned* gain (init 1e-4). 197 is a global *fixed* scalar. Different.

## Mechanism
Standard pre-norm transformer:
```
def block(x):
    attn_out = attention(LN(x))      # [B, T, d_model]
    x = x + attn_out                 # residual addition
    ffn_out = ffn(LN(x))             # [B, T, d_model]
    x = x + ffn_out                  # residual addition
    return x
```
The residual stream grows by `O(sqrt(L))` over L blocks (each block adds a contribution of magnitude O(1) in expectation). For L=12 (tiny1m3m), the residual grows by `sqrt(12) ≈ 3.5×` from input to output.

With DeepNet α init:
```
def block(x):
    attn_out = attention(LN(x))
    x = x + alpha * attn_out        # alpha = (2L)^(-1/2) for L layers total
    ffn_out = ffn(LN(x))
    x = x + alpha * ffn_out
    return x
```
For tiny1m3m (L=12), `alpha = (2*12)^(-1/2) = (24)^(-1/2) = 1/sqrt(24) ≈ 0.204`. Each block adds a contribution of magnitude `0.204` per component, so after 12 blocks the residual grows by `sqrt(12) * 0.204 ≈ 0.71` per component — i.e., the residual's magnitude is *bounded* by `O(1)` throughout the network.

**Step-0 byte-identity**: with `alpha = 0.204` (a fixed scalar), the residual contribution at step 0 is `alpha * attn_out = 0.204 * O(1) = 0.204` per component. The baseline has contribution `1.0 * attn_out = O(1)`. The two are **not** bit-identical — the lever scales the residual contribution by 0.204 at every block.

**The lever is "step-0 ≠ baseline" by construction.** It's a *fixed* init-time scale on every block's residual contribution. The model trains with this smaller residual scale, and the optimizer must adapt.

For **step-0 byte-identity**, the lever would need to be a *learned* per-block scalar (init at `1` for baseline, init at `0.204` for DeepNet, but the optimizer could grow it toward 1 if needed). The fixed-form lever (this one) is *not* step-0 byte-identical; it accepts a different forward at step 0 in exchange for a theoretically motivated init.

**Alternative form for step-0 byte-identity**: use a *learned* per-block scalar `α_l` (init at 0.204, the DeepNet value) and let the optimizer grow it toward 1 if it wants the un-scaled residual. This adds 12 params and gives step-0 ≈ DeepNet, with the optimizer able to re-discover the un-scaled form. Default to the fixed form for simplicity; document the alternative.

## Design sketch
- **Files**:
  - `models/llm.py` (or `models/layers.py`) — in the transformer block, add `use_deepnet_alpha: bool = False` config flag. When `True`, multiply the attention output and FFN output by `alpha = (2 * config.n_layers) ** -0.5` before adding to the residual.
  - `configs/llm_config.py` — add `use_deepnet_alpha: bool = False` to `LLMConfig`. Add `Tiny1M3MDeepNetAlphaConfig(Tiny1M3MConfig)` with `use_deepnet_alpha: bool = True`.
- **Config flag**: `use_deepnet_alpha: bool = False`.
- **Param count**: **0 new params** (fixed scalar).
- **Intuition (why it might lower val loss)**: the residual stream's magnitude grows by `O(sqrt(L))` over L blocks. For tiny1m3m (L=12), the residual grows by `sqrt(12) ≈ 3.5×`. The DeepNet α init *bounds* the growth to `O(1)` throughout the network, which means the LM head sees a well-conditioned input regardless of depth. The closed Sub-LN-sandwich (017) and ReZero (130) suggest that *depth-conditional* levers don't bind at 12L, but the *fixed-init* form (DeepNet α) is a different choice: it's a *theoretically motivated* init that has been validated at 200-1000L, and at 12L it's a *coarse* init that may or may not help.
- **Why it might bind at 0.94M where 017/130/142 didn't**: those levers are *per-block* (learned per-block scalars or norms), which gives the optimizer a *lot* of freedom to specialize the depth-conditional behavior. DeepNet α is a *single global scalar* (no per-block freedom), which is a stronger *theoretical* prior (the magnitude of the residual is bounded by `O(1)`) but a *weaker* optimizer lever (no per-block adaptation). The closed per-block levers null because the optimizer has too much freedom and the per-block specialization is noise at 12L; the DeepNet form has *no* freedom (it's a fixed init), so the noise is removed.

## Scale evidence
- DeepNet (Wang et al. 2022) — 200-1000 layers on machine translation and language modeling. Direct validation of the lever form.
- Primer (So et al. 2021) — 100M-1.5B. The Primer paper uses a *learned* depth-conditional scaling; DeepNet uses a *fixed* form. Both are validated.
- **Transfer-risk: low** — the lever has direct validation at 200L+ (DeepNet) and 100M+ (Primer), and the fixed-form is a *theoretical* init with strong prior support.

## Why it's worth a slot
The bet, in one sharp sentence: **DeepNet's `α = 1/sqrt(2L)` is a theoretically motivated *fixed* depth-conditional init that has been validated at 200-1000L, and the closed per-block depth-conditional levers (017 Sub-LN, 130 ReZero, 142 LayerScale) test *learned* forms, not the *fixed* form** — a fixed global scalar removes the optimizer's freedom to over-fit per-block noise, which is a *theoretically cleaner* prior at small L; a null at 0.94M would close the *fixed-depth-conditional-init* axis at this tier, and a win would give a 0-param init lever with strong theoretical backing.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- 017-sub-ln-sandwich (null) — *per-block* LN-sandwich (learned). 197 is *fixed global scalar*. Different axis.
- 130-rezero (null) — *per-block* learned scalar (init 0). 197 is *fixed global scalar*. Different axis.
- 142-layerscale (null) — *per-channel* learned gain (init 1e-4). 197 is *global fixed scalar*. Different axis.
- 111-drop-path (null) — drop-path regularizer on the residual. 197 is *fixed init*, not a regularizer.
- 116-hyper-connections (null) — multi-stream residual. 197 is *single-stream scaled residual*.
