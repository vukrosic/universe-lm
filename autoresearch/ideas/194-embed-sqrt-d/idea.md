---
id: 194-embed-sqrt-d
status: needs-run
round: 1
updated: 2026-06-15T08:51:04Z
transfer-risk: low
plain: Scale the input embedding down by 1/sqrt(d_model) (GPT-2 style) so its magnitude matches the attention and FFN outputs and the residual stream doesn't explode.
---

# 194 — Embedding 1/sqrt(d_model) Scaling (GPT-2-Style Residual Init)

## Source
- Radford et al., "Language Models are Unsupervised Multitask Learners" (OpenAI 2019, GPT-2) — scales the residual stream by `1/sqrt(2 * n_layers)` to keep the residual's magnitude bounded as depth grows. This is the *post-norm* version. The pre-norm version is the standard modern convention (LLaMA, Mistral, etc.).
- "T5" (Raffel et al. 2020, arXiv:1910.10683) — uses a different scaling: the embedding is scaled by `sqrt(d_model)` to ensure the *output* (LM head) has a consistent magnitude.
- "Primer" (So et al. 2021, arXiv:2109.08668) — the paper introduces a "scaled initialization" that scales the embedding and the LM head by `sqrt(d_model)` and `1/sqrt(d_model)` respectively, to keep the residual stream's magnitude stable. Validated at 100M-1.5B.
- "ReZero" (Bachlechner et al. 2020, arXiv:2003.04887) — uses an *identity residual* with a learnable scalar `α = 0` init, so the residual contribution is exactly 0 at step 0. Different mechanism (per-block scalar, not init).
- In-repo context: 159-emb-layernorm (null, DRIFT) — closed the *embedding-side LN* axis (rescaling the input distribution). 194 is *not* an LN; it's a *scalar multiplication* on the embedding output, which preserves the *direction* of each token's embedding and only rescales the magnitude. Different mechanism.
- 130-rezero, 142-layerscale (null) — depth-conditional residual-stream levers. 194 is *init-time only* (no per-step overhead), and applies to the *embedding* (not the per-block residual).
- 183-pre-lm-head-rmsnorm (null) — output-side norm. 194 is *input-side scalar*, not output-side norm.

## Mechanism
Standard embedding (no scaling):
```
x = Embedding(token_ids)            # [B, T, d_model], each row has magnitude ~1 (Kaiming init)
x = x + pos_embedding               # [B, T, d_model]
out = transformer(x)
logits = LM_head(out)
```
With embedding 1/sqrt(d_model) scaling (Primer-style):
```
x = Embedding(token_ids)            # [B, T, d_model], each row has magnitude ~1
x = x * (1 / sqrt(d_model))          # scale down by 1/sqrt(d_model) = 1/8 for d_model=64
x = x + pos_embedding                # [B, T, d_model]
out = transformer(x)
logits = LM_head(out) * sqrt(d_model)  # scale up by sqrt(d_model) in the LM head to compensate
```
The pre-norm transformer (LLaMA-style) has each block compute `x = x + sublayer(LN(x))`. The sublayer output has magnitude `O(1)` per component (from Kaiming init). After 12 blocks, the residual has magnitude `O(sqrt(12))` (sum of 12 unit-magnitude contributions). The 1/sqrt(d_model) scaling on the embedding brings the *initial* residual contribution to `O(1/sqrt(64)) ≈ 0.125` per component, which then grows by `O(sqrt(12))` to `O(sqrt(12)/sqrt(64)) ≈ 0.43` per component. The Primer analysis shows this is the *optimal* magnitude for the residual stream to match the LM head's output magnitude (the LM head has weights of magnitude `O(1/sqrt(d_model))`, so the output logit magnitude is `O(1/sqrt(d_model))` for unit-magnitude input).

**Step-0 byte-identity**: scaling the embedding by `1/sqrt(d_model)` and the LM head by `sqrt(d_model)` is *exactly* a magnitude rescaling of the input and output. The loss at step 0 is *unchanged* (the logits are the same in magnitude, the softmax is the same, the cross-entropy is the same). **Step-0 byte-identity is exact in the loss value, but the *logit values* are different** (the LM head output is `sqrt(d_model)` larger, the softmax is sharper, but the loss is the same).

For **strict step-0 byte-identity in logits and loss**, the lever must be a *pure* embedding rescaling (no LM head compensation). In that case, the loss at step 0 is the same (the *target distribution* is the same, and the model's prediction matches baseline), but the logit magnitudes are different. The optimizer then adapts the LM head to the new scale, which is a *re-fit cost* that the 92-step horizon may not have time to amortize.

**The lever is "step-0 ≈ baseline"** in the loss, but the *logit magnitudes* differ. This is a subtle but important distinction: the loss is the same, but the gradient signal is different (the softmax is sharper with the lever, so the gradient is more peaked on the argmax token).

## Design sketch
- **Files**:
  - `configs/llm_config.py` — add `use_embed_sqrt_d_scaling: bool = False` to `LLMConfig`. Add `Tiny1M3MEmbedSqrtDConfig(Tiny1M3MConfig)` with `use_embed_sqrt_d_scaling: bool = True`.
  - `models/llm.py` — in the embedding forward (or in the post-embedding hook), apply `x = x * (1.0 / math.sqrt(config.d_model))`. Optionally, also scale the LM head output by `sqrt(config.d_model)` to keep the logit magnitudes constant. The lever can be either form (with or without LM head compensation); document both.
  - **Recommended form**: only scale the embedding (no LM head compensation). This gives the most "shocking" gradient signal to the optimizer and tests whether the smaller-magnitude input helps the residual stream's stability.
- **Config flag**: `use_embed_sqrt_d_scaling: bool = False`.
- **Param count**: **0 new params** (init-time / forward-time scaling).
- **Intuition (why it might lower val loss)**: the residual stream accumulates `O(sqrt(n_layers))` magnitude over the depth of the network. For tiny1m3m (12 layers), the residual grows by `sqrt(12) ≈ 3.5×` from input to output. The 1/sqrt(d_model) scaling on the embedding brings the input magnitude to `O(1/sqrt(64)) ≈ 0.125` per component, so the final residual is `O(0.43)` per component — well-matched to the LM head's `O(1/sqrt(d_model)) = 0.125` weight magnitude, which produces output logit magnitudes of `O(0.43 * 0.125) = O(0.054)`. The optimal logit magnitude for a 92-step training run is somewhere in this range; the standard init (no scaling) may produce too-large logits, which can cause the softmax to be too sharp and the gradient to be too peaked on the argmax token. The 1/sqrt(d_model) scaling *reduces* the logit magnitudes and *flattens* the softmax, giving a more uniform gradient signal across all vocab tokens.
- **Why it might bind at 0.94M where 159-emb-layernorm didn't**: 159 was a *full LayerNorm* on the embedding, which rescaled each token's embedding to unit RMS *and* applied a per-channel gain. This is a *directional* change (the embedding's direction is altered by the LN). 194 is a *scalar* multiplication, which preserves the *direction* of each token's embedding and only rescales the magnitude. The 159 DRIFT was caused by the directional change (the network had to re-fit the rescaled directions); 194 doesn't change directions, so the re-fit cost is minimal.

## Scale evidence
- Primer (So et al. 2021) — 100M-1.5B language modeling. Direct validation of the embedding `1/sqrt(d_model)` scaling.
- T5 (Raffel et al. 2020) — 60M-11B. The "embedding scaled by sqrt(d_model)" form is a *different* direction (T5 scales up; Primer scales down). The two forms are related but not identical.
- GPT-2 (Radford et al. 2019) — 100M-1.5B. Uses `1/sqrt(2 * n_layers)` residual scaling, which is similar in spirit to Primer's `1/sqrt(d_model)`.
- **Transfer-risk: low** — the lever has direct validation at 100M+ for the embedding-scaling form.

## Why it's worth a slot
The bet, in one sharp sentence: **Primer's `1/sqrt(d_model)` embedding scaling is a 0-param init lever with direct 100M-1.5B validation, and the closed 159-emb-layernorm (DRIFT) tested the *directional* embedding rescaling, not the *magnitude-only* rescaling** — 194 tests the magnitude-only axis, which avoids the re-fit cost that 159 incurred; a null at 0.94M would close the *embedding-magnitude* axis at our tier (the hand-tuned init is already optimal), and a win would give a 0-param init lever that transfers to larger scales (Primer's headline).

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- 159-emb-layernorm (null, DRIFT) — full LayerNorm on embedding (directional change). 194 is scalar multiplication (magnitude-only).
- 183-pre-lm-head-rmsnorm (null) — output-side norm. 194 is input-side scalar.
- 130-rezero, 142-layerscale (null) — depth-conditional residual-stream levers. 194 is init-time / forward-time scalar, no per-step overhead.
- 017-sub-ln-sandwich (null) — depth-conditional LN placement. 194 is a single scalar at the input.
- T5 (not in repo) — embedding scaled by *sqrt(d_model)* (opposite direction). 194 is the Primer (1/sqrt(d_model)) form.
