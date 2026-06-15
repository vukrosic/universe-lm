---
id: 183-pre-lm-head-rmsnorm
status: needs-run
round: 1
updated: 2026-06-15T07:28:58Z
transfer-risk: low
plain: Add one final normalization layer (Gemma 2 / LLaMA-3 style) right before the language-model head so the output sees a well-scaled hidden state, starting with the gain at 1 so step-0 matches the baseline.
---

# 183 — Pre-LM-Head RMSNorm (Gemma 2 / LLaMA-3 Final Norm)

## Source
- Gemma 2 (Team et al. 2024, arXiv:2408.00118) — adds a final RMSNorm right before the LM head ("output norm"). Validated at 2B / 9B / 27B.
- LLaMA 3 (Dubey et al. 2024, arXiv:2407.21783) — same pattern, "norm after the last decoder block". Validated at 8B / 70B / 405B.
- Qwen 2.5 (Yang et al. 2024, arXiv:2412.15115) — also uses a pre-LM-head norm. Validated at 0.5B-72B.
- OLMo 2 (OLMo Team 2024, arXiv:2412.04454) — pre-LM-head LayerNorm. Validated at 1B / 7B.
- In-repo context: 159-emb-layernorm closed drift (DRIFT, embedding-side LN; rescaled the per-token N(0,σ_c²) distribution and the 0.94M/12L model had no spare capacity to re-fit). 183 is the *output-side* analog, NOT a duplicate: the failure mode at 0.94M was that an extra LN on the *input* embedding cost a re-fit; an extra LN on the *output* stream stabilizes the LM head's input distribution and shouldn't trigger the same DRIFT. NormFormer (Shleifer et al. 2021, arXiv:2110.09423) is the earlier architectural reference for "extra LN at attention / FFN output", but the *pre-LM-head* placement is the modern LLaMA-3 / Gemma-2 / Qwen-2.5 / OLMo-2 convention.

## Mechanism
Standard LM head:
```
logits = LM_head(final_residual)         # final_residual: [B, T, d_model]
```
With pre-LM-head RMSNorm:
```
normed = RMSNorm(final_residual)         # [B, T, d_model]
logits = LM_head(normed)
```
RMSNorm(x) = x * rsqrt(mean(x²) + ε) * gain + bias. Init `gain = 1`, `bias = 0` ⇒ `RMSNorm(x) = x * rsqrt(mean(x²) + ε)`. The norm rescales the per-token hidden state to unit RMS *before* the LM head, equalizing the magnitude the head sees across tokens and reducing the impact of residual-stream outliers on the output distribution.

**Step-0 identity**: with `gain = 1, bias = 0`, the output is `x * rsqrt(mean(x²) + ε)`. At step 0 the residual stream is roughly the sum of a few near-Gaussian components; `rsqrt(mean(x²) + ε)` is close to `1 / σ` where `σ` is the per-token RMS. This is *not* bit-identical to baseline (the residual stream gets rescaled by a small factor ≈ 1/σ per token), but the loss and the gradient direction are essentially unchanged at step 0 — the cross-entropy gradient w.r.t. the residual stream is dominated by the LM head's weight magnitudes, not by the rescaling factor. The lever is `step-0 ≈ baseline` (within 1e-4 in val loss at step 0, not byte-identical). This matches the 175-alibi-slopes lever's "step-0 ≈ baseline" pattern (the slope init of 0 is also not bit-identical, but the loss effect is null at step 0).

If a strict byte-identity is required, parameterize as `RMSNorm(x) * scale + (1 - scale) * x` with a learned `scale` init at `0`; with `scale = 0` the output is `x` exactly at step 0, and the optimizer can grow `scale` toward `1` to engage the norm. This adds 1 scalar param and gives true byte-identity. **Default to the gate-form for simplicity**; document the alternative.

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_pre_lm_head_rmsnorm: bool = False` to `MinimalLLM.__init__` (or pass via a new module `PreLMHeadRMSNorm` in `models/layers.py`). Allocate `self.pre_head_norm = nn.RMSNorm(d_model, eps=1e-6)` (or hand-rolled `x * rsqrt(mean(x²)+ε) * gain`). Init: PyTorch's `nn.RMSNorm` defaults `gain=1, bias=0`.
  - `configs/llm_config.py` — add `use_pre_lm_head_rmsnorm: bool = False`. Add `Tiny1M3MPreLMHeadRMSNormConfig(Tiny1M3MConfig)` with `use_pre_lm_head_rmsnorm: bool = True`.
  - `models/llm.py` — in `MinimalLLM.forward`, after the final transformer block, apply `x = self.pre_head_norm(x)` *before* `self.lm_head(x)`. The `lm_head` reads from the normalized stream.
- **Config flag**: `use_pre_lm_head_rmsnorm: bool = False`.
- **Param count**: d_model=64, no bias. 1 RMSNorm × 64 gain weights = **64 params (+0.007% of 0.94M)**. Negligible.
- **Intuition (why it might lower val loss)**: the LM head reads from a residual stream whose magnitude is the sum of contributions from all 12 transformer blocks. At 0.94M, with tied embeddings, the LM head's weight matrix and the embedding's weight matrix are the same, so the LM head's effective input scale and the embedding's output scale are tied. The pre-LM-head norm decouples them — the LM head always sees unit-RMS inputs, regardless of how the residual stream accumulates. This is most useful at small scales where the residual stream's magnitude is poorly calibrated (typical failure mode: a few "outlier" tokens dominate the LM head's logits, suppressing probability on the rest of the vocab). A unit-RMS input gives the LM head a more uniform gradient signal across tokens.
- **Why it might bind at 0.94M where 159-emb-layernorm DRIFTed**: 159 was *embedding-side* — it rescaled the input, requiring the rest of the network to re-fit the rescaled distribution. 183 is *output-side* — it normalizes the final stream, which the LM head reads from, but does not change the residual stream's input to the network. The model can learn the same internal representations as baseline; the LM head just sees a better-conditioned input. The DRIFT pattern from 159 (rescaling the input distribution) doesn't apply on the output side.

## Scale evidence
- Gemma 2: 2B / 9B / 27B (≥100M, direct).
- LLaMA 3: 8B / 70B / 405B (≥100M, direct).
- Qwen 2.5: 0.5B-72B (≥100M, direct).
- OLMo 2: 1B / 7B (≥100M, direct).
- **Transfer-risk: low** — the lever is well-validated across four recent frontier-model families at 0.5B-405B. The mechanism is scale-free (a single LN/RMSNorm at a fixed architectural location). The DRIFT at 0.94M from 159 (embedding-side) is a *different* placement and shouldn't transfer.

## Why it's worth a slot
The bet, in one sharp sentence: **the pre-LM-head norm is the most widely-adopted architectural change in frontier LMs from 2024-2025 (Gemma 2, LLaMA 3, Qwen 2.5, OLMo 2 all use it), and the closest in-repo analog (159-emb-layernorm) tested the embedding-*input* side and DRIFTed for a different reason (rescaling the input distribution), not the output side** — 183 tests the output-side placement that the four frontier families all use, and if it binds at 0.94M it would confirm that the binding LM-head-conditioning benefit can be captured at our tier; a null at 0.94M would close the pre-LM-head-norm axis at our tier (consistent with the broader "per-layer / per-block conditional levers don't bind at 12L" pattern from 017/111/116/130/142) and tells us the four frontier families' gain is a Phase-2+ scale effect.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: `autoresearch/baseline-cache.json` box `5b8a7fea8963` (RTX 3060), `val_mean = 6.3988`, `noise_band = 0.04`, `n_measurements = 3`. Re-pull on run day.
- **WIN**: `trt_val ≤ ctrl_val_mean − 0.005` AND clears the two-ctrl rule (per the §2 pipeline rule).
- **NULL**: `|trt_val − ctrl_val_mean| < 0.01` (matches the band the 159 DRIFT was measured in; 159 was +0.0712, so a NULL for 183 would be the 183-specific confirmation that *output-side* LN doesn't bind at 0.94M, distinct from the 159 *input-side* DRIFT).
- **DRIFT**: `trt_val > ctrl_val_mean + 0.01`. If 183 DRIFTs the same way as 159 (input-side), the conclusion is that *any* extra LN at our tier is a re-fit cost the model can't afford — broader lever family closure. If 183 NULLs inside band but doesn't DRIFT, the conclusion is "output-side LN is silent at 0.94M, gain is a Phase-2 scale effect".
- **Sub-noise is inconclusive** per one-seed-only rule: `|Δ| < 0.005` ⇒ logged NULL with `cache_authoritative: true`.

## Distinct from closed axes (defensive)
- 159-emb-layernorm — input-side LN, DRIFT. Different placement (embedding input vs final stream pre-LM-head).
- Closed norm zoo (pnorm / manhattan / center / squash / clip / channelscale) — all *modify the existing norm's operation*; 183 *adds a new norm at a new location*. Different axis.
- 142-layerscale, 130-rezero, 017-sub-ln-sandwich — depth-conditional residual-stream levers, all null at 12L. 183 is NOT depth-conditional (single global norm, applied once at the end). Distinct.
- 021-value-residual (WIN), 168-av-output-carry (null), 164-q-carry (null) — cross-block V/Q carry. 183 is *not* a carry; it's a static normalization. Distinct.
- 016-qk-norm (WIN) — pre-softmax attention normalization. 183 is *output-side*, pre-LM-head. Different placement.

## Plan

- **Files**:
  - `configs/llm_config.py` — add `use_pre_lm_head_rmsnorm: bool = False` to the `LLMConfig` base, and a `Tiny1M3MPreLMHeadRMSNormConfig(Tiny1M3MAlibiConfig)` subclass with `use_pre_lm_head_rmsnorm: bool = True` (stacks on the current champion).
  - `models/llm.py` — in `MinimalLLM.__init__`, when the flag is on, register `self.pre_head_norm = nn.RMSNorm(config.d_model, eps=1e-6)` (default `weight=1, bias=0` survives the global `_init_weights`) and `self.pre_head_scale = nn.Parameter(torch.zeros(()))` (scalar gate). In `_run_post_embed`, after `self.output_dropout(x)` and before the `lm_head` call, apply the gated mix `x = (1 − scale) · x + scale · RMSNorm(x)`. With flag off, the modules are not built and the forward path is byte-identical to the champion. With flag on at step 0, `scale = 0` ⇒ the mix is exactly `x`, byte-identical to the champion.
  - `_arq_183-pre-lm-head-rmsnorm.py` (repo root, same convention as `_arq_175-alibi-slopes.py`) — subclass `Tiny1M3MPreLMHeadRMSNormConfig` and call `train_llm.main()` with the standard tiny1m3m/seed-42 args.
- **Config flag**: `use_pre_lm_head_rmsnorm: bool = False`. Default off; baseline path is byte-identical.
- **Step-0 identity**: with the gate-form `y = (1 − scale)·x + scale·RMSNorm(x)` and `scale = 0` init, the output is exactly `x` for every token, in fp32 (no rounding, no rescaling, no dropout interaction). The optimizer grows `scale` toward `1` to engage the RMSNorm; the param count is 1 scalar (AdamW) + 64 gain weights (Muon) at d_model=64.
- **Param count**: 64 (RMSNorm gain) + 1 (scalar gate) = 65 extra params (+0.007% of 0.94M). Negligible.
- **Run command**:
  ```
  /venv/main/bin/python _arq_183-pre-lm-head-rmsnorm.py
  ```
  on the RTX 3060 box at `tiny1m3m`, seed 42. Champion reference: `autoresearch/champion.json` (`Tiny1M3MAlibiConfig`, val 6.2403, band 0.04).
- **Read val**: tail `autoresearch/records.jsonl` for the matching trt record; the daemon's judge compares `trt_val` against the champion `val` and the cached `baseline-cache.json` 175-box (val_mean 6.3988, band 0.04). Sub-noise (|Δ| < 0.005) is logged NULL with `cache_authoritative: true` per the one-seed-only rule.
