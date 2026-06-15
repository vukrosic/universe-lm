---
id: 187-lm-head-bias
status: needs-review
round: 1
updated: 2026-06-15T07:19:24Z
transfer-risk: low
plain: Add one learned additive bias to the language-model output (a single knob per vocab token, starting at 0 so step-0 is byte-identical) — a way to break output-tie without disturbing the input embedding.
---

# 187 — LM-Head Additive Bias (Per-Vocab Additive Logit Bias, init=0)

## Source
- Press et al., "T5" (JMLR 2020, arXiv:1910.10683) — uses an `lm_head bias` initialized to 0, added to the logits before softmax. The bias is per-vocab, additive, and decoupled from the input embedding.
- LLaMA 1/2/3 (Touvron et al. 2023, Dubey et al. 2024) — do *not* use an LM head bias; the embedding and the LM head share weights (tied), with no separate output bias. This is the standard modern-decoder convention.
- PaLM (Chowdhery et al. 2022, arXiv:2204.02311) — also tied embeddings, no separate LM head bias.
- Mistral / Yi / Qwen / Gemma / OLMo — tied, no separate bias. The 2023+ frontier-decoder trend is to *avoid* the LM head bias for parameter efficiency.
- T5's *untying* (separate input embed, separate LM head, plus an additive bias) was the older encoder-decoder convention; it has been mostly abandoned for decoder-only LMs in favor of tied weights.
- In-repo context: closed.md line "V/Q/K/O embeds + combos, q_gain / k_gain" closed the additive / multiplicative gain axis on the QKV projections, not the LM head. No prior lever in the repo tests a per-vocab additive bias on the LM logits. Tied-embedding is the standard config; 187 *adds* an LM head bias *on top of* tied embeddings (the bias is decoupled from the embedding matrix and is not tied to the input).

## Mechanism
Standard LM head (tied embeddings):
```
logits = Embedding.T @ final_residual                # [B, T, V] — tied with input embed
loss = cross_entropy(logits, targets)
```
With LM head bias:
```
logits = Embedding.T @ final_residual + lm_head_bias    # [B, T, V], lm_head_bias: [V]
loss = cross_entropy(logits, targets)
```
The bias `lm_head_bias ∈ R^V` is a per-vocab additive shift. With tied embeddings, the LM head's weight matrix is the same as the input embedding's weight matrix; the bias is a *separate* parameter, not part of the embedding.

**Step-0 byte-identity**: `lm_head_bias` init at `0` (a zero tensor of shape `[V]`) ⇒ `logits + 0 = logits` exactly ⇒ **byte-identical to baseline at step 0**. The gradient on the bias is `softmax(logits) − onehot(target)` — at step 0 the gradients are non-zero (the model is making confident-but-wrong predictions on the first batch), so the bias moves immediately. Step-0 *forward* is bit-identical; step-0 *backward* introduces the new gradient on the bias (which is independent of the embedding gradient and doesn't interfere with the baseline gradient flow).

**Why this lever can help even with tied embeddings**: with tied embeddings, the per-token "logit" for token `v` is the dot product of `Embedding[v]` and `final_residual`. The optimizer can adjust the per-token output *only* by changing the embedding vector, which simultaneously changes the *input* for token `v` whenever it appears in the input sequence. Adding an LM head bias lets the optimizer adjust the per-token output *independently* of the input, breaking this tie. Common use cases: (a) bias toward / against rare tokens (correcting the prior from the training data), (b) bias toward function words vs content words, (c) bias toward the most common bigrams in the corpus. The optimizer can learn any of these without disturbing the input embedding.

**Step-0 byte-identity (re-verify)**: with `lm_head_bias = 0` (a literal zero tensor), `logits + 0 = logits` exactly (no fp32 epsilon). The forward and the loss are bit-identical; the backward introduces the bias gradient. Implementer should verify with `max_abs_diff(trt_step0_logits, ctrl_step0_logits) == 0.0` *and* `max_abs_diff(trt_step0_loss, ctrl_step0_loss) == 0.0`.

## Design sketch
- **Files**:
  - `models/llm.py` — in `MinimalLLM.__init__`, allocate `self.lm_head_bias = nn.Parameter(torch.zeros(vocab_size))` (init 0 ⇒ no bias). In `forward`, after `self.lm_head(x)`, apply `logits = logits + self.lm_head_bias`. If the LM head is tied to the embedding (the standard `use_tied_embeddings=True` config), the bias is added *on top* of the tied projection and is *not* part of the tied weights. If untied, the bias is added on top of the separate LM head.
  - `configs/llm_config.py` — add `use_lm_head_bias: bool = False`. Add `Tiny1M3MLMHeadBiasConfig(Tiny1M3MConfig)` with `use_lm_head_bias: bool = True`.
  - `models/llm.py` — thread `use_lm_head_bias` into `MinimalLLM.__init__` and `forward`.
- **Config flag**: `use_lm_head_bias: bool = False`.
- **Param count**: vocab_size = 8192 (typical for tiny1m3m; verify from the config). 8192 bias params × 1 = **8192 params (+0.87% of 0.94M)**. Small but not negligible — this is the largest of the 5 filed ideas in terms of param overhead.
- **Intuition (why it might lower val loss)**: at 0.94M with tied embeddings and 3M tokens, the per-token output logit for token `v` is the dot product of `Embedding[v]` (shared with input) and `final_residual`. The optimizer can adjust this logit only by changing the embedding vector, but that has a coupling cost — the embedding also serves as the *input* for token `v`. With a separate LM head bias, the optimizer can decouple the output logit from the input embedding. A 1% param overhead buys the model a per-token output knob that doesn't disturb the input. The T5 paper showed this was a useful lever for encoder-decoder LMs at 220M-11B; the question is whether it binds at 0.94M / decoder-only / tied embeddings.
- **Why it might bind at 0.94M**: even with tied embeddings, the per-token output logit is a *dot product* between two vectors, and the dot product is a single scalar. The bias is a *direct* per-token shift, which is a much more "efficient" parameter than tweaking the embedding vector (which has d_model=64 components, all coupled to the input). With 8192 vocab tokens and 92 update steps, the bias has enough samples per token to learn a useful shift for the most common tokens. The 0.87% param overhead is on the same order as the per-block param count of 154's rebase (which had a WIN at 0.94M).

## Scale evidence
- T5 at 220M / 770M / 3B / 11B (encoder-decoder, untied, with LM head bias) — direct validation of the lever form.
- LLaMA 1/2/3 / Mistral / Yi / Qwen / Gemma / OLMo (decoder-only, tied, *no* LM head bias) — current frontier-decoder convention is to *not* use this lever; the 187 bet is that adding it back is a useful at 0.94M (where the per-token output knob is more valuable than at 7B+ where the model has enough capacity to adjust the embedding vector directly).
- **Transfer-risk: low** — the lever is well-validated at ≥100M (T5) and is a single architectural addition with no coupling concerns.

## Why it's worth a slot
The bet, in one sharp sentence: **at 0.94M with tied embeddings, the per-token output logit is fully coupled to the per-token input embedding (a 64-dim vector that's used for both input lookup and output projection), and a separate LM head bias gives the optimizer a per-token output knob that doesn't disturb the input — this decoupling is the binding benefit of the older T5-style "untying" approach, captured in a 0.87% param overhead without the full untying cost**. A null at 0.94M would close the LM-head-bias axis and tell us the 2023+ frontier-decoder trend (tied, no bias) is correct even at our tier. A win would unlock a per-vocab bias lever for the pipeline, with the bias being a small but clean architectural addition.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: `autoresearch/baseline-cache.json` box `5b8a7fea8963` (RTX 3060), `val_mean = 6.3988`, `noise_band = 0.04`, `n_measurements = 3`. Re-pull on run day.
- **WIN**: `trt_val ≤ ctrl_val_mean − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val_mean| < 0.01`. Most likely outcome per the broader "tiny parameter-count regularizers don't bind at 92 steps" pattern.
- **DRIFT**: `trt_val > ctrl_val_mean + 0.01`. Could occur if the bias over-fits to the training distribution and the val distribution has a different token prior (typical failure mode: bias toward common training tokens, away from the val distribution's less common tokens).
- **Sub-noise is inconclusive** per one-seed-only rule.

## Distinct from closed axes (defensive)
- 159-emb-layernorm (DRIFT) — input-side LN, rescaled the per-token N(0,σ_c²) distribution. 187 is *output-side* additive bias; doesn't touch the input distribution.
- 167-logit-zloss (null) — regularizer on logsumexp, an additive penalty. 187 is an additive parameter on the logits, not a regularizer. Different lever axis.
- Closed "logit softcap" axis — softcap is per-position soft-clipping. 187 is per-vocab additive shift. Different mechanism.
- Closed "tied QK" axis — the QK matrices are tied, but this is unrelated to the LM head bias. Different architectural location.
- T5-RPE (166, null) — per-head additive bias on the *attention* logits. 187 is per-vocab additive bias on the *LM head* logits. Different architectural location (attention vs LM head).
- No prior lever in the repo tests a per-vocab LM head bias. Fresh axis.
