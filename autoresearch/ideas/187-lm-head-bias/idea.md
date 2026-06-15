---
id: 187-lm-head-bias
status: needs-review
round: 2
updated: 2026-06-15T07:58:25Z
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
- In-repo context: closed.md line "V/Q/K/O embeds + combos, q_gain / k_gain" closed the additive / multiplicative gain axis on the QKV projections, not the LM head. No prior lever in `closed.md` tests a per-vocab additive bias on the LM logits. **But see "Existing wiring" below — the lever is already wired as the OH5 VocabBias output-head probe.** Tied-embedding is the standard config; 187 *enables* the LM-head-bias flag on top of the ALiBi champion (the bias is decoupled from the embedding matrix and is not tied to the input).

## Existing wiring (already in repo — OH5 VocabBias)
The exact lever `logits += b_v` with `b_v = zeros(vocab_size)` is **already wired**:
- **Config flag**: `use_vocab_bias: bool = False` at `configs/llm_config.py:545` (LLMConfig field, comments reference `docs/research/output_head/plan.md` row OH5).
- **Param allocation**: `models/llm.py:1307-1313` — gated on `use_vocab_bias`, allocates `self.vocab_bias = nn.Parameter(torch.zeros(config.vocab_size))`.
- **Forward hook**: `models/llm.py:1880-1884` — `if self.use_vocab_bias: logits = logits + self.vocab_bias`, applied after softcap so it's a logit op that flows into eval CE legitimately.
- **Plan row**: `docs/research/output_head/plan.md:86` (OH5) — *"VocabBias | logits += b_v | per-vocab additive bias (learned unigram prior), b=0 init | vocab_size"*. Plan also notes (line 88): *"OH5 is many params but trivial compute (one add); it mostly re-learns token frequency."*

So 187 doesn't *add* wiring — it specifies the **A/B design** (control = ALiBi champion vs treatment = ALiBi champion + vocab bias; expected Δ; pass/fail bar) for a run that uses the existing OH5 flag. Canonical spec is the OH5 plan row; the prose below is the design sketch / intuition / pass-fail bar.

## Mechanism
Standard LM head (tied embeddings):
```
logits = Embedding.T @ final_residual                # [B, T, V] — tied with input embed
loss = cross_entropy(logits, targets)
```
With LM head bias (OH5 lever):
```
logits = Embedding.T @ final_residual + vocab_bias    # [B, T, V], vocab_bias: [V]
loss = cross_entropy(logits, targets)
```
The bias `vocab_bias ∈ R^V` is a per-vocab additive shift. With tied embeddings, the LM head's weight matrix is the same as the input embedding's weight matrix; the bias is a *separate* parameter, not part of the embedding.

**Step-0 byte-identity**: `vocab_bias` init at `0` (a zero tensor of shape `[V]`) ⇒ `logits + 0 = logits` exactly ⇒ **byte-identical to the ALiBi-champion baseline at step 0**. The gradient on the bias is `softmax(logits) − onehot(target)` — at step 0 the gradients are non-zero (the model is making confident-but-wrong predictions on the first batch), so the bias moves immediately. Step-0 *forward* is bit-identical; step-0 *backward* introduces the new gradient on the bias (which is independent of the embedding gradient and doesn't interfere with the baseline gradient flow).

**Why this lever can help even with tied embeddings**: with tied embeddings, the per-token "logit" for token `v` is the dot product of `Embedding[v]` and `final_residual`. The optimizer can adjust the per-token output *only* by changing the embedding vector, which simultaneously changes the *input* for token `v` whenever it appears in the input sequence. Adding an LM head bias lets the optimizer adjust the per-token output *independently* of the input, breaking this tie. Two complementary framings of *what the bias will learn*:
- **(187 framing)** output/input decoupling — a per-token scalar shift that absorbs the part of the output logit that the embedding can't efficiently represent without disturbing the input.
- **(OH5 framing)** learned unigram prior — at the limit, the bias converges to `log p(v)` (training token frequency), absorbing an "unmodeled unigram prior" at training cost.

Both are correct; they describe the same equilibrium from different angles. A null on this A/B would tell us the optimizer either can't absorb the unigram prior under 92 steps, or the val distribution's token prior matches the training prior well enough that the bias has nothing to correct.

**Step-0 byte-identity (verify against champion)**: with `vocab_bias = 0` (a literal zero tensor), `logits + 0 = logits` exactly (no fp32 epsilon). The forward and the loss are bit-identical; the backward introduces the bias gradient. Implementer must verify with `max_abs_diff(trt_step0_logits, ctrl_step0_logits) == 0.0` *and* `max_abs_diff(trt_step0_loss, ctrl_step0_loss) == 0.0`, where the **ctrl is the Tiny1M3MAlibiConfig champion stack**, not plain Tiny1M3MConfig.

## Design sketch
- **Files**: no new files; 187 only adds a config subclass and reuses the existing wiring.
  - `configs/llm_config.py` — add `Tiny1M3MAlibiLMHeadBiasConfig(Tiny1M3MAlibiConfig)` with `use_vocab_bias: bool = True`. **Stacks on the current ALiBi champion** (val 6.2403), matching the 184-logit-scale precedent (`Tiny1M3MLogitScaleConfig(Tiny1M3MAlibiConfig)` at line 2416) and the 183-pre-LM-head-RMSNorm precedent (`Tiny1M3MPreLMHeadRMSNormConfig(Tiny1M3MAlibiConfig)` at line 2449).
  - No changes to `models/llm.py` — the OH5 flag and forward hook are already in place.
- **Config flag**: `use_vocab_bias: bool = True` (already exists in LLMConfig; 187 just enables it on top of the ALiBi champion).
- **Param count**: vocab_size = **49152** (verified from `configs/llm_config.py:26`). 49152 bias params × 1 = **49,152 params (+5.23% of 0.94M)** — a *sizeable* param injection, on the same order as the entire per-block budget of typical lever ideas. The plan.md row OH5 explicitly tags this as *"many params but trivial compute"* — the cost is param budget, not compute. (Earlier draft said vocab_size=8192 / 0.87% — that was wrong; the 6× correction reframes this as a *budget-matched* lever rather than a "small but not negligible" one.)
- **Intuition (why it might lower val loss)**: at 0.94M with tied embeddings, the per-token output logit for token `v` is `Embedding[v] · final_residual` (a 64-d dot product). The optimizer can adjust this logit only by changing the embedding vector — but the embedding also serves as the *input* for token `v`, so the move is constrained by the input-coupling cost. With a separate per-vocab bias, the optimizer can decouple the output logit from the input embedding. A ~5% param overhead buys the model a per-token output knob (1-D scalar per token) that doesn't disturb the input.
- **Why it might bind at 0.94M**: even with tied embeddings, the per-token output logit is a *dot product* — a single scalar — and a per-vocab bias is a *direct* scalar shift, which is a strictly more parameter-efficient axis than moving the embedding vector (64 components, all coupled to the input). With 49,152 vocab tokens and 92 update steps × ~3M tokens of gradient signal, the bias has enough samples per token to learn a meaningful shift for the most common tokens; rare tokens stay pinned near 0 (a built-in sparsity-of-effect — the lever self-limits to where the data lives). The ~5% overhead is *larger* than a typical lever, so the bar is correspondingly higher — a +5% param cost needs a Δval ≪ −0.005 to clear the portfolio threshold.

## Scale evidence
- T5 at 220M / 770M / 3B / 11B (encoder-decoder, untied, with LM head bias) — direct validation of the lever form.
- LLaMA 1/2/3 / Mistral / Yi / Qwen / Gemma / OLMo (decoder-only, tied, *no* LM head bias) — current frontier-decoder convention is to *not* use this lever; the 187 bet is that adding it back is useful at 0.94M (where the per-token output knob is more valuable than at 7B+ where the model has enough capacity to adjust the embedding vector directly).
- **Transfer-risk: low** — the lever is well-validated at ≥100M (T5) and is a single architectural addition with no coupling concerns.

## Why it's worth a slot
The bet, in one sharp sentence: **at 0.94M with tied embeddings, the per-token output logit is fully coupled to the per-token input embedding (a 64-dim vector shared between input lookup and output projection), and a separate per-vocab additive bias gives the optimizer a direct scalar output knob that doesn't disturb the input — at +5.23% param cost (49,152 params) this is a budget-matched lever, not a free knob, and the bar should be set accordingly**. A null at 0.94M would close the LM-head-bias axis and tell us the 2023+ frontier-decoder trend (tied, no bias) is correct even at our tier. A win would unlock a per-vocab bias lever for the pipeline (with a meaningful param cost, so the unlock is gated on the Δ).

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: `autoresearch/baseline-cache.json` box `5b8a7fea8963` (RTX 3060). As of 2026-06-15T07:04:48Z (the pinned measurement): `val_mean = 6.2403`, `noise_band = 0.04`, `n_measurements = 3`. **Re-pull on run day** — the `baseline.sh` script writes the freshest measurement, and the cache can be refreshed mid-week.
- **Control stack**: `Tiny1M3MAlibiConfig` (the current champion; 175 alibi-slopes WIN of Δ-0.1585 over the pre-ALiBi baseline of 6.3988). The earlier draft cited the pre-ALiBi 6.3988 baseline — that's wrong; the 187 A/B is vocab-bias-only vs the ALiBi champion stack, so the bar is `Tiny1M3MAlibiLMHeadBiasConfig` vs `Tiny1M3MAlibiConfig`, not vs `Tiny1M3MConfig`.
- **WIN**: `trt_val ≤ ctrl_val_mean − 0.005` AND clears the two-ctrl rule (i.e. `trt_val < both_two_ctrl_means`). With ctrl ≈ 6.2403, the WIN bar is **trt ≤ 6.2353** before the two-ctrl rule applies.
- **NULL**: `|trt_val − ctrl_val_mean| < 0.01` ⇒ trt ∈ [6.2303, 6.2503] is treated as a null per the one-seed-only rule (sub-noise is inconclusive).
- **DRIFT**: `trt_val > ctrl_val_mean + 0.01` ⇒ trt > 6.2503. Could occur if the bias over-fits to the training unigram prior and the val distribution has a meaningfully different token prior.
- **Sub-noise is inconclusive** per one-seed-only rule (seed 42, no seed sweeps).
- **Expected Δ (OH5 framing)**: Δval ∈ [−0.005, −0.02]. The OH5 plan row is the prior for this lever at this tier; the win threshold of −0.005 sits at the *low end* of that range, so a borderline null is the most likely outcome (consistent with the broader "tiny parameter-count regularizers don't bind at 92 steps" pattern).

## Distinct from closed axes (defensive)
- 159-emb-layernorm (DRIFT) — input-side LN, rescaled the per-token N(0,σ_c²) distribution. 187 is *output-side* additive bias; doesn't touch the input distribution.
- 167-logit-zloss (null) — regularizer on logsumexp, an additive penalty. 187 is an additive parameter on the logits, not a regularizer. Different lever axis.
- Closed "logit softcap" axis — softcap is per-position soft-clipping. 187 is per-vocab additive shift. Different mechanism.
- Closed "tied QK" axis — the QK matrices are tied, but this is unrelated to the LM head bias. Different architectural location.
- T5-RPE (166, null) — per-head additive bias on the *attention* logits. 187 is per-vocab additive bias on the *LM head* logits. Different architectural location (attention vs LM head).
- **OH5 VocabBias is already wired in the repo** (`use_vocab_bias` flag, plan row) — 187 specifies the *A/B design* (control vs ALiBi champion + treatment vs ALiBi champion + vocab bias), not new wiring. No prior idea in `closed.md` or `ideas/` has *run* this A/B on the ALiBi champion stack at 0.94M.

## Reviser note (r1)
- All four r1 findings applied:
  1. vocab_size corrected from 8192 → 49152; param overhead recomputed from +0.87% → **+5.23%** (49,152 params); design sketch and "Why it's worth a slot" reframed as a *budget-matched* lever rather than a "small but not negligible" one.
  2. Baseline-cache reference updated from the stale 6.3988 (pre-ALiBi) to the pinned 6.2403 (post-175-ALiBi champion). A/B framing clarified: control = `Tiny1M3MAlibiConfig`, treatment = `Tiny1M3MAlibiLMHeadBiasConfig`. WIN bar tightened from 6.3938 → **6.2353** to reflect the actual champion.
  3. "Existing wiring" section added: `use_vocab_bias` flag at `configs/llm_config.py:545`, param allocation at `models/llm.py:1307-1313`, forward hook at `models/llm.py:1880-1884`, plan row OH5 at `docs/research/output_head/plan.md:86`. Design sketch now calls for a `Tiny1M3MAlibiLMHeadBiasConfig(Tiny1M3MAlibiConfig)` subclass (stacking on the ALiBi champion), matching the 184-logit-scale and 183-pre-LM-head-RMSNorm precedents.
  4. Mechanism prose trimmed to cite OH5 plan row as canonical; OH5 framing ("re-learns token frequency") and 187 framing ("output/input decoupling") unified in the "Why this lever can help" paragraph so the run spec is unambiguous about both the expected *mechanism* and the expected *magnitude*.
- Step-0 byte-identity verification: explicit note added that the ctrl must be the ALiBi champion stack, not plain `Tiny1M3MConfig`.
- No disagreements with the r1 findings; all applied.
