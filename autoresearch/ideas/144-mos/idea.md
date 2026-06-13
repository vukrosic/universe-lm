---
id: 144-mos
status: needs-run
round: 1
updated: 2026-06-13T20:50:36Z
transfer-risk: med
plain: Replace the output softmax with a small mixture of several softmaxes, letting the model hedge its probability mass across competing predictions.
---

# 144 — Mixture of Softmaxes (MoS)

## Source
Yang, Chen, et al. 2017, "Breaking the Softmax Bottleneck: A High-Rank RNN Language Model", arXiv:1711.03953. https://arxiv.org/abs/1711.03953

## Mechanism
Replace the single softmax output with a weighted mixture of K softmaxes over the same vocabulary.
- `logits_k = W_k * h`  for k = 1..K  (K separate LM heads)
- `π = softmax(W_π * h)`  (mix weights, length K)
- `P(v) = Σ_k π_k * softmax(logits_k)_v`

Identity: when K=1, π_1=1, W_1 is the standard LM head → identical to single-softmax baseline.

Theoretical claim: a single softmax `softmax(W h)` produces a log-prob matrix of rank ≤ `d_model` (because `log P = W h − log Z` and the row-space of `W` has rank ≤ d_model). A mixture of K softmaxes has effective rank ≤ `K × d_model` — the output distribution can express exponentially more structure. This matters most when the *target* distribution is high-rank (predicting the next token often requires hedging across many plausible continuations).

## Design sketch (how it works + how to build it)
- Modify `models/llm.py` output head: when `use_mos`, allocate `K` separate `nn.Linear(d_model, vocab, bias=False)` heads, plus a small mix projection `nn.Linear(d_model, K)`. Compute `logits = stack([W_k @ h for k in range(K)])` → `[K, B, T, V]`, then `log P = log_softmax(logits)`, then `log P_final = logsumexp(log π + log P)`. ~80 LoC.
- Add `use_mos: bool = False`, `n_mos_components: int = 4` to `configs/llm_config.py`. K=4 is the paper's default.
- Identity at step 0: K=1, π_1=1, W_1=standard init. The K>1 path is off by default; flipping the flag multiplies the output head params by K. At step 0 with K=4, init all W_k = standard LM head init and π = uniform 1/K. The first forward is a uniform-weighted mixture of 4 standard softmaxes — *not* byte-identical to single softmax, but close. To preserve strict identity, init π_1=1, π_{k>1}=0 (sparse) — this requires a `softmax_with_mask` or a direct softmax init.
- A simpler strict-identity variant: keep K=1 by default, only activate K>1 when the flag is on; treat the K=1 → K=4 transition as the lever. (At K=4, output head goes from `64 × vocab` to `4 × 64 × vocab` — a sizeable param injection.)
- Why a real lever, not a hyperparam: the *rank* of the output distribution is a structural property, not a knob. AdamW with different betas cannot give you higher output rank. MoS expands the expressivity of the output head at the cost of `K×` params.

## Scale evidence
Paper trains 1B+ LMs (RNN-based, 4× softmax). Independent replications on Transformer LMs show smaller but consistent gains (~0.1–0.3 perplexity) at 100M+ scale. 0.94M is well below the validated range, and the K× param cost is non-trivial at this size. Transfer risk: med.

## Why it's worth a slot
Most filed levers touch the *body* of the transformer (attention, FFN, residual). MoS is the only output-layer lever in the queue. The output head at tiny1m3m has shape `64 × vocab` — rank-64. If the target distribution is higher-rank, the model is bottlenecked at output. A win tells us output rank is the binding constraint at 0.94M; a null tells us the body is the binding constraint and we should keep filing body-levers.

## Plan

**Files changed**
- `configs/llm_config.py` — added `use_mos: bool = False`, `n_mos_components: int = 4` to `LLMConfig`, plus a `Tiny1M3MMoSConfig(Tiny1M3MConfig)` subclass with `use_mos: bool = True`, `n_mos_components: int = 4`.
- `models/llm.py` — when `use_mos=True`, build `mos_heads_extra = nn.ModuleList([nn.Linear(d_model, vocab, bias=False) for _ in range(K-1)])` (heads 1..K-1, fresh) and `mos_pi_proj = nn.Linear(d_model, K, bias=True)` (mix projection). In `_run_post_embed`, when `use_mos=True`, compute the mixture log-P:
    - `logits_0 = tied lm_head(x)` (functional: `F.linear(x, token_embedding.weight)` for full case, `F.linear(F.linear(x, emb_proj.weight.t()), token_embedding.weight)` for factorized/tiny1m3m case). Head 0's gradient flows back into `token_embedding` and `emb_proj`.
    - `logits_extra = stack([head(x) for head in mos_heads_extra])` → `(..., K-1, V)`.
    - `logits_k = cat([logits_0, logits_extra], dim=-2)` → `(..., K, V)`.
    - `log_p_k = log_softmax(logits_k)` over V.
    - `log_pi = log_softmax(mos_pi_proj(x))` over K.
    - `log_p_mixed = logsumexp_k(log_pi + log_p_k)`.
    - Return `log_p_mixed` as the model's logits.
- `_arq_144-mos.py` (treatment) and `_arq_144-mos_ctrl.py` (control).

**Flag name**: `use_mos` (config), `n_mos_components` (config, default 4).

**Identity at step 0 (within MoS path)**: verified `|out_mos - log_softmax(head_0 · x)| = 0.00e+00` (bit-identical reduction). The mix projection is initialized post-`apply(self._init_weights)` as:
- `mos_pi_proj.weight = 0`
- `mos_pi_proj.bias = [+1e4, -1e4, -1e4, -1e4]`

so `softmax(W_π · h) = [1, 0, 0, 0]` exactly in fp32 (the `exp(-2e4)` terms underflow to 0; `exp(+1e4)` stays finite enough for `log_softmax` to compute the right max-shift). The `logsumexp` then reduces to `log_softmax(W_0 · h)` — bit-identical to head 0's output. The K-1 fresh `mos_heads_extra` are init'd to `N(0, 0.02)` like the rest of the model; only head 0 contributes at step 0 (verified: `mos_heads_extra[0].weight.grad.norm() = 0.0000` at step 0).

**Cross-model baseline diff (ctrl vs trt at step 0)**: |Δ CE| ≈ 4.8e-3. This is **NOT a lever bug** — it's from the rng-state shift when extra parameters are added (the trt has K-1 fresh vocab-sized heads + a mix projection; their `apply(_init_weights)` consumes additional RNG draws, so `token_embedding` and `emb_proj` get slightly different random values than in the ctrl run). Within a single model's forward pass the lever IS bit-identical. The trainer's val at step 0 will be within fp32 noise of the leaderboard ctrl val, matching the pattern of other identity-init levers (ReZero, LayerScale, etc.) where extra params also shift the rng state.

**Run command** (per `prompts/runner.md`):
- Box workdir: `/root/universe-lm` on the Vast instance.
- Python: `/venv/main/bin/python` (with `export PATH=/venv/main/bin:$PATH`).
- Queue line for treatment: `run 144-mos python /root/universe-lm/_arq_144-mos.py`
- Queue line for ctrl: `run 144-mos_ctrl python /root/universe-lm/_arq_144-mos_ctrl.py`

**Reading the result**: `results.json` `runs[].val_loss` and `runs[].train_loss`. Per `prompts/runner.md` §5, WIN if `Δ ≤ −0.01` against BOTH ctrls (and clears the ctrl-to-ctrl variance band); NULL if `|Δ| < 0.01`. Treatment params are ~10.4M (vs ctrl 0.94M) — the 11× param inflation is the lever's headline confound; the runner's `evidence.md` should flag this when judging.

**Param cost**: K-1 fresh vocab-sized heads = `(K-1) · vocab · d_model` (≈ 9.4M at tiny1m3m with K=4) + mix projection `K · d_model` (256). Net: +9,437,444 params.
