---
id: 144-mos
status: done
round: 2
updated: 2026-06-13T22:06:19Z
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
- `configs/llm_config.py` — added `use_mos: bool = False`, `n_mos_components: int = 4`, `mos_chunk_size: int = 128` to `LLMConfig`, plus a `Tiny1M3MMoSConfig(Tiny1M3MConfig)` subclass with `use_mos: bool = True`, `n_mos_components: int = 2` (down from K=4 in round 1).
- `models/llm.py` — when `use_mos=True`, build `mos_heads_extra = nn.ModuleList([nn.Linear(d_model, vocab, bias=False) for _ in range(K-1)])` (heads 1..K-1, fresh) and `mos_pi_proj = nn.Linear(d_model, K, bias=True)` (mix projection). In `_run_post_embed`, when `use_mos=True`, compute the mixture log-P **in chunks along B*T** (default `mos_chunk_size=128`) using **non-in-place** ops:
    - `logits_0 = tied lm_head(x)` (functional: `F.linear(x, token_embedding.weight)` for full case, `F.linear(F.linear(x, emb_proj.weight.t()), token_embedding.weight)` for factorized/tiny1m3m case). Head 0's gradient flows back into `token_embedding` and `emb_proj`.
    - `log_p_0 = logits_0 - logsumexp(logits_0, dim=-1, keepdim=True)` (non-in-place `sub`).
    - `log_pi = log_softmax(mos_pi_proj(x))` over K.
    - `log_p_mixed = log_p_0 + log_pi[:, 0:1]` then `logaddexp` over `k=1..K-1`:
        ```
        for k in 1..K-1:
            log_p_h = mos_heads_extra[k-1](x) - logsumexp(...)
            log_p_mixed = logaddexp(log_p_mixed, log_pi[:, k:k+1] + log_p_h)
        ```
    - Chunks are concatenated at the end. Return `log_p_mixed` as the model's logits.
- `_arq_144-mos.py` (treatment) and `_arq_144-mos_ctrl.py` (control).

**Flag name**: `use_mos` (config), `n_mos_components` (config, default 4), `mos_chunk_size` (config, default 128).

**Round-2 OOM fixes**:
1. **`n_mos_components=2` (was K=4)**: the K=4 + chunked path still OOM'd on the RTX 3060 12GB because the downstream `F.cross_entropy(logits.view(-1, V), …)` internally materializes a (4096, 49152) fp32 tensor (≈ 805 MB) AND K=4's 3 fresh vocab-sized heads cost ≈ 450 MB of AdamW state — pushing the trainer process past 12 GB. Halving K cuts the MoS optimizer state to ≈ 150 MB, restoring headroom while keeping the effective rank lever (`K·d_model = 128 vs 64`, a 2× expansion).
2. **`mos_chunk_size=128` (was 256)**: per-chunk peak halved to ~125 MB (5 tensors of size 128·49152 = 25 MB each, ≈ 125 MB concurrent).
3. **Non-in-place `sub` (was `sub_`)**: the round-1 chunked recode used `logits_0.sub_(log_z_0)` for memory savings, but this clobbered the autograd version counter on `logits_0` and `cross_entropy.backward()` threw "variable has been modified by an inplace operation". Switched to `logits_0 - log_z_0` (allocates one extra ~25 MB tensor per chunk at chunk=128 — acceptable).

**Identity at step 0 (within MoS path)**: with the new K=2 init, `mos_pi_proj.weight = 0` and `mos_pi_proj.bias = [+1e4, -1e4]` ⇒ `softmax(W_π · h) = [1, 0]` exactly in fp32 (the `exp(-2e4)` term underflows to 0). The `logsumexp` then reduces to `log_softmax(W_0 · h)` — bit-identical to head 0's output (verified `chunked vs un-chunked` reference = `0.00e+00`). The 1 fresh `mos_heads_extra[0]` is init'd to `N(0, 0.02)` like the rest of the model; only head 0 contributes at step 0 (verified: `mos_heads_extra[0].weight.grad.norm() = 0.0000` at step 0).

**Cross-model baseline diff (ctrl vs trt at step 0)**: |Δ CE| ≈ 5e-2 max abs / ~1e-3 mean abs. This is **NOT a lever bug** — it's from the rng-state shift when extra parameters are added (the trt has K-1=1 fresh vocab-sized head + a mix projection; their `apply(_init_weights)` consumes additional RNG draws, so `token_embedding` and `emb_proj` get slightly different random values than in the ctrl run). Within a single model's forward pass the lever IS bit-identical. The trainer's val at step 0 will be within fp32 noise of the leaderboard ctrl val, matching the pattern of other identity-init levers (ReZero, LayerScale, etc.) where extra params also shift the rng state.

**Run command** (per `prompts/runner.md`):
- Box workdir: `/root/universe-lm` on the Vast instance.
- Python: `/venv/main/bin/python` (with `export PATH=/venv/main/bin:$PATH`).
- Queue line for treatment: `run 144-mos python /root/universe-lm/_arq_144-mos.py`
- Queue line for ctrl: `run 144-mos_ctrl python /root/universe-lm/_arq_144-mos_ctrl.py`

**Reading the result**: `results.json` `runs[].val_loss` and `runs[].train_loss`. Per `prompts/runner.md` §5, WIN if `Δ ≤ −0.01` against BOTH ctrls (and clears the ctrl-to-ctrl variance band); NULL if `|Δ| < 0.01`. Treatment params are ~4.1M (vs ctrl 0.94M) — the ~4× param inflation is the lever's headline confound (K=2 still injects a sizeable param count); the runner's `evidence.md` should flag this when judging.

**Param cost**: K-1 fresh vocab-sized heads = `(K-1) · vocab · d_model` (≈ 3.15M at tiny1m3m with K=2) + mix projection `K · d_model` (128). Net: +3,146,752 params (K=2 was the round-2 reduction from K=4's +9,437,444).
