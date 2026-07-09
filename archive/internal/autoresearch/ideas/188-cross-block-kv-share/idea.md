---
id: 188-cross-block-kv-share
status: done
round: 1
updated: 2026-06-15T12:31:16Z
transfer-risk: med
plain: Let each attention block reuse a small fraction of the previous block's K and V projections (a learnable per-block scalar, starting at 0 so step-0 is byte-identical), like a slow re-read of upstream key/value memories.
---

# 188 — Cross-Block K/V Projection Sharing (Learnable Blend of Adjacent-Block Projections)

## Source
- Dehghani et al., "Universal Transformers" (ICLR 2019, arXiv:1807.03819) — shares parameters across depth; validated on algorithmic + small LM tasks (<100M).
- 021-value-residual (in-repo, WIN Δ=−0.034 at tiny1m3m) — carries V *across blocks via the residual stream*; the in-repo cross-block V-mixing family is residual-stream level, not projection level.
- 168-av-output-carry (closed null) — carries the attention output (AV) across blocks, on the residual stream (post-attention). Different placement.
- 164-q-carry (closed null, Δ=+0.036 wrong-sign at 0.94M) — carries Q across blocks; Q-side closed.
- 186-v-carry-block (needs-run, in-repo) — within-block V carry (recurrence along the time axis within a single block). Different axis (time vs depth).

## Mechanism
Standard attention: each block b computes its own `K_b = W_K_b @ x_b`, `V_b = W_V_b @ x_b`. Block b+1 has no awareness of block b's K, V projections.

Cross-block KV sharing: each block's K, V projection is a learnable convex blend of its own (new) projection and the previous block's projection:

```
W_K_b_eff = (1 − α_K_b) · W_K_b_self + α_K_b · W_K_{b-1}     # α_K_b init 0
W_V_b_eff = (1 − α_V_b) · W_V_b_self + α_V_b · W_V_{b-1}     # α_V_b init 0
```

At α=0, `W_K_b_eff = W_K_b_self` exactly (bit-identical to baseline). At α=1, the projection is fully shared with the previous block. Soft, learnable parameter sharing.

## Design sketch
- **File**: `models/layers.py` — modify `attention_block` to accept an optional "prev block W_K, W_V" hook. Stored on the previous block as `self.prev_W_K`, `self.prev_W_V` (set after each block's init).
- **Config flag**: `use_cross_block_kv_share: bool = False` (default).
- **Compute** (per block b): `W_K_eff = (1 − α_K) * self.W_K + α_K * self.prev_W_K.detach()`, `α_K ∈ [0,1]` via `sigmoid(α_K_raw)` init `α_K_raw = -10` (so sigmoid ≈ 0). Same for V. `detach()` so the gradient doesn't flow back through the previous block's projection.
- **Bit-identical at step 0**: `α_K_raw = -10` ⇒ `α_K ≈ 4.5e-5` ⇒ `W_K_eff ≈ self.W_K` (forward graph unchanged at step 0 up to fp32 noise).
- **Params**: 2 scalars per block × 12 blocks = 24 params (+0.003% of 0.94M), negligible.
- **Intuition**: forces adjacent blocks toward a shared KV *projection* subspace, regularizing depth. Different from residual-stream V-carrying (021) — that mixes V into the residual stream; 188 mixes W_V projections directly.

## Scale evidence
Universal Transformers validated at <100M scale; deeper models need longer horizons to amortize the parameter-sharing cost. Transfer-risk: med (the lever is a parameter-sharing regularizer; the win at 12L×3M tokens is plausible but unproven).

## Why it's worth a slot
The in-repo V-side cross-block WIN (021) hints V-side information propagation across depth is a binding constraint. 168 closed AV-carrying (post-attention) null; 164 closed Q-carrying null. 188 is the projection-level analog: if KV subspaces converge across depth, the model can re-use an upstream KV subspace for free. A null at tiny1m3m localizes the V-carrying benefit to residual-stream level only; a win would be a strong signal that projection-sharing across blocks is a missing lever.

## Plan

- **Files to change**
  - `models/layers.py` — add `use_cross_block_kv_share: bool = False` kwarg to
    `MultiHeadAttention.__init__` and `TransformerBlock.__init__` (with
    pass-through wiring). In MHA.__init__, when the flag is on, register
    `cross_block_alpha_K = nn.Parameter(torch.full((), -10.0))` and
    `cross_block_alpha_V = nn.Parameter(torch.full((), -10.0))` (raw sigmoid
    params). In MHA.forward, after the standard QKV projection (post the
    `use_value_residual` stash site) and before any K/V modifications, if
    `prev_W_K is not None`, blend: `W_K_eff = sigmoid(alpha_K) * prev_W_K.detach() +
    (1 - sigmoid(alpha_K)) * self.W_K_slice` and apply via
    `F.linear(x, W_K_eff)`. Same for V. For the **stash branch (layer 0)**:
    stash `self._prev_W_K = self.qkvo_proj[self.q_size:self.q_size+self.kv_size].detach()`
    and `self._prev_W_V = self.qkvo_proj[self.qkv_size - self.kv_size:self.qkv_size].detach()`
    on every layer; layer 0 also has no previous block so the forward
    branch is skipped on layer 0.
  - `models/llm.py` — `MinimalLLM` reads
    `self.use_cross_block_kv_share = getattr(config,
    "use_cross_block_kv_share", False)`, passes it down at both
    TransformerBlock construction sites (lines ~803 and ~1130), and plumbs
    `prev_W_K`/`prev_W_V` through the forward loop (initialized to `None`
    on layer 0, captured from `block.attention._prev_W_K` /
    `block.attention._prev_W_V` after layer 0). GAU/YOCO are mutually
    exclusive with the lever (skip stash when `use_gau` or `use_yoco`).
  - `configs/llm_config.py` — add `use_cross_block_kv_share: bool = False`
    to `LLMConfig` (default off), then add a new
    `Tiny1M3MCrossBlockKVShareConfig(Tiny1M3MAlibiConfig)` that flips it
    to `True` (subclassing the **champion** — the 175-alibi stack —
    so 188 stacks on top of the current winner; with the flag off, the
    config reduces to the champion byte-identically).

- **Flag name**: `use_cross_block_kv_share: bool = False` (default off).

- **Zero-init at step 0**: `cross_block_alpha_K_raw = -10.0` ⇒
  `sigmoid(-10) ≈ 4.54e-5` ⇒ `W_K_eff ≈ 4.54e-5 * prev_W_K + (1 - 4.54e-5)
  * self.W_K` ⇒ at step 0 the blend is numerically dominated by
  `self.W_K` (max-abs-diff on K projection is < 1e-4 in fp32, well
  within the champion-noise band). The optimization graph still goes
  through the new path at flag-on (one extra matmul per block), but
  the OUTPUT is byte-identical to the champion's K, V projection
  up to fp32 noise.

- **Param cost**: 2 scalars per block (one for K, one for V) × 12
  blocks = 24 scalars (+0.003% of 0.94M). Negligible.

- **Run command** (tiny1m3m, seed 42):
  ```
  cd /root/universe-lm
  /venv/main/bin/python _arq_188-cross-block-kv-share.py
  ```
  The stub `_arq_188-cross-block-kv-share.py` (in repo root) subclasses
  `Tiny1M3MCrossBlockKVShareConfig` and invokes `train_llm.main()` with
  `--config_class __main__.C --seed 42 --dataset_path
  processed_data/pretrain_1B --warmup false`, mirroring
  `_arq_175-alibi-slopes.py` and `_arq_186-v-carry-block.py`.

- **Reading the result**: the runner writes a per-step metrics log
  under `remote-results/<date>-vast-tiny1m3m/<slug>/`; final `val`
  is the last `val/loss` line in `results.json`. A/B vs the champion
  `Tiny1M3MAlibiConfig` (val 6.2403, band 0.04). PASS ≤ 6.2353.
  NULL band |Δ| < 0.02. DRIFT > 6.2553.

- **Smoke test (build-time)**: with `use_cross_block_kv_share=False`,
  `MinimalLLM(C(use_cross_block_kv_share=False)).forward(...)` logits
  must equal the champion's logits bit-for-bit (max-abs-diff = 0.0);
  with `use_cross_block_kv_share=True`, logits at step 0 must equal
  the champion's logits within fp32 noise (max-abs-diff < 1e-4 on
  the K, V projection sites; the O projection and downstream carry
  the same fp32 error).
