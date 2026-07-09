---
id: 154-rebased-attn
status: needs-implement
round: 1
updated: 2026-06-15T06:00:30Z
transfer-risk: med
plain: Project the keys and values through a small learned "rebase" matrix so attention can re-mix positions cheaply, like a soft lookup with a learned codebook.
---

# 154 — Rebased Attention (Linear-Complexity K/V Rebasing)

## Source
Shi et al. "Rebased: Linear Attention with Efficient Rebasing for Long Sequences" (arXiv:2407.06641, 2024). Combines linear-attention complexity with the rebasing trick from linear-complexity softmax substitutes.

## Mechanism
Standard attention computes `softmax(QK^T) V` at O(T²). Rebased attention instead:
1. Computes `K' = rebasing_matrix @ K`, `V' = rebasing_matrix @ V` where `rebasing_matrix ∈ R^{T×T}` is a structured low-rank transform.
2. Performs the softmax over the rebased K, V sequences.

The "rebasing" step effectively pools tokens into `R` rebasins (R << T), letting attention read from a learned set of summary positions. With a fixed rebasing matrix (e.g., average-pool by R-block), the operation is O(T·R) and structurally different from NSA / diff-attn (closed) because the rebasing is *before* the attention softmax, not on the attention output.

## Design sketch
- **File**: `models/layers.py` — add a `RebasedAttention` module that pools K, V by a fixed stride-R average-pool *before* the softmax.
- **Config flag**: `use_rebased_attn: bool`, `rebase_stride: int = 8` (default). When True, replaces the standard `Q @ K^T` softmax attention path with `softmax(Q @ K_rebased^T) @ V_rebased`.
- **Step-0 identity**: `K_rebased = avg_pool(K, R)` is *not* identical to `K` at step 0. The byte-identity claim must be made on the *output* of attention — because the rebased softmax uses fewer effective positions, the output is similar but not identical. Use a stride so large that `R == T` (no rebasing) when the toggle is off. The `use_rebased_attn=False` flag keeps the standard path; `use_rebased_attn=True` with `R=T` falls back to full attention. So step-0 of the lever is *full attention with the rebased module plumbed in* — the implementation must guard against this.
- **Intuition**: rebasing reduces noise from rare/uninformative positions by averaging them into a few rebasins. At T=2048 with R=256, each rebasin covers 8 tokens.

## Scale evidence
arXiv:2407.06641 tested up to 1B tokens / GPT-2-medium scale. Transfer risk is **med** (10-100M source scale, but strong theoretical argument).

## Why it's worth a slot
We expect the rebased softmax to act as a soft locality prior — a different lever than the closed NSA/diff-attn axis (which operate on the *output* of attention, not on the input tokens). A null at tiny1m3m would confirm that the rebasing bet is a long-sequence-only lever; a win would be a strong signal the locality-prior is missing in baseline softmax.

## Plan

**Re-code fix (round 1 → 2)**

The previous GPU run was a `SMOKE-FAIL`: the rebase block at the old
`models/layers.py:2101` ran BEFORE the `Q,K,V.transpose(1,2)` at line
2125, so K was in `[B, T, H, d_k]` (pre-transpose) and the reshape
read `K.size(1)` as `T` instead of `H` — arithmetic was valid but the
pool ran along the head axis, not the time axis. **Fix**: moved the
rebase block from its old pre-transpose site to right after the
`Q,K,V.transpose(1,2)` line. Now K, V are `[B, H, T, d_k]` when the
reshape reads `K.size(1) = H` (the head axis), so the avg-pool
correctly compresses along `T` to `R = ceil(T / rebase_stride)`
positions. The pre-transpose site now only initializes
`self._rebase_R = 0` (a no-op at the pre-transpose site, since the
real pool runs post-transpose).

**Files touched**

- **EDIT** `models/layers.py` (~10 LoC net, post-transpose site): the
  rebase block now runs AFTER `Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)`.
  The pad+reshape+mean logic itself is unchanged; the layout context
  it now operates on is `[B, H, T_padded, d_k]`, so `K.size(1)` is
  the head axis and `K.size(-1)` is `d_k` — the mean over the new
  `dim=3` correctly averages over the `R`-sized time blocks. The
  `_rebase_R` flag is read by the manual attention branch at
  `models/layers.py:2523` (which was already written against the
  post-transpose `[B, H, R, d_k]` layout, so it now actually matches).
- The `Tiny1M3MRebasedAttnConfig` dataclass in
  `configs/llm_config.py:4932` is unchanged (`use_rebased_attn=True`,
  `rebase_stride=8`).
- `_arq_154-rebased-attn.py` (already in repo) is unchanged.

**Config flag and zero-init at step 0**

- `use_rebased_attn: bool = False` (default) ⇒ the rebase block's
  `if self.use_rebased_attn:` is False, `_rebase_R` stays `0`, the
  standard softmax path runs at full T-T resolution → bit-identical
  to no-flag baseline. Verified locally: two `MinimalLLM(Tiny1M3MConfig)`
  with `use_rebased_attn=False` produce `max|Δ logits| = 0.0` on the
  same seed and the same input ids.
- `use_rebased_attn: bool = True` with `rebase_stride=8` ⇒ K, V are
  pooled to `[B, H, R=256, d_k=16]` (at T=2048); the manual attention
  branch at `models/layers.py:2523` reads `_rebase_R=256`, computes
  `[B, H, T, R]` scores with a `[T, R]` causal mask, and returns
  `[B, H, T, d_k]`. Verified locally: forward at T=2048 succeeds and
  `_rebase_R=256` on every block.

**Run command** (matches the runner pattern in `_arq_149-ttt-linear.py`):

```
python _arq_154-rebased-attn.py
```

The script sets `argv = ["train_llm.py", "--config_class",
"__main__.C", "--seed", "42", "--dataset_path",
"processed_data/pretrain_1B", "--warmup", "false"]` and calls
`train_llm.main()`. Final val loss is read from the training log
printed by `train_llm.py` (the same path all other `_arq_*` runners
use). Pass/FAIL/NULL band is read against the cached baseline at
`autoresearch/baseline-cache.json` (`baseline.sh verdict <results.json>
<trt_val>`).

**Bet**: at tiny1m3m (T=2048, R=256, ~1.5% of keys), the rebased
softmax acts as a soft locality prior — query `t` only attends to
rebasin `r` whose start `r·8` is ≤ `t`. PASS ≤ ctrl − 0.005 (small
leverage band — soft locality prior at 12L depth). NULL band
`|Δ| < 0.005`. DRIFT > +0.005 (the avg-pool destroys per-token K/V
resolution and the rebased softmax can't recover the lost position
information).

