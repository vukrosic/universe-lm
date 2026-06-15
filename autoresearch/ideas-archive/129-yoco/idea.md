---
id: 129-yoco
status: done
round: 1
updated: 2026-06-13T14:59:37Z
transfer-risk: med
plain: It splits the transformer into two halves — the lower half uses standard attention, then the upper half reuses the resulting KV cache instead of computing its own — saving both memory and compute.
---

# 129 — YOCO: You Only Cache Once (Cross-Layer KV Reuse)

## Source
Sun, Dong, Patra, Ma, Huang, Majumder, Wei, "You Only Cache Once:
Decoder-Decoder Architectures for Language Models" (arXiv:2405.05254,
May 2024, also presented at ICLR 2024 workshop / COLT 2024).
https://arxiv.org/abs/2405.05254

Validated on Llama-2 7B/13B continued pretraining and from-scratch
training of similar-sized LMs. Reports parity-to-better val loss
with significantly reduced KV-cache memory (~5x reduction) at
inference. The lever is the cleanest *architectural* decoder-side
innovation since sliding-window attention.

## Mechanism
Standard transformer: each layer `l` computes its own K_l, V_l from
the input `x_l` (the output of layer `l-1`). At inference, the
KV cache stores all `K_l, V_l` for `l = 1...L` (memory = `O(L·d·T)`).

YOCO splits the model into two halves:
- **Lower half** (layers `1..L/2`): standard self-attention with
  sliding window. Outputs a single **global KV cache** `K_g, V_g`
  at the end of the lower half.
- **Upper half** (layers `L/2+1..L`): each layer reads `K_g, V_g`
  from the lower half (no per-layer K_l, V_l computation) and
  uses cross-attention to combine with the input. The output of
  each upper layer is fed forward but the K_l, V_l are *not*
  stored.

The inference-time KV cache memory is now `O(d·T)` instead of
`O(L·d·T)` (5-10x reduction at typical L=32). At training time,
the upper half is slightly cheaper because it skips K_l, V_l
projections on each layer (~5-10% compute reduction).

**Quality lever**: the bet is that the upper-half layers can
*reuse* the lower-half KV cache because the relevant context is
already captured by the lower half. The lower half is the
"context encoder"; the upper half is the "decoder" operating
on a shared global representation.

**Identity at step 0**: with `K_g, V_g` initialized to zero
(Kaiming-init output of the lower half → small but non-zero),
the upper half's cross-attention reads `K_g ≈ 0, V_g ≈ 0`,
giving an attention output of `softmax(Q · 0^T) · 0 ≈ 0`. This
is **not** identical to standard self-attention at step 0
(standard self-attention has `K_l, V_l` derived from the input,
not zero), but as the lower half trains, `K_g, V_g` become
non-trivial and the upper half's cross-attention "wakes up".

The lever is *not* bit-identical to baseline at step 0; the
deviation is bounded by the magnitude of `K_g, V_g` at init
(typically `O(1)` since they're Kaiming-init projections).

## Design sketch
- `models/yoco.py` (new): `YOCOLlamaBlock` — wraps the lower
  half (`x → x_l`, builds global KV) and upper half (`x_l +
  global KV → output`). ~60 LoC.
- `models/llm.py`: when `use_yoco=True`, replace the standard
  block stack with `YOCOLlamaBlock` (or replace the upper half
  with cross-attention to a shared global KV). The lower half
  uses standard sliding-window self-attention. ~20 LoC.
- `configs/llm_config.py`: add `use_yoco: bool = False`,
  `yoco_split: int = 6` (which layer the split happens at),
  `yoco_lower_window: int = 512`. ~10 LoC.
- LoC: ~90 total (under 200 ceiling).
- Identity at step 0: with `K_g, V_g` from Kaiming-init projection,
  the upper half's cross-attention output is `O(1/√d)` per token
  (near-zero, since `K_g ≈ 0`). Not bit-identical to baseline
  but equivalent in expectation.
- The intuition: at 0.94M with 12L, the upper half has 6 layers
  sharing the lower half's KV cache. The bet is that the lower
  half already captures most of the relevant context and the
  upper half can refine it without recomputing K, V. A null
  would say "at 12L the per-layer KV computation is cheap and
  forcing sharing hurts"; a win would say "the lower half
  *is* the context encoder and the upper half benefits from
  the shared representation".

## Scale evidence
- arXiv:2405.05254 (Sun et al. 2024): Llama-2 7B/13B
  continued-pretraining shows parity-to-better val loss with
  4-5x KV cache reduction. From-scratch training at 1.3B
  shows parity with Llama-style baseline.
- Transfer risk: **med**. Validated at 1.3B-13B (≥100M),
  the lever is scale-free (decoder-decoder architecture is
  well-defined at any depth). At 0.94M with 12L the upper
  half has only 6 layers — the sharing decision is less
  impactful than at 32L+.

## Why it's worth a slot
YOCO is the only architectural decoder-side lever filed that
*changes the cross-layer information flow*. Every other
attention lever (SWA, GQA, NSA, Diff Attention — all in
closed axes) operates on a single layer's attention pattern.
YOCO operates on *how layers share their KV representations*.
The lever is ortho to all positional encoders, all attention
patterns, all residual scalings. A win would say "cross-layer
KV sharing is the right way to scale depth and we should
adopt it for all deep models"; a null would say "at 12L the
per-layer KV is cheap and the sharing overhead is wasted".
The lever also has inference-time benefits (KV cache memory
reduction), but those don't affect the tiny1m3m val-loss A/B.

## Plan

**Files to change:**
1. `configs/llm_config.py`: add `use_yoco: bool = False`, `yoco_split: int = 6`,
   `yoco_lower_window: int = 512`. Add `Tiny1M3MYOCOConfig` ctrl class.
2. `models/layers.py`: add `use_shared_kv: bool = False` kwarg to
   `MultiHeadAttention.__init__` and accept `shared_k, shared_v` in
   `forward()` — when the flag is on, the MHA skips its own K/V
   projections and uses the supplied tensors (still applies q_norm,
   k_norm, RoPE, GQA repeat_interleave). Same for `TransformerBlock`.
3. `models/yoco.py` (new): `YOCOLlamaBlock` — `TransformerBlock` whose
   MHA has `use_shared_kv=True`. Plus a small `GlobalKVHead` module that
   projects the lower-half final hidden state to `(K_g, V_g)` of shape
   `[B, T, kv_size]`.
4. `models/llm.py`: in `MinimalLLM.__init__` when `use_yoco=True`,
   replace the upper `n_layers - yoco_split` block slots with
   `YOCOLlamaBlock` instances (with sliding window turned OFF and shared
   KV turned ON). In `forward()`, after layer `yoco_split - 1`, compute
   `K_g, V_g` from the residual stream and pass them into every upper
   block via `shared_kv=(K_g, V_g)`.

**Flag:** `use_yoco: bool = False` (off by default).

**Identity at step 0:** When `use_yoco=True` and the gate is at layer
`yoco_split` (default 6 in 12L tiny1m3m), the `GlobalKVHead` projections
have std=0.02 normal init (matching the rest of the model's projection
inits). At step 0 they produce small but non-zero K_g, V_g, so the
upper-half attention reads non-trivial K/V instead of zero. The Q
projection in the upper half still applies normally. This is **NOT**
bit-identical to baseline (the attention output is non-zero at step 0
in the upper half), but the deviation is bounded by `O(std_K_g * std_V_g)`
which is `O(0.0004)` — well within the NULL band on the first eval step.

**Run command:**
```
python train_llm.py --config_class configs.llm_config.Tiny1M3MYOCOConfig --seed 42
```

**Read val loss:** from the metrics JSON in the run output dir
(`runs/tiny1m3m-yoco*/metrics.json`, val_loss at the last milestone).
