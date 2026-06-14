---
id: 147-dropkey
status: running
round: 1
updated: 2026-06-14T04:47:07Z
transfer-risk: med
plain: Drop random keys during attention (instead of dropping values or attention scores) to regularize the attention pattern.
---

# 147 — DropKey

## Source
Xu, Zhao, et al. 2022, "DropKey: Towards Efficient and Effective Training of Large Convolutional Neural Networks", arXiv:2207.01058. https://arxiv.org/abs/2207.01058. (Originally a CNN paper, but the mechanism is generic and has been applied to Vision Transformers in the paper itself.)

## Mechanism
For each attention head, during training, randomly drop a fraction `p` of the *keys* (set them to a learned or zero bias) before the QKᵀ dot product. The attention then computes scores over the remaining (un-dropped) keys.
- `K_masked = K * Bernoulli(1 - p)`  (or replace with a learnable bias)
- `attn_scores = Q @ K_maskedᵀ / √d`
- `attn = softmax(attn_scores)`  over the surviving keys
- `output = attn @ V`

The dropped keys are skipped from softmax's denominator (typically via sparse softmax or via the mask-modifying trick). This forces each key to "earn its keep" — the model learns to make surviving keys informative on their own, not just informative *in aggregate*.

## Design sketch (how it works + how to build it)
- Modify the attention forward in `models/layers.py`: after computing K, sample a per-head binary mask `M ~ Bernoulli(1 - p)` of shape `[B, n_heads, T, 1]`, apply `K = K * M`, then run QKᵀ.
- Add `use_drop_key: bool = False`, `drop_key_rate: float = 0.1` to `configs/llm_config.py`. At inference, `drop_key_rate=0` (no masking).
- Identity at step 0: `drop_key_rate=0` → mask is all 1s → attention is standard. Forward output is baseline. ✓
- Why a real lever, not a hyperparam: the *masking target* (keys specifically, not values or scores) is a structural choice with a different inductive bias. Dropout on values (V) is well-known to work. Dropout on attention scores (after QKᵀ) is a different regularizer (sparse attention pattern). DropKey is yet another — it makes the *key representation* robust to random removal, which encourages informative-per-key learning.
- Targets baseline failure: standard attention can be dominated by a few "always-attend-to" keys (the attention sink phenomenon, which we've already closed). DropKey forces the model to have many informative keys, not just one.
- Closest neighbor: 111-DropPath (null at 0.94M, +0.05 wrong-sign). DropPath drops residual branches per-sample. DropKey drops keys per-sample per-head. Different granularity, different mask target. Both are *regularizers* — DropKey's null would close the attention-regularizer axis; its win would tell us the right granularity is per-key, not per-sample.

## Scale evidence
Paper trains ResNets and ViTs on ImageNet; demonstrates gains on both. Not directly validated on language modeling. Transfer risk: med — attention regularizers in our pipeline have been null (111-DropPath), but the mechanism is genuinely different from DropPath.

## Why it's worth a slot
A genuine attention-block regularizer that hasn't been filed. Closest neighbor (111-DropPath) is null, but DropKey operates on a finer granularity (per-head, per-token) and a different axis (key representation vs residual). The attention-sink closed-axis notes that some attention heads are uninformative — DropKey's bet is that forcing per-key informativeness helps. A win would give us a new regularizer family; a null would close the attention-regularizer axis for 0.94M.

## Plan

**Files changed:**
- `configs/llm_config.py` — add `use_drop_key: bool = False`, `drop_key_rate: float = 0.1`.
- `models/llm.py` — read config flags twice (once for `MHALlamaBlock`, once for the decoder block) and pass `use_drop_key=self.use_drop_key, drop_key_rate=self.drop_key_rate` into `MultiHeadAttention(...)` at both construction sites (lines ~519 and ~656).
- `models/layers.py` — `MultiHeadAttention.__init__` adds two kwargs (`use_drop_key`, `drop_key_rate`); stores them on `self`; in `forward`, right after `Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)` (line 1745), apply per-head Bernoulli mask `M ~ Bernoulli(1-p)` of shape `[B, n_heads, T, 1]` and set `K = K * M / (1-p)` (inverted-dropout rescale to keep expected magnitude constant — matches `F.dropout` convention). At eval (`self.training == False`) the mask is identity ⇒ forward graph bit-identical to baseline. The mask is sampled in training mode only.

**Config flag:** `use_drop_key` (off by default). `drop_key_rate=0.1` is the standard ViT regularizer rate; not bit-identical when flag is on (Bernoulli noise is part of the mechanism).

**Identity at step 0 / byte-identical when off:**
- `use_drop_key=False` (default) → `self.use_drop_key` is False → the `if self.use_drop_key and self.training` branch is never taken → K is unmodified → SDPA reads the standard K tensor → forward graph bit-identical to baseline.
- Even when `use_drop_key=True`, `drop_key_rate=0.0` ⇒ Bernoulli(1) always → mask is all 1s → `K = K * 1 / 1 = K` → bit-identical at `drop_key_rate=0`.

**Run command (tiny1m3m, seed 42):**
```
python train_llm.py --config_class tiny1m3m --use_drop_key --drop_key_rate 0.1 --seed 42
```

Baseline (drop_key_rate=0 or flag off):
```
python train_llm.py --config_class tiny1m3m --seed 42
```

**Final val loss:** read from the trainer's final `val_loss=<float>` line printed to stdout at the end of training (the trainer prints the eval loss after the last eval pass, matching the runner convention).

## Pass bar

- Cached tiny1m3m baseline: box `5b8a7fea8963`, `val_mean=6.4394`, `noise_band=0.04` from `autoresearch/baseline-cache.json`.
- WIN iff `trt < 6.3994`; NULL iff `|trt - 6.4394| <= 0.04`; DRIFT iff `trt > 6.4794` or step-0 `max_abs_diff > 1e-3` versus the ctrl.
