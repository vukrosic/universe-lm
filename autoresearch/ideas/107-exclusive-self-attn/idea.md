---
id: 107-exclusive-self-attn
status: needs-run
round: 1
updated: 2026-06-13T06:33:47Z
transfer-risk: low
plain: It tries to make attention stop spending effort on the current word itself so it can use more context.
---

# 107 — Exclusive Self Attention

## Source
Exclusive Self Attention, arXiv:2603.09078 https://arxiv.org/abs/2603.09078

## Mechanism
After standard attention, subtract the component of the attention output that points along the current token's value vector. Implement it as a tiny post-attention projection with a learnable coefficient initialized to zero, so step 0 is the baseline Transformer.

## Scale evidence
The paper reports gains on language modeling runs up to 2.7B parameters trained for 100B tokens, so this is a low-risk transfer bet for a 0.94M model.

## Why it's worth a slot
If the gain comes from removing self-overlap rather than from scale, this should help tiny models too; if not, we learn that attention/FFN separation is already saturated here.

## Plan
**Flag:** `use_exclusive_self_attn` in `configs/llm_config.py`, default `False`.

**Change:** Thread `use_exclusive_self_attn` from `LLMConfig` through `models/llm.py` into `TransformerBlock` and `MultiHeadAttention`. In `models/layers.py`, after the attention matmul, subtract the component of `attn_output` along the current token's value vector. The gate is a zero-init per-head scalar, so `use_exclusive_self_attn=False` is baseline-identical and `True` starts as a no-op.

**Configs:** Control `configs.llm_config.Tiny1M3MConfig`; Treatment `configs.llm_config.Tiny1M3MExclusiveSelfAttnConfig`. Tier `tiny1m3m`, seed `42`.

**Cost:** One scalar per head plus one normalize, one dot product, and one axpy on the attention output. No tokenizer, data, or sequence-length changes.

**Run:** `python train_llm.py --config_class <config_class> --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`

**Bar:** `delta <= -0.01` = WIN; otherwise `|delta| <= 0.005` or wrong sign = NULL.
