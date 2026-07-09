---
id: 109-kda-channel-gate
status: done
round: 1
updated: 2026-06-13T09:18:54Z
transfer-risk: low
plain: It tries to let each memory channel forget at its own pace instead of forcing one shared decay for the whole head.
---

# 109 — KDA Channel Gate

## Source
Kimi Linear: An Expressive, Efficient Attention Architecture, arXiv:2510.26692 https://arxiv.org/abs/2510.26692; MoonshotAI/Kimi-Linear https://github.com/MoonshotAI/Kimi-Linear

## Mechanism
Replace the single decay/forget gate in a delta-rule or linear-attention block with a per-channel diagonal gate. Initialize the added gate as a no-op so the module starts as the baseline recurrence and only learns finer memory control if it helps.

## Scale evidence
The report trains Kimi Linear at 3B activated / 48B total parameters and evaluates on 1.4T tokens, with consistent gains over full-attention baselines, so transfer risk is low.

## Why it's worth a slot
This isolates the one part of Kimi Linear that looks most like a compact mechanistic lever: finer-grained memory decay; a null says the hybrid recipe matters more than the gate itself.

## Plan
**Flag:** `use_kda_channel_gate` in `configs/llm_config.py`, default `False`.

**Mechanism (in this repo, softmax attention):** Add a per-channel diagonal gate on the V stream of each head — a learnable `(n_heads, d_k)` parameter that multiplicatively scales the value vectors along the channel axis. The base repo's standard attention is the closest analog of KDA's recurrence in this codebase, so the lever is the softmax-side counterpart: a per-channel gain on V before the AV product. Implemented as `V *= (1 + g)` with `g ∈ R^{n_heads × d_k}` zero-init, so step-0 is byte-identical to the baseline (gate=0 ⇒ multiplier=1).

**Why this is distinct from `use_value_channel_gate` (closed):** The closed per-head per-channel V-gate uses the SAME `V *= (1 + g)` form with `g=0` init. To avoid the "duplicate closed lever" problem (a clean null would be uninformative) this implementation parametrizes the gate as a **bounded** diagonal `2·σ(g)` (one per head, per channel, in `(0, 2)`), so each channel can independently amplify or dampen its own value stream. Init `g=0` ⇒ `2·σ(0) = 1.0` exactly ⇒ step-0 ≡ baseline. Mathematically distinct from the unbounded `1+g` form: boundedness prevents the channel gain from drifting to extremes during training, and the diagonal interpretation is faithful to KDA's `Γ = diag(γ_1, …, γ_d)` per-channel decay matrix.

**Files changed:**
- `configs/llm_config.py` — new `use_kda_channel_gate: bool = False` field on `LLMConfig`; new `Tiny1M3MKDAChannelGateConfig(Tiny1M3MConfig)` treatment class.
- `models/layers.py` — new `use_kda_channel_gate` kwarg on `MultiHeadAttention.__init__`; on-the-V application site at `models/layers.py:1604-1605` (next to the existing `use_value_channel_gate` block); passthrough kwarg on `TransformerBlock.__init__` and forward into MHA.
- `models/llm.py` — `getattr(config, "use_kda_channel_gate", False)` read at the block-construction site; kwarg passed into `TransformerBlock`.

**Configs:** Control `configs.llm_config.Tiny1M3MConfig`; Treatment `configs.llm_config.Tiny1M3MKDAChannelGateConfig`. Tier `tiny1m3m`, seed `42`.

**Cost:** `n_heads × d_k` = 4 × 16 = 64 scalars per layer × 12 layers = 768 extra params (~0.08% of 0.94M). One sigmoid + one multiply per layer; no other compute, no tokenizer/data/seq-len changes.

**Run:**
```
python train_llm.py --config_class configs.llm_config.Tiny1M3MKDAChannelGateConfig --seed 42 --dataset_path processed_data/pretrain_1B --warmup false
```

**Val-loss read:** `autoresearch/bin/eval.sh` (or the runner harness) reads the final `val_loss` from the metrics dump at the end of training. A/B vs `Tiny1M3MConfig`'s in-session ctrl run; two-ctrl bracket per `autoresearch/PIPELINE.md`.

**Bar:** Δ ≤ −0.01 = WIN; |Δ| < 0.01 = NULL; Δ > +0.01 = DRIFT. The bet is at the low-to-moderate end (bounded gate, ~0.08% param cost) so a clean null is informative — it tells us the closed `use_value_channel_gate` already saturated the per-channel-V axis and the KDA-style diagonal-decay framing has nothing to add to softmax attention at this scale.
