---
id: 160-rms-gain-per-head
status: running
round: 2
updated: 2026-06-14T05:29:21Z
transfer-risk: low
plain: After the attention block, apply a small per-head RMS-normalized gain on the value output so each head's contribution to the residual stream has controlled magnitude — start with gain=1 so behavior matches the baseline exactly.
---

# 160 — Per-Head RMS Gain on Attention Value Output

## Source
- Gemma 2 (Google DeepMind, 2024, 2B-27B) — uses per-head RMS scaling on the V output before the output projection. Documented in the official Gemma 2 implementation.
- Bai et al. "Qwen2.5" technical report (2024) — similar per-head gain on V output.
- Recent "head-output normalization" literature (2024-2025).

## Mechanism
After computing the attention output `out = softmax(QK^T) V`, multiply by a learnable per-head gain `g_h` (shape `[H]`), broadcast over heads and sequence: `out_h *= g_h`. With `g_h = 1` at init, the gain is a no-op at step 0. ~10 LoC. Mathematically: `RMS(out_h) → g_h * RMS(out_h)`.

## Design sketch
- **File**: `models/layers.py` — add `self.head_gain = nn.Parameter(torch.ones(n_heads))` to the `Attention` module, multiply the V-projected output (or post-attention output reshaped to `[B, H, T, d_head]`) by the gain before the output projection.
- **Config flag**: `use_head_gain: bool` (default False).
- **Step-0 identity**: `head_gain` is initialized to exactly `1.0`, so the multiplied output is byte-identical to baseline at step 0.
- **Intuition**: lets each head control the *magnitude* of its contribution to the residual stream without changing the *direction*. Different from qk_norm (016, normalizes Q/K magnitudes pre-softmax); this normalizes V-output magnitudes post-softmax. Different from LayerScale (closed, per-channel diagonal on residual output); this is per-head on the attention block output specifically.
- **Why now**: qk_norm (016) won at -0.014; the question is whether the *output* magnitude axis is also binding. Per-head V-output gain tests the post-attention magnitude axis vs the pre-attention Q/K axis.

## Scale evidence
Gemma 2 (2B-27B), Qwen 2.5 (0.5B-72B). Transfer risk is **low** (≥100M source scale, multiple production validations).

## Why it's worth a slot
A win would tell us the *post-attention* magnitude axis is the binding constraint at 0.94M (extending qk_norm's win on the pre-attention magnitude axis); a null would close the post-attention magnitude axis.

## Plan
- **Files**: `models/layers.py`, `models/llm.py`, `configs/llm_config.py`.
- **Config flag**: `use_head_gain: bool = False` (default off on `LLMConfig`).
- **Config class**: `Tiny1M3MHeadGainConfig(Tiny1M3MConfig)` with `use_head_gain=True`.
- **MultiHeadAttention** (`models/layers.py`):
  - Add `use_head_gain: bool = False` kwarg.
  - When on, register `self.head_gain = nn.Parameter(torch.ones(self.n_heads))`.
  - In `forward()`, after `attn_output` is produced (post-SDPA or post-manual path, before the [B,H,T,D]→[B,T,d_model] reshape and before the existing `use_attn_output_gate` branch), apply `attn_output = attn_output * self.head_gain.view(1, self.n_heads, 1, 1)`.
- **TransformerBlock** (`models/layers.py`): add `use_head_gain` kwarg and forward to the inner MHA.
- **MinimalLLM** (`models/llm.py`): thread `use_head_gain=getattr(config, "use_head_gain", False)` into the standard `TransformerBlock(...)` constructor call.
- **Step-0 identity**: `head_gain` init = `1.0` exactly. With the flag on, `attn_output * 1 = attn_output` at step 0 — no Parameter is read, no branch is taken when the flag is off (baseline path bit-identical).
- **LoC budget**: ~12 LoC across the three files (well under the 200 LoC cap).
- **Run command** (after code lands + sync to box):
  ```
  cd /root/universe-lm && /venv/main/bin/python train_llm.py --config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false
  ```
  via `_arq_160-rms-gain-per-head.py` (`class C(Tiny1M3MHeadGainConfig): pass`). The runner picks this up from `autoresearch/bin/orchestrate.sh` once the idea is in `needs-run`.
- **Read final val loss** from the JSONL the trainer emits (`val_loss` in the last line). Compare against `Tiny1M3MConfig` baseline val 6.4306 (per `autoresearch/baseline-cache.json`).
- **Prediction**: small wash or marginal win (|Δ| < 0.01) — the *post*-AV magnitude axis is a redundant degree of freedom given the W_O projection that follows, but heads can still learn to attenuate noise. PASS ≤ ctrl − 0.005, NULL band |Δ| < 0.005, DRIFT > +0.005.
