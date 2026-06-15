---
id: 184-logit-scale
status: implementing
round: 1
updated: 2026-06-15T07:15:21Z
transfer-risk: low
plain: Multiply all the language-model output logits by one learned scalar (a single dials-up-or-down knob for the model's confidence), starting at 1 so step-0 is byte-identical to the baseline.
---

# 184 — Learned Logit Scale (Global Output Temperature, init=1)

## Source
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, ICML 2021, arXiv:2103.00020) — uses a learned `logit_scale = exp(logit_scale_param)` (with `logit_scale_param` init at `ln(1/0.07) ≈ 2.66`) as a temperature on the contrastive logits. The exp-parameterization guarantees positivity; the learned scalar trades off precision vs recall on the contrastive task.
- TRL (von Werra et al. 2020-, HuggingFace) — uses a learned `logit_scale` in DPO / IPO / SimPO losses for RLHF, with init at `log(2.5)` (Hyperparameters. β in TRL is a scalar temperature on the implicit reward).
- GLObal-LM-head-bias literature (Press et al. 2021 / ?) — T5 used a `lm_head bias` initialized to 0; the bias was added to the logits before softmax. The bias has a different effect (per-token shift) than a global scale (uniform multiplicative).
- In-repo context: closed.md line "logit softcap" closed the softcap-tanh axis (logit softcap is a per-position soft-clipping on the logits before softmax). 184 is a *uniform multiplicative* on the entire logit tensor — different mechanism from softcap (which clips) and from additive bias (which shifts per token). 167-logit-zloss closed null (z-loss regularizer on the log-partition function — a regularizer, not a scale). No prior lever in the repo tests a learned global scale on the LM logits.

## Mechanism
Standard LM head:
```
logits = LM_head(final_residual)         # [B, T, V]
loss = cross_entropy(logits, targets)
```
With learned logit scale:
```
logits = LM_head(final_residual)
logits = logits * logit_scale             # single scalar, broadcast over [B, T, V]
loss = cross_entropy(logits, targets)
```
The scale acts as a *temperature*: `logit_scale = 1.0` is the baseline; `logit_scale > 1` sharpens the output distribution (more confident, higher peak probability on the argmax token); `logit_scale < 1` flattens it (more uniform, lower confidence).

**Parameterization**: `logit_scale = exp(logit_scale_param)`. Init `logit_scale_param = 0` ⇒ `logit_scale = exp(0) = 1` ⇒ `logits * 1 = logits` exactly ⇒ **byte-identical to baseline at step 0** (within fp32 epsilon from the `exp(0)` evaluation, which is `1.0` exactly in IEEE 754). The exp-parameterization guarantees positivity without an explicit clamp.

Alternatively (simpler but unconstrained): a single `nn.Parameter(torch.tensor(1.0))` directly multiplied into the logits. Init `1.0` ⇒ step-0 = baseline exactly. The optimizer can push it positive or negative; if it goes negative, the softmax inverts (worst-token becomes the highest-probability). Use the exp-form for safety.

**Step-0 byte-identity**: with the exp-form and `logit_scale_param = 0` init, `logit_scale = 1.0` exactly (no fp32 epsilon) ⇒ `logits * 1.0 = logits` exactly ⇒ loss, gradient, and predictions are bit-identical to baseline. This is the cleanest possible byte-identity at step 0.

## Design sketch
- **Files**:
  - `models/layers.py` (or `models/llm.py`) — in `MinimalLLM.__init__`, allocate `self.logit_scale_param = nn.Parameter(torch.tensor(0.0))` (init 0 ⇒ `exp(0) = 1`). In `forward`, after `self.lm_head(x)`, apply `logits = logits * self.logit_scale_param.exp()`.
  - `configs/llm_config.py` — add `use_logit_scale: bool = False`. Add `Tiny1M3MLogitScaleConfig(Tiny1M3MConfig)` with `use_logit_scale: bool = True`.
  - `models/llm.py` — thread `use_logit_scale` into `MinimalLLM.__init__` and `forward`.
- **Config flag**: `use_logit_scale: bool = False`.
- **Param count**: **1 scalar param (+0.0001% of 0.94M)**. Negligible.
- **Intuition (why it might lower val loss)**: the standard cross-entropy loss is `−log p(target) = −logit_target + logsumexp(all_logits)`. The gradient on the logits is `softmax(logits) − onehot(target)`. The *magnitude* of the logits (their scale) affects the *sharpness* of the softmax — sharper softmaxes have more confident predictions, which lead to either larger gradients (good if the prediction is correct) or vanishing gradients on incorrect tokens (bad). The optimal logit scale at any training step is a function of (a) the model's current accuracy, (b) the gradient noise level, and (c) the LR schedule. Hard-coding `logit_scale = 1` fixes the temperature to whatever the LM head's weight matrix happens to produce; a learned scale lets the optimizer adjust the temperature to match the training dynamics. At 0.94M with 3M tokens and 92 update steps, the LR schedule has a specific shape (warmup + cosine) and the model's accuracy curve has a specific shape; a learned scale could plausibly match.
- **Why it might bind at 0.94M**: the standard LM head's `W` is initialized to a small magnitude (typical `std = 0.02` for tied embeddings, or `1/sqrt(d_model)` for nn.Linear), so the initial logit magnitudes are `O(1)` and the softmax is moderately sharp. As training progresses, `W` grows and the softmax sharpens. A learned scale could track this growth and provide a more uniform training signal across the 92 update steps. CLIP and TRL use this lever in the multi-million-sample RLHF / contrastive regime; the question is whether the lever binds at 92 update steps in language modeling.

## Scale evidence
- CLIP at 400M image-text pairs (≥100M effective, direct validation for the lever form).
- TRL DPO/IPO/SimPO at 1B-70B SFT models (≥100M, direct).
- The lever itself is *single-scalar* — a single learned parameter. **Transfer-risk: low** (the lever is scale-free, validated at ≥100M for related losses, and the parameter is so cheap that the cost is identical at 0.94M and 7B).

## Why it's worth a slot
The bet, in one sharp sentence: **at 0.94M with 92 update steps, the LM head's logit magnitudes are tied to the tied-embedding matrix and the residual stream's accumulated magnitude — both of which grow during training — so the effective softmax temperature is set by the joint dynamics of the embedding matrix and the residual stream, and a single learned scalar can decouple the temperature from those dynamics and let the optimizer find the right sharpness for cross-entropy at this tier**. A null would close the lever and confirm that the implicit temperature from the tied embedding is the binding constraint (no room to decouple); a win would unlock a 1-parameter temperature lever for the entire pipeline (cheap, no architectural change, just a scalar on the logits).

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: `autoresearch/baseline-cache.json` box `5b8a7fea8963` (RTX 3060), `val_mean = 6.3988`, `noise_band = 0.04`, `n_measurements = 3`. Re-pull on run day.
- **WIN**: `trt_val ≤ ctrl_val_mean − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val_mean| < 0.01`. Most likely outcome per the 167-logit-zloss null (related loss-shape lever also null at 0.94M) and the broader "tiny parameter-count regularizers don't bind at 92 steps" pattern.
- **DRIFT**: `trt_val > ctrl_val_mean + 0.01`. A negative direction (sharpening too much early) could cause the wrong-token softmax to dominate and stall training; a positive direction (flattening too much) could under-train. Both should be caught by the +0.01 bar.
- **Sub-noise is inconclusive** per one-seed-only rule.

## Distinct from closed axes (defensive)
- 167-logit-zloss — additive regularizer on `logsumexp(logits)`. 184 is *multiplicative* on the logits themselves, not an additive regularizer. Different lever axis.
- Closed "logit softcap" axis — softcap is per-position soft-clipping via `tanh`. 184 is uniform multiplicative (a temperature). Different mechanism.
- 170-swiglu-ffn (null), 153-relu2-ffn (null) — FFN-shape levers; 184 is at the output, not the FFN. Different placement.
- 142-layerscale (null) — per-channel diagonal gain on the residual stream. 184 is a single global scalar on the logits, not per-channel. Different placement and shape.
- No prior lever in the repo is a "learned global logit temperature". Fresh axis.

## Plan

- **Champion baseline**: `autoresearch/champion.json` ⇒ `Tiny1M3MAlibiConfig` (`val 6.2403`). The new treatment config `C` **subclasses** `Tiny1M3MAlibiConfig` (so the lever stacks on top of the current champion stack).
- **Files to change**:
  1. `configs/llm_config.py`
     - Add `use_logit_scale: bool = False` to `LLMConfig` (default off; off ⇒ byte-identical).
     - Add `class Tiny1M3MLogitScaleConfig(Tiny1M3MAlibiConfig): use_logit_scale: bool = True`.
  2. `models/llm.py`
     - In `MinimalLLM.__init__` (next to the existing `use_output_temp` block, ~line 1304): read `self.use_logit_scale = getattr(config, "use_logit_scale", False)`. If on, allocate `self.logit_scale_param = nn.Parameter(torch.zeros(()))` (a 0-D scalar; init 0 ⇒ `exp(0) = 1` exactly in IEEE 754 ⇒ exact no-op at step 0).
     - In the logits branch of `forward` (the `compute_logits` / tail of `forward`, right after the existing `use_output_temp` and `use_vocab_bias` hooks, ~line 1846): apply `if self.use_logit_scale: logits = logits * self.logit_scale_param.exp()`. With `use_logit_scale=False`, the if-branch is dead and the path is bit-identical to the champion.
  3. `_arq_184-logit-scale.py` — stub `class C(Tiny1M3MLogitScaleConfig): pass` (matches the 175 stub style). Run via `train_llm.py` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.
- **Param count**: 1 scalar (+0.0001% of 0.94M). 1-D tensor, routes to AdamW under the existing rule.
- **Step-0 byte-identity**: `logit_scale_param = 0` init ⇒ `exp(0) = 1.0` exactly in fp32 ⇒ `logits * 1.0 = logits` exactly ⇒ loss, gradient, predictions bit-identical to the champion baseline. The flag is off by default; with `use_logit_scale=False` the if-branch is never entered and the forward graph is byte-identical to champion.
- **Run command** (runnable, do not necessarily run here):
  ```bash
  cd /root/universe-lm   # adjust to the runner's cwd
  /venv/main/bin/python _arq_184-logit-scale.py
  ```
  (the stub re-`sys.argv`s `train_llm.main()` with `--config_class __main__.C --seed 42 --warmup false`, matching the 175-alibi-slopes stub convention.)
- **Reading the result**: `val_mean` in the runner's print / the `records.jsonl` entry. Compare to champion `val = 6.2403` (cache reference in `autoresearch/baseline-cache.json`, re-pull on run day). Pass: `trt_val ≤ 6.2403 − 0.005` AND clears the two-ctrl rule. Null: `|trt_val − 6.2403| < 0.01`. Drift: `trt_val > 6.2403 + 0.01`. Sub-noise is inconclusive per one-seed-only rule.
