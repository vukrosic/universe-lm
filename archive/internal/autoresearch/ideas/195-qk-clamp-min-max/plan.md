# Plan — 195 tight hard QK logit clamp (c=2.0, active at step 0)

## Flag
`use_qk_clamp: bool = False` and `qk_clamp_c: float = 2.0`, default OFF, on `LLMConfig` (`configs/llm_config.py:193-194`, alongside `use_qk_rms_scaling`). New `Tiny1M3MQKClampConfig(Tiny1M3MConfig)` subclass at `configs/llm_config.py:6916` with both flags flipped ON. Off ⇒ `if self.use_qk_clamp:` branch never taken ⇒ baseline path bit-identical (the clamp is a pre-softmax score op; off-path the literal `torch.clamp` call never executes).

## Change
- **`configs/llm_config.py`**
  - Add `use_qk_clamp: bool = False` and `qk_clamp_c: float = 2.0` fields on `LLMConfig` (line 193-194).
  - Add `Tiny1M3MQKClampConfig(Tiny1M3MConfig)` subclass with both flipped on (line 6916).
- **`models/layers.py`** — `MultiHeadAttention.__init__`
  - Accept `use_qk_clamp: bool = False` and `qk_clamp_c: float = 2.0` kwargs (line 1108).
  - Store `self.use_qk_clamp = use_qk_clamp` and `self.qk_clamp_c = float(qk_clamp_c)` (line 1962). 0 new params.
  - In `forward`, immediately after the standard `* scale` (or the per-head-temperature multiply) on the `scores = Q·K^T / sqrt(d_k)` step, apply:
    ```python
    if self.use_qk_clamp:
        scores = torch.clamp(scores, min=-self.qk_clamp_c, max=self.qk_clamp_c)
    ```
    This is applied in BOTH the FIRE branch (line 4200) and the main manual branch (line 4376), immediately after the if/else for `use_per_head_temp` and BEFORE any additive transforms (FIRE/CoPE) or downstream score-side multiplies (188 qk_rms_scaling). The clamp binds on the raw scaled QK^T logit surface — clean pre-softmax invariant `|scores| ≤ c`.
  - Add `or self.use_qk_clamp` to the manual-path forcing list at line 4328 (SDPA's flash kernel fuses QK^T+softmax+AV and cannot expose the pre-softmax logit for clamping; same control the 188-qk-rms-scaling and 204-cross-block-score-share levers use).
  - Mirror in the wrapper class (`TransformerBlock.__init__`): accept the kwargs (line ~5163) and pass `use_qk_clamp=use_qk_clamp, qk_clamp_c=qk_clamp_c` to `MultiHeadAttention(...)` (line 5805).
- **`models/llm.py`** — plumb `use_qk_clamp` from config through `MinimalLLM.__init__`'s block-construction call (lines 983, 1357): `use_qk_clamp=getattr(config, "use_qk_clamp", False), qk_clamp_c=getattr(config, "qk_clamp_c", 2.0)`. Default off ⇒ `getattr` defaults to `False` / `2.0` ⇒ the no-flag path is bit-identical to baseline.

## Control
- **Control**: `Tiny1M3MConfig` (val_mean ≈ 6.4216 baseline per the pipeline's cache reference; re-pulled on run day from `autoresearch/baseline-cache.json`).
- **Treatment**: `Tiny1M3MQKClampConfig(Tiny1M3MConfig)` — same arch, `use_qk_clamp=True, qk_clamp_c=2.0`. 0 new params.
- **Tier**: `tiny1m3m` (12L/4H/d_model=64/2M-param model, 3M tokens, 92 update steps).
- **Seed**: 42, one seed only per the one-seed rule. The box-noise band (`±0.01 val`) is the null bound.

## Cost
- **Params**: **0 new params** (the clamp is a fixed config constant `c`, not a learnable parameter). The treatment config has the same parameter count as the control.
- **FLOPs**: 1 element-wise clamp on `[B, H, T, T]` per layer per forward — the compare-and-set is ~B·H·T²·2 FLOPs at tiny1m3m (B=2, H=4, T=2048 ⇒ 32K elements × 2 = 64K FLOPs/layer × 12 layers ≈ 800K FLOPs/forward). End-to-end ≈ +0.001% per step — dwarfed by the QK^T matmul itself (B·H·T²·d_k = 2·4·2048²·16 ≈ 540M FLOPs/layer).
- **Memory**: 0 new params, no extra buffers. `self.qk_clamp_c` is a plain Python float on the module.
- **No new dependencies** (uses the existing `torch.clamp`).

## Run
- **Command** (the daemon reads `run.json` and runs the artifact):
  ```bash
  /venv/main/bin/python _arq_195-qk-clamp-min-max.py
  ```
  The stub re-`sys.argv`s `train_llm.main()` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`, matching the 184/188 stub convention.
- **Tier**: `tiny1m3m`.
- **Seed**: 42, one seed only.
- **Expected wall-clock**: ~6 minutes on the RTX 3060 (3M tokens, 92 update steps, batch=2 — matches the 184/188/175 wall-clock).
- **Pass/fail bar** (copied verbatim from `idea.md`):
  - **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule (cache baseline ≈ 6.4216, champion ≈ 6.24, so 0.005 is real money on this tier).
  - **NULL**: `|trt_val − ctrl_val| < 0.01`. A null closes the *hard-clamp sub-axis* of the logit-bounding family; the soft sub-axis is already closed by the tanh softcap at c=8. Both family endpoints would then be nulled at this tier — informative.
  - **DRIFT**: `trt_val > ctrl_val + 0.01`. Would mean the discontinuous gradient at c=2.0 is too aggressive for 0.94M/12L/4H; closes the *tight-clip* sub-axis with a clear mechanistic reason.
  - **Sub-noise**: per the one-seed-only rule, any `|Δ| ∈ [0.005, 0.01]` is inconclusive — log and move on.
- **Step-0 byte-identity caveat**: at `qk_clamp_c=2.0`, the lever is **NOT** bit-identical to baseline at step 0 (the 2-sigma Gaussian tail of Kaiming-init QK^T entries exceeds |c| ≈ 5% of the time, so the clamp actively fires on ~5% of init logits). This is an explicit, reviewed departure from r0 and is the bet — the regularizer effect is what's being tested, not a null activation. See `idea.md` "Step-0 byte-identity caveat" and `review.md` "Step-0 byte-identity caveat is acknowledged and justified." Off-path (`use_qk_clamp=False`) the `if` branch is never taken and the forward is bit-identical to baseline.