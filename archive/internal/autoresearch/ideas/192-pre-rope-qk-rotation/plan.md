# Plan — 192 Pre-RoPE Per-Head Q/K Rotation

## Flag
- `use_pre_rope_rotation: bool = False` (default OFF) on:
  - `configs/llm_config.py:LLMConfig` (base dataclass field, around L1487)
  - `models/layers.py:MultiHeadAttention.__init__` (kwarg, around L1608)
  - `models/layers.py:TransformerBlock.__init__` (pass-through kwarg, around L5889)
  - `models/llm.py:TransformerBlock.__init__` (config read, around L376)
- `pre_rope_rotation_init: float = 0.0` on `LLMConfig` (default 0.0).
- New config subclass: `Tiny1M3MPreRoPEQKRotationConfig(Tiny1M3MConfig)` with
  `use_pre_rope_rotation: bool = True` and `pre_rope_rotation_init: float = 0.0`.

## Change
**File 1 — `models/layers.py` (`MultiHeadAttention.__init__`)**
- Add `use_pre_rope_rotation: bool = False` kwarg.
- In `__init__`, after the 200-`use_per_layer_k_rotation` block, add:
  ```python
  self.use_pre_rope_rotation = use_pre_rope_rotation
  if use_pre_rope_rotation:
      assert self.d_k % 2 == 0, (
          "use_pre_rope_rotation=True requires even d_k "
          f"(got d_k={self.d_k})"
      )
      self.pre_rope_rotation_angles = nn.Parameter(
          torch.zeros(self.n_heads, self.d_k // 2)
      )
  ```
  Param count: `n_heads × d_k//2 = 4 × 8 = 32` per block, **384 params total** (+0.041% of 0.94M — negligible).

**File 2 — `models/layers.py` (`MultiHeadAttention.forward`)**
- Insert the rotation block right after the Q/K/V head reshape (around L4004-4006)
  and BEFORE the qk_norm + RoPE block (around L4055). At this insertion point:
    - Q is `[B, T, n_heads, d_k]`
    - K is `[B, T, n_kv_heads, d_k]` (pre-GQA-repeat)
- Apply per-pair 2D rotation to BOTH Q and K using the
  `pre_rope_rotation_angles` tensor (shape `[n_heads, d_k//2]`):
  - **Q side**: full `[n_heads, d_k//2]` angle grid. Reshape Q to
    `[B, T, H, d_k/2, 2]`, apply cos/sin, reshape back.
  - **K side**: first `[n_kv_heads, d_k//2]` rows of the angle grid (per-
    KV-head projection of the per-head angle grid; at GQA-active configs
    the per-head "second row" of each GQA group is unused). Reshape K to
    `[B, T, n_kv, d_k/2, 2]`, apply cos/sin, reshape back.
- Init `φ=0` ⇒ `cos(0)=1.0, sin(0)=0.0` in fp32 (bit-exact) ⇒
  `Q_a_new = Q_a` and `Q_b_new = Q_b` exactly (same for K) ⇒ Q, K
  bit-identical to the input ⇒ baseline forward is bit-identical when
  the flag is OFF (parameter never registered; the `if self.use_pre_rope_rotation:`
  branch is never taken).

**File 3 — `models/layers.py` (`TransformerBlock.__init__`)**
- Add `use_pre_rope_rotation: bool = False` pass-through kwarg.
- Pass through to the MHA constructor: `use_pre_rope_rotation=use_pre_rope_rotation,`.

**File 4 — `models/llm.py` (`TransformerBlock.__init__` sites)**
- Add `self.use_pre_rope_rotation = getattr(config, "use_pre_rope_rotation", False)`.
- Pass through at BOTH block construction sites (the standard block dispatch
  and the YOCO upper-half block dispatch):
  `use_pre_rope_rotation=self.use_pre_rope_rotation,`.

**File 5 — `configs/llm_config.py`**
- (Already done in prior worker pass.) Base dataclass field exists at L1487-1488
  (`use_pre_rope_rotation: bool = False`, `pre_rope_rotation_init: float = 0.0`).
- New subclass `Tiny1M3MPreRoPEQKRotationConfig` exists at L7953 with
  `use_pre_rope_rotation: bool = True`.

## Control
- **Ctrl**: `Tiny1M3MConfig` (baseline, no flag) — val_mean=6.3988,
  noise_band=0.04, cache `5b8a7fea8963` per `autoresearch/baseline-cache.json`.
- **Trt**: `Tiny1M3MPreRoPEQKRotationConfig` (above) with
  `use_pre_rope_rotation: bool = True`.
- **Seed**: 42 (one seed only — see one-seed-only rule).
- **Tier**: tiny1m3m.
- **Run pair**: 2 ctrl + 1 trt (the daemon owns the ctrl bracket).

## Cost
- **Params**: +384 (+0.041% of 0.94M).
- **FLOPs**: per block per forward: `2 × n_heads × T × d_k × 4` (one cos,
  one sin, two mul-add pairs per `(b, t, h, i)` plane for EACH of Q and K).
  At T=2048, H=4, d_k=16 ⇒ 2 × 4 × 2048 × 16 × 4 ≈ 1.05M flops/block.
  Over 12 blocks × ~1.5K forward calls: ~19G extra flops — well under 0.5% of total model FLOPs.
- **Memory**: one extra `[n_heads, d_k//2] = [4, 8]` fp32 Parameter per
  block × 12 blocks = **384 fp32 values = 1.5KB** (negligible). Two extra
  `[B, T, H, d_k/2, 2]` reshapes on Q and K (no new memory — views).
- **No new optimizer state beyond the param** (AdamW adds 2× momentum/variance
  per param → +768 fp32 values ≈ 3KB).
- **No new dependencies**.

## Run
- **Tier**: tiny1m3m.
- **Seed**: 42 (fixed).
- **Command** (mirrors the `_arq_185-static-per-head-k-rotation.py` shape):
  ```
  python _arq_192-pre-rope-qk-rotation.py
  ```
- **Stub**: `_arq_192-pre-rope-qk-rotation.py` defines
  `class C(Tiny1M3MPreRoPEQKRotationConfig): pass` and the `__main__` block
  drives `train_llm.main()` with
  `argv = ["train_llm.py", "--config_class", "__main__.C", "--seed", "42",
  "--dataset_path", "processed_data/pretrain_1B", "--warmup", "false"]`.
- **Daemon handoff**: `autoresearch/ideas/192-pre-rope-qk-rotation/run.json`
  = `{"name": "192-pre-rope-qk-rotation",
  "arq_file": "_arq_192-pre-rope-qk-rotation.py", "job_timeout": "12m"}`.
- **Pass/fail bar** (copied from `idea.md`):
  - **WIN**: `trt_val ≤ ctrl_val_mean − 0.005` AND clears the two-ctrl rule.
  - **NULL**: `|trt_val − ctrl_val_mean| < 0.01`.
  - **DRIFT**: `trt_val > ctrl_val_mean + 0.01`.
  - Sub-noise is inconclusive per one-seed-only rule.

## Self-check (before release)
1. `use_pre_rope_rotation=False` reproduces the control bit-identically:
   build two `MinimalLLM(Tiny1M3MConfig)` instances with `seed=42` and the
   same input ids, verify `max|Δ logits| = 0.0` (the flag-off path never
   touches Q or K).
2. The treatment path exercises the new code:
   `use_pre_rope_rotation=True` ⇒ `self.pre_rope_rotation_angles` is
   registered (shape `[n_heads, d_k//2]`, init zeros) and the forward block
   runs the per-plane rotation on both Q and K before RoPE.
3. Plan's pass/fail bar matches `idea.md` (WIN/NULL/DRIFT bins verbatim).
4. Run artifact exists and builds: `run.json` + `_arq_192-pre-rope-qk-rotation.py`
   written, stub defines top-level `C`, and `MinimalLLM(C())` constructs on
   CPU without error (the same build-smoke the daemon runs).
5. Step-0 byte-identity verified locally:
   `max_abs_diff(trt_step0_logits, ctrl_step0_logits) == 0.0`
   (the spec-required smoke per `idea.md`).