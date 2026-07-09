# Plan — 185 Static Per-Head K-Rotation

## Flag
- `use_static_k_rotation: bool = False` (default OFF) on:
  - `configs/llm_config.py:LLMConfig` (base dataclass field, after `use_rebased_attn: bool = False` block at L905-906, see §"Files" below for exact line)
  - `models/layers.py:MultiHeadAttention.__init__` (kwarg `use_static_k_rotation: bool = False`)
  - `models/layers.py:TransformerBlock.__init__` (kwarg `use_static_k_rotation: bool = False`)
- New config subclass: `Tiny1M3MStaticKRotationConfig(Tiny1M3MConfig)` with `use_static_k_rotation: bool = True`.

## Change
**File 1 — `models/layers.py` (`MultiHeadAttention`)**
- Add `use_static_k_rotation: bool = False` kwarg after `use_rebased_attn` (L1209).
- In `__init__`, after the rebase block (around L2128, after `self.use_rebased_attn = use_rebased_attn`), add:
  ```python
  # 185 — Static per-head K-rotation (learned orthogonal rebase of K).
  # Self.use_static_k_rotation = use_static_k_rotation
  # if self.use_static_k_rotation:
  #     self.k_rotation_angles = nn.Parameter(
  #         torch.zeros(self.n_heads, self.d_k // 2)
  #     )
  ```
  Param count: `n_heads × d_k//2 = 4 × 8 = 32` per block, **384 params total** (+0.041% of 0.94M — negligible).
- In `forward`, apply the rotation to K right BEFORE the GQA `repeat_interleave` (L3033-3036), i.e., while K is in `[B, T, n_kv_heads, d_k]` layout. To keep the spec's per-head axis clean with GQA, we apply after repeat so the rotation broadcasts per head. Concretely:
  - Place the rotation block after the GQA repeat (L3033-3036) and before `use_k_gain` (L3037), so K is `[B, T, n_heads, d_k]`:
    ```python
    if self.use_static_k_rotation:
        # angles: [n_heads, d_k//2]. Build block-diagonal R_h ∈ R^{d_k × d_k}
        # per head (a product of d_k//2 2D rotations on disjoint planes).
        cos_a = self.k_rotation_angles.cos()  # [H, d_k/2]
        sin_a = self.k_rotation_angles.sin()  # [H, d_k/2]
        # Reshape K to [B, T, H, d_k/2, 2] for paired (2i, 2i+1) planes.
        K_pairs = K.reshape(
            batch_size, seq_len, self.n_heads, self.d_k // 2, 2
        )
        K_a = K_pairs[..., 0]  # [B, T, H, d_k/2]
        K_b = K_pairs[..., 1]  # [B, T, H, d_k/2]
        # R_h^i @ (K_a, K_b) on each plane i:
        K_a_new = K_a * cos_a.view(1, 1, self.n_heads, self.d_k // 2) \
                - K_b * sin_a.view(1, 1, self.n_heads, self.d_k // 2)
        K_b_new = K_a * sin_a.view(1, 1, self.n_heads, self.d_k // 2) \
                + K_b * cos_a.view(1, 1, self.n_heads, self.d_k // 2)
        K = torch.stack([K_a_new, K_b_new], dim=-1).reshape(
            batch_size, seq_len, self.n_heads, self.d_k
        )
    ```
  - **Why after repeat**: at tiny1m3m, `n_kv_heads=2 < n_heads=4`, so a `[n_heads, d_k//2]` parameter is the natural per-head spec from `idea.md`. After `repeat_interleave`, K is `[B, T, n_heads, d_k]` and the rotation is genuinely per-head. (Placing it pre-repeat would force a `[n_kv_heads, d_k//2]` parameter — slightly weaker lever, and the per-head axis the idea asks for collapses to per-KV-head under GQA.)
  - **Why before RoPE**: K rotation is *position-independent* and *basis-changing*, the same family as 154's fixed rebase. RoPE is applied later in the pipeline (L2950 onwards), but 185's spec puts the rotation pre-RoPE/pre-qk_norm as a basis change on the raw K (after QKV split, after GQA repeat, before RoPE/qk_norm). The orthogonal `R_h` preserves norms and inner products, so this commutes cleanly with subsequent normalization/rotation — at init `θ=0` ⇒ `R=I` ⇒ all ops are bit-identical.
  - **Forces the manual attention path**? No — the rotation is on K only, doesn't change scores' shape, doesn't touch the mask. SDPA still works (only the K input differs). No branch change in the manual-attention gate.
  - **Step-0 byte-identity**: `θ=0` ⇒ `cos(0)=1.0`, `sin(0)=0.0` in fp32 (bit-exact) ⇒ `K_a_new = K_a*1 - K_b*0 = K_a` and `K_b_new = K_a*0 + K_b*1 = K_b` exactly ⇒ K is bit-identical to the input. The QK^T is unchanged, softmax is unchanged, attention output is unchanged. Baseline path is bit-identical when the flag is OFF (parameter never registered; the `if self.use_static_k_rotation:` branch is never taken).

**File 2 — `models/layers.py` (`TransformerBlock`)**
- Add `use_static_k_rotation: bool = False` kwarg (after `use_mqa_gated`/around L4695, in the existing batch).
- Pass through to the MHA constructor (after `use_mqa_gated=use_mqa_gated` block at L4751): `use_static_k_rotation=use_static_k_rotation,`.

**File 3 — `models/llm.py` (`TransformerBlock` sites — there are two: the standard block dispatch at L951 and the YOCO upper-half block dispatch around L640)**
- Add `self.use_static_k_rotation = getattr(config, "use_static_k_rotation", False)` in `__init__` (around the existing batch at L313-L320 in `TransformerBlock.__init__`).
- Pass through at BOTH block construction sites: `use_static_k_rotation=self.use_static_k_rotation,`.

**File 4 — `configs/llm_config.py`**
- Add `use_static_k_rotation: bool = False` to the base `LLMConfig` dataclass, immediately after `use_rebased_attn: bool = False` at L905-906.
- Append a new subclass at the end of the file (after `Tiny1M3MRebasedAttnConfig` at L5821, in the existing family):
  ```python
  @dataclass
  class Tiny1M3MStaticKRotationConfig(Tiny1M3MConfig):
      """Tiny1M3M with static per-head learned K-rotation.

      A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`,
      val 6.3988 cached at `5b8a7fea8963` ±0.04, see
      `autoresearch/baseline-cache.json`).

      Per-head learnable orthogonal matrix `R_h ∈ R^{d_k × d_k}`
      applied to K as `K = R_h @ K` pre-RoPE / pre-qk_norm. Each
      `R_h` is a product of `d_k/2 = 8` 2D rotations on disjoint
      `(2i, 2i+1)` planes, parametrized by `n_heads × d_k/2 = 32`
      angles per block. Init `θ_{h,i} = 0` ⇒ `R_h = I_{d_k}`
      exactly in fp32 ⇒ `K = R_h @ K = K` exactly ⇒ step-0
      forward is bit-identical to the no-flag baseline.

      `R_h` orthogonal ⇒ `||R_h v|| = ||v||` and `<R_h q, R_h k>
      = <q, k>` ⇒ QK^T magnitudes are preserved (no softmax
      temperature shift) — same "preserve the dot product"
      property RoPE has for its position rotation and 154's
      fixed orthogonal rebase has. Default off ⇒ no Parameter
      registered, no branch taken, baseline path bit-identical.

      Param cost: `n_heads × d_k/2 × n_layers = 4 × 8 × 12 = 384`
      params (+0.041% of 0.94M — negligible).

      Distinct from:
        - 154-rebased-attn (WIN, fixed *random shared* rebase of K
          and V with the same matrix): 185 is *learned, per-head,
          K-only*.
        - 172-per-head-rope-base (closed null, position-dependent
          per-head RoPE base): 185 is *position-independent*.
        - 176-v-pre-av-norm (closed null, V-norm pre-AV):
          different tensor, different op.
        - 180-qk-logit-conv (rejected, pre-softmax QK^T smoothing):
          different op.
        - 152/155/160/166 per-head scalar family (closed): 185 is
          per-head *matrices* on K, not scalars on scores.

      PASS ≤ ctrl − 0.005 AND clears the two-ctrl rule. NULL
      band |Δ| < 0.01. DRIFT > +0.01. See
      `autoresearch/ideas/185-static-per-head-k-rotation/idea.md`.
      """
      use_static_k_rotation: bool = True
  ```

## Control
- **Ctrl**: `Tiny1M3MConfig` (baseline, no flag) — val_mean=6.3988, noise_band=0.04, cache `5b8a7fea8963` per `autoresearch/baseline-cache.json`.
- **Trt**: `Tiny1M3MStaticKRotationConfig` (above) with `use_static_k_rotation: bool = True`.
- **Seed**: 42 (one seed only — see one-seed-only rule).
- **Tier**: tiny1m3m.
- **Run pair**: 2 ctrl + 1 trt (the daemon owns the ctrl bracket).

## Cost
- **Params**: +384 (+0.041% of 0.94M).
- **FLOPs**: per block per forward: `n_heads × T × d_k × 4` (one cos, one sin, two mul-add pairs per `(b, t, h, i)` plane) ≈ 4 × 2048 × 16 × 4 ≈ 524k flops/block at T=2048. Over 12 blocks × ~1.5K forward calls (eval_milestones + train steps): ~10G extra flops — well under 0.5% of total model FLOPs.
- **Memory**: one extra `[n_heads, d_k//2] = [4, 8]` fp32 Parameter per block × 12 blocks = **384 fp32 values = 1.5KB** (negligible). One extra `[B, T, H, d_k/2, 2]` reshape (no new memory — view).
- **No new optimizer state beyond the param** (AdamW adds 2× momentum/variance per param → +768 fp32 values ≈ 3KB).
- **No new dependencies**.

## Run
- **Tier**: tiny1m3m.
- **Seed**: 42 (fixed).
- **Command** (mirrors the `_arq_154-rebased-attn.py` shape):
  ```
  python _arq_185-static-per-head-k-rotation.py
  ```
- **Stub**: `_arq_185-static-per-head-k-rotation.py` defines `class C(Tiny1M3MStaticKRotationConfig): pass` and the `__main__` block drives `train_llm.main()` with `argv = ["train_llm.py", "--config_class", "__main__.C", "--seed", "42", "--dataset_path", "processed_data/pretrain_1B", "--warmup", "false"]`.
- **Daemon handoff**: `autoresearch/ideas/185-static-per-head-k-rotation/run.json` = `{"name": "185-static-per-head-k-rotation", "arq_file": "_arq_185-static-per-head-k-rotation.py", "job_timeout": "12m"}`.
- **Pass/fail bar** (copied from `idea.md`):
  - **WIN**: `trt_val ≤ ctrl_val_mean − 0.005` AND clears the two-ctrl rule.
  - **NULL**: `|trt_val − ctrl_val_mean| < 0.01`.
  - **DRIFT**: `trt_val > ctrl_val_mean + 0.01`.
  - Sub-noise is inconclusive per one-seed-only rule.

## Self-check (before release)
1. `use_static_k_rotation=False` reproduces the control bit-identically: build two `MinimalLLM(Tiny1M3MConfig)` instances with `seed=42` and the same input ids, verify `max|Δ logits| = 0.0` (the flag-off path never touches K).
2. The treatment path exercises the new code: `use_static_k_rotation=True` ⇒ `self.k_rotation_angles` is registered (shape `[n_heads, d_k//2]`, init zeros) and the forward block runs the per-plane rotation.
3. Plan's pass/fail bar matches `idea.md` (WIN/NULL/DRIFT bins verbatim).
4. Run artifact exists and builds: `run.json` + `_arq_185-static-per-head-k-rotation.py` written, stub defines top-level `C`, and `MinimalLLM(C())` constructs on CPU without error (the same build-smoke the daemon runs).
5. Step-0 byte-identity verified locally: `max_abs_diff(trt_step0_logits, ctrl_step0_logits) == 0.0` (the spec-required smoke per `idea.md`).