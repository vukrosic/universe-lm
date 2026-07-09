# Plan — 194 lowrank-ffn (W_V low-rank residual correction)

## Flag

- `use_lowrank_wv: bool = False` — `configs/llm_config.py:155` (added in
  this plan, near the existing `use_lowrank_wo` 207 flag block at
  line 1279).
- `wv_rank: int = 8` — `configs/llm_config.py:160`.
- `wv_lowrank_alpha_init: float = -10.0` — `configs/llm_config.py:161`.

All three default OFF (mirroring `use_lowrank_wo` defaults). The
subclassed `Tiny1M3MLowrankWVConfig` (champion Alibi baseline) flips
`use_lowrank_wv = True`.

## Change

Three files:

1. **`configs/llm_config.py`** — add `use_lowrank_wv`, `wv_rank`,
   `wv_lowrank_alpha_init` fields. Add a new
   `Tiny1M3MLowrankWVConfig(Tiny1M3MAlibiConfig)` dataclass that
   flips the flag on (mirrors the 188/191/204 stack-on pattern).
2. **`models/layers.py`** — in `MultiHeadAttention`:
   - Extend `__init__` signature with `use_lowrank_wv`, `wv_rank`,
     `wv_lowrank_alpha_init` (sits next to the `use_lowrank_wo`
     block at line 1279).
   - In `__init__`, allocate `wv_a ∈ R^{d_model × r}` (normal-init
     std=0.02, matches `qkvo_proj` init), `wv_b ∈ R^{r × d_model}`
     (zero-init ⇒ `wv_a @ wv_b == 0` exactly at step 0), and a
     0-dim `wv_lowrank_alpha` scalar (init `-10.0` ⇒
     `σ(-10) ≈ 4.5e-5`). Mirror the 207 W_O pattern at line 1864.
   - In `forward`, after the QKV split (and after the 188 cross-
     block blend if active — the 188 blend reads `W_V_self` from
     `qkvo_proj` directly and re-projects; we want our correction
     to apply AFTER 188's blend, but 188's `W_V_eff` is already
     detached-with-self so we just need to apply our correction to
     the W_V slice used by the *current* layer's projection). The
     safest, least-surprising site is the same site as 188 — apply
     the W_V correction once at the W_V projection site so the
     identity contract holds. Specifically, when neither YOCO,
     tied-QK, nor MLA is active (the standard `qkv.split` path
     at line 3106), we replace the V slice computation with a
     low-rank-corrected W_V before the F.linear.
   - Concretely: extract
     `W_V = self.qkvo_proj[self.qkv_size - self.kv_size:self.qkv_size]`,
     if `use_lowrank_wv` add `α · (wv_a @ wv_b)` to it, then use
     the corrected W_V in the projection. **Two notes**:
     - YOCO's `use_shared_kv=True` branch (line 3083-3085) reads
       V from `shared_kv` and never projects through W_V. Our
       branch is gated on `not self.use_shared_kv and not
       self.use_tied_qk and not self.use_mla` so it never fires
       in those modes (consistent with 188's behavior).
     - The 188 cross-block-KV-share code at line 3136 reads
       `W_V_self` from `qkvo_proj` *before* our correction. To
       keep 188's stash bit-identical at step 0, we apply our
       W_V correction AFTER 188's stash code (i.e., immediately
       before the F.linear call). At step 0, 188's stash still
       reads the un-corrected W_V from qkvo_proj, and our
       correction is a `σ(-10)·(wv_a @ 0) ≈ 0` no-op, so 188's
       stash is unchanged at step 0.
   - Apply the correction to V as a post-blend, pre-F.linear
     patch: `V = F.linear(x, W_V + α · (wv_a @ wv_b))`. At step 0
     `wv_b = 0` ⇒ `wv_a @ wv_b = 0` ⇒ the addition is a numerical
     no-op ⇒ `V = F.linear(x, W_V)` bit-identical to baseline.
3. **`models/llm.py`** — plumb the new kwargs through
   `TransformerBlock.__init__` and into `MultiHeadAttention(...)`
   (alongside the existing `use_lowrank_wo` plumbing path; since
   207's plumbing is unfinished, we'll add the complete path for
   both `use_lowrank_wo` and `use_lowrank_wv` in the same edit —
   flag-off preserves bit-identity for both).

Step-0 ≈ baseline (when off): the new `wv_a`, `wv_b`,
`wv_lowrank_alpha` Parameters are not allocated (gated on the
flag), no branch is taken in forward, so the forward graph and
weight values are bit-identical to the un-flagged baseline. The
Tiny1M3MAlibiConfig champion control reproduces the cached
baseline 6.4394±0.04 (one seed = 42).

## Control

- **Control**: `Tiny1M3MAlibiConfig` (no flag, val 6.4394, band
  0.04).
- **Treatment**: `Tiny1M3MLowrankWVConfig` (champion + the new
  `use_lowrank_wv=True`).
- **Seed**: 42 (one seed only per the one-seed-only rule).
- **Tier**: `tiny1m3m` (0.94M params, 3M tokens).

## Cost

- **Params**: 2 × (d_model · r + r · d_model) × 12 blocks = 2 ×
  (64·8 + 8·64) × 12 = **12,288 params** (+1.3% of 0.94M), plus
  12 α scalars. Total: 12,300 params, +1.31% of the model.
- **FLOPs**: at step 0, no extra cost (the correction is exactly
  0). After warmup, one `wv_a @ wv_b` matmul per block per
  forward (rank-r d_model²). At r=8, d_model=64, T=2048: 8·64·64
  matmul ⇒ ~0.13M flops/block, ~1.5M flops/step, ~0.05% of the
  base forward FLOPs. Negligible.
- **Memory**: 12,288 floats ≈ 49 KB (fp32) for the params, plus
  one rank-r matrix for the forward. Trivial.

## Run

- **Command** (via the daemon, no ctrl — daemon owns baseline):
  `autoresearch/bin/queue-daemon.sh` runs the stub at
  `_arq_194-lowrank-ffn.py`, which defines `C =
  Tiny1M3MLowrankWVConfig` and calls `train_llm.main()` with
  `--config_class __main__.C --seed 42 --warmup false
  --dataset_path processed_data/pretrain_1B`.
- **Tier**: tiny1m3m, seed 42 (one seed only).
- **Expected wall-clock**: ~3-5 min on V100 (matches the 188/191/
  204 cost profile — three new matrices of size 64×8 per block,
  matmul is a tiny fraction of the d_model²·T attention path).
- **Pass/fail bar** (from `idea.md`):
  - Pre-registered: if `effective_rank(W_V) < 32` → expect
    `Δ < −0.005` (optimizer activates the rank-r path).
  - Pre-registered: if `effective_rank(W_V) ≥ 56` → expect null
    (closes the entire low-rank-residual sub-block family at
    0.94M).
  - **Win criterion**: `Δval < −0.01` (matches the review's PASS
    bar; well outside the 0.04 noise band).
  - **Null criterion**: `|Δval| < 0.04` (within the
    `closed.md:139 170-swiglu-ffn` row's noise band of 0.04).
  - **Drift**: `Δval > +0.04` ⇒ drift, axis rejected.
  - **Sub-noise effect**: `|Δ| ∈ (0.005, 0.04)` is INCONCLUSIVE on
    one seed per the one-seed-only rule. Log and move on.
- **Smoke test (build, no GPU)**: import the stub, instantiate
  `MinimalLLM(C())` on CPU, run a forward pass with a 4-token
  input. Confirms the params register cleanly and the forward
  graph is valid. Also run a step with `use_lowrank_wv=False`
  to confirm the bit-identical-to-champion contract holds.
