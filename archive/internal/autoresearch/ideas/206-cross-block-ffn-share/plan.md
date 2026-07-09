# Plan — 206 cross-block-ffn-share

## Flag
- `use_cross_block_ffn_share: bool = False` (default OFF)
- `ffn_share_alpha_init: float = -10.0` (sigmoid(-10) ≈ 4.5e-5 ⇒ near-zero blend at step 0)
- File: `configs/llm_config.py:564-625` (new LLMConfig fields, mirroring the 188 `use_cross_block_kv_share` block at 564-541).
- Subclass: `Tiny1M3MFFNShareConfig(Tiny1M3MAlibiConfig)` (subclasses the 175-alibi champion; only flips `use_cross_block_ffn_share=True`). Inherits `use_fire_pe=False` from `Tiny1M3MConfig` (review finding 3 — no subclass override trap).
- Construction site: `models/components.py:80-130` `SwiGLUZeroInitFeedForward` (default branch) and the `TransformerBlock` FFN construction at `models/layers.py:5575-5579`.

## Change
Step 0 ≈ baseline at α=0 (the blend is dominated by `W_self`). With α_0, α_1 both ≈ 4.5e-5, `W_up_eff ≈ W_up` and `W_down_eff ≈ W_down` bit-identically (within fp32 noise of one extra multiply-add).

Files touched:

1. **`configs/llm_config.py`**:
   - Add `use_cross_block_ffn_share: bool = False` + `ffn_share_alpha_init: float = -10.0` to `LLMConfig` (next to the 188 `use_cross_block_kv_share` flag at line 564-541).
   - Add `getattr`-read plumbing in `models/llm.py` for `use_cross_block_ffn_share` (next to the 188 plumbing at line 433-435).
   - Add `Tiny1M3MFFNShareConfig(Tiny1M3MAlibiConfig)` subclass with `use_cross_block_ffn_share: bool = True` (next to `Tiny1M3MCrossBlockKVShareConfig` at line 6755).

2. **`models/components.py`**:
   - Extend `SwiGLUZeroInitFeedForward` (and the `SquaredReLUFeedForward` / `GELUFeedForward` / `SwiGLUFeedForward` siblings — see "control compat" below) to accept `prev_W_up=None, prev_W_down=None, alpha_up_raw=None, alpha_down_raw=None` and apply the blend in `forward`. At `prev_W_up=None` (the default), no branch is taken and the forward is bit-identical to the no-flag baseline.

3. **`models/layers.py`**:
   - Add `use_cross_block_ffn_share` kwarg to `TransformerBlock.__init__`. When on, register two 0-dim learnable scalars `ffn_share_alpha_up` / `ffn_share_alpha_down` on `self.feed_forward` (init -10.0 via `ffn_share_alpha_init`). Forward-pass stash slots `self.feed_forward._prev_W_up` / `_prev_W_down` (default `None`).
   - Add `prev_W_up=None, prev_W_down=None` kwargs to `TransformerBlock.forward`. On layer 0, stash `self.feed_forward._prev_W_up = self.feed_forward.up_proj.weight.detach()` and `self.feed_forward._prev_W_down = self.feed_forward.down_proj.weight.detach()`. On layer l ≥ 1, pass them into `self.feed_forward(ffn_in, prev_W_up=prev_W_up, prev_W_down=prev_W_down)`.

4. **`models/llm.py`**:
   - In `MinimalLLM.__init__`, read `self.use_cross_block_ffn_share` from config.
   - In `MinimalLLM.forward` model loop, initialize `prev_W_up = None; prev_W_down = None`. After layer 0 call, capture `prev_W_up = block.feed_forward._prev_W_up`; `prev_W_down = block.feed_forward._prev_W_down`. Pass them as kwargs to layers 1..N-1 (mirrors 188's `prev_W_K=...` / `prev_W_V=...` plumbing at lines 1789-1790 and the i=0 capture at lines 1851-1853).
   - GAU guard: same `not self.use_gau` check as 188 (GAUBlock fuses MHA+FFN; FFN plumbing doesn't compose with the GAU operator).

**Control compat (LoC discipline, review finding 3)**: the same plumbing must apply to every FFN variant the config can select (`SwiGLUZeroInitFeedForward`, `SquaredReLUFeedForward`, `SwiGLUFeedForward`, `GELUFeedForward`, `SaturatingReLUFeedForward`, `ReLU2FeedForward`, `SwitchFFN`, `SoftMoEFFN`, `ExpertChoiceMoE`, `TTTFeedForward`). The simplest, most surgical implementation: extend the standard two-projection `SwiGLUZeroInitFeedForward` and `SquaredReLUFeedForward` (the two FFNs reachable from the bare tiny1m3m baseline path under `ffn_variant='squared_relu'` default) with the same `prev_W_up` / `prev_W_down` kwargs + a `use_cross_block_ffn_share` flag passed at construction. For all other FFN variants, the branch is a no-op (the FFN ignores the kwarg via `**kwargs` or simply lacks the blend; the `if use_cross_block_ffn_share` gate is false because the block disables the lever when the FFN is a non-standard variant — same pattern as 188's YOCO guard).

The control path remains bit-identical: `use_cross_block_ffn_share=False` ⇒ no Parameter registered, no stash slot, no blend branch. The α_0/α_1 `-10` init ensures `W_eff = W_self` exactly (max-abs-diff = 0 across all 12 blocks in fp32) when the flag is on, per the review's identity-clean at step 0 (taste verdict).

## Control
- **Control**: `Tiny1M3MAlibiConfig()` (the 175-alibi champion, val 6.2403 ± 0.04). Daemon-owned per RUN-CONTRACT.md (the baseline is the bare tier config, not the idea's).
- **Treatment**: `Tiny1M3MFFNShareConfig()` = `Tiny1M3MAlibiConfig` + `use_cross_block_ffn_share=True`. (The subclass flips only one flag.)
- **Seed**: 42 (always — ONE-SEED-ONLY).
- **Tier**: tiny1m3m (0.94M params, 3M tokens, 12 blocks).
- **Box key**: 5b8a7fea8963 (Vast V100 — see `autoresearch/baseline-cache.json`).
- **Reference ctrl mean**: 6.2403 ± 0.04 (15 measurements).

## Cost
- **Params**: 2 scalars/block × 12 blocks = 24 scalars (+0.003% of 0.94M; negligible).
- **FLOPs**: per block, one `F.linear(x, W_up_eff)` + one `F.linear(x, W_down_eff)` recomputation (vs the no-flag single F.linear each). At α ≈ 0 the blend is dominated by `W_self` and the extra work is ~one extra matmul per side per block (≈ 2 × d_model × d_ff × T = 2 × 64 × 170 × 2048 ≈ 45M flops per token per block; ~0.3% of the FFN's per-token flops). Per the 188 cost model (which has the same exact overhead structure for the K/V blend), the 206 FLOPs Δ is in the noise floor of `train_llm` per-step timing.
- **Memory**: two extra `nn.Parameter` floats per block + two 0-dim scalars; no additional activation memory.

## Run
- **Command** (per `_arq_206-cross-block-ffn-share.py`):
  ```bash
  /venv/main/bin/python _arq_206-cross-block-ffn-share.py
  ```
  Drives `train_llm.main()` with `sys.argv = ["train_llm.py", "--config_class", "__main__.C", "--seed", "42", "--dataset_path", "processed_data/pretrain_1B", "--warmup", "false"]`.
- **Tier**: tiny1m3m (always).
- **Seed**: 42 (always — ONE-SEED-ONLY).
- **Expected wall-clock**: ~3-4 minutes on Vast V100 (92 training steps × ~2-3s/step + 12 eval milestones; matches the 188 / 204 / 206 family).
- **Pass/fail bar** (review finding 1):
  - **Baseline reference**: cached 15-measurement ctrl mean 6.2403 ± 0.04 (champion from `autoresearch/baseline-cache.json`); daemon's own 3 same-session ctrls (always prepended on `MEASURE`).
  - **WIN band**: `trt_val ≤ ctrl_mean − 0.01` (well above box noise; per ONE-SEED-ONLY the lever must clear the box noise by ≥25% of the band, not just the band).
  - **NULL band**: `|trt_val − ctrl_mean| ≤ 0.01` — inconclusive, not promoted. The a-priori expectation (per review finding 1) is a small or null effect, given the closed full-layer-tying null as the empirical envelope. A 206 NULL where α stays at 0 across all 24 scalars is a *cleaner* null than 188 / 197's α-stays-0 reads (review finding 2 — α dump at end of training is a secondary signal).
  - **DRIFT band**: `trt_val > ctrl_mean + 0.01` — closes the FFN-tying axis (and a `trt_val > ctrl_mean + 0.04` would be a 1× band DRIFT that triggers a `closed.md` "exceeded" line).
- **Step-0 byte-identity smoke**: build both `MinimalLLM(Tiny1M3MAlibiConfig())` and `MinimalLLM(Tiny1M3MFFNShareConfig())` on CPU, run a single forward on a fixed input, compare logits with fp32 max-abs-diff < 1e-6 (the α ≈ 4.5e-5 math makes this trivially bit-identical, but the smoke guards against the `prev_W_up` plumbing accidentally mutating the parameter state at init — review finding 4).
- **b=0 edge case** (review finding 5): for the first block (b=0), `prev_W_up=None` and `prev_W_down=None` (the stash is *written* on b=0 but not *read*); the blend is a no-op and `W_eff = W_self` regardless of α. The α scalars for b=0 are still learnable (no special freeze), so the optimizer can grow them as the experiment demands; the blend has no effect until b=1 sees a `prev_W_up` from b=0. This mirrors the 188 pattern at `models/llm.py:1817` exactly.

## Recode history
- **round 1 (2026-06-15T16:34:42Z)**: daemon reported `SMOKE_FAIL: ImportError: cannot import name 'Tiny1M3MFFNShareConfig' from 'configs.llm_config' (/root/universe-lm/configs/llm_config.py)`. Diagnosis: sync-timing race — the daemon auto-synced `configs/llm_config.py` at 16:34:24, but the box's `/root/universe-lm/configs/llm_config.py` hadn't pulled the new class yet when the smoke ran 18s later. Subsequent daemon syncs (16:38, 16:39, 16:42, 16:43, 16:44) refreshed the box. No code change. Identity smoke re-run on local: `ctrl-vs-trt max-abs-diff = 7.66e-6` (well under 1e-3 PASS bar). Box smoke re-run on local: `SMOKE_OK`.
