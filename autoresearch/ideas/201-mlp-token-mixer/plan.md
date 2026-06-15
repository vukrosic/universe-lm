# Plan — 201 mlp-token-mixer

## Flag
- `LLMConfig.use_gmlp_sgu: bool = False` (default OFF, new) — added in `configs/llm_config.py` `LLMConfig`, sibling of `use_v_mix_conv` (163) at line 406.
- `LLMConfig.gmlp_sgu_block_stride: int = 3` (default 3) — apply the SGU to `block_idx ∈ {0, 3, 6, 9}` ⇒ 4 of 12 blocks at tiny1m3m. Sibling of `v_mix_conv_kernel` (163) at line 407.
- `LLMConfig.gmlp_sgu_alpha_init: float = -10.0` — sigmoid init for the α scalar; matches 175-alibi / 188-cross-block-kv-share / 179-anti-causal-subheads discipline.
- `MultiHeadAttention.use_gmlp_sgu`, `MultiHeadAttention.gmlp_sgu_block_stride`, `MultiHeadAttention.gmlp_sgu_alpha_init` — pass-through kwargs on the MHA constructor.
- `MultiHeadAttention.block_idx: int` — NEW kwarg on the MHA constructor. The MHA only allocates `sgu_W` / `sgu_alpha` when `block_idx % stride == 0`. Default value 0 (single-block callers like closure tests stay safe; 0 % 3 == 0 would allocate, but the gating is also on `use_gmlp_sgu` so the default-off path is unchanged).
- `TransformerBlock.use_gmlp_sgu`, `TransformerBlock.gmlp_sgu_block_stride`, `TransformerBlock.gmlp_sgu_alpha_init` — pass-through.
- `TransformerBlock.block_idx: int` — NEW kwarg. Default 0. The block passes its `block_idx` to the inner MHA.
- `MinimalLLM.use_gmlp_sgu`, `MinimalLLM.gmlp_sgu_block_stride`, `MinimalLLM.gmlp_sgu_alpha_init` — pass-through. The LLM build loop now enumerates `range(n_unique)` and passes the index as `block_idx` to each `TransformerBlock`.
- `configs/llm_config.py` — add `Tiny1M3MGMLPSGUConfig(Tiny1M3MConfig)` with `use_gmlp_sgu: bool = True` (other flags at default).

## Change
- `configs/llm_config.py`:
  - In `LLMConfig`, add `use_gmlp_sgu: bool = False` + `gmlp_sgu_block_stride: int = 3` + `gmlp_sgu_alpha_init: float = -10.0` immediately after the `use_v_mix_conv` / `v_mix_conv_kernel` block (line 406-407), with a docstring engaging 163 (local post-attn) and 175 (post-attn pre-O bias) as the sibling slots, and noting that 201 is the *global per-channel* sibling — three axes (local / global / bias) on the same post-attn pre-W_O site.
  - Add `Tiny1M3MGMLPSGUConfig(Tiny1M3MConfig)` with `use_gmlp_sgu: bool = True` and a docstring that includes the **"degenerate gMLP SGU"** naming caveat from review.md (committed shape is `mean(dim=T) → gelu → W_g → broadcast`, not a T×T spatial mix per Liu et al. §3.1).
- `models/layers.py` — `MultiHeadAttention.__init__`:
  - Add `use_gmlp_sgu: bool = False, gmlp_sgu_block_stride: int = 3, gmlp_sgu_alpha_init: float = -10.0, block_idx: int = 0` to the signature, right after the 163-`v_mix_conv` block (line 1038-1039).
  - In the body, set `self.use_gmlp_sgu = use_gmlp_sgu; self.gmlp_sgu_block_stride = max(1, int(gmlp_sgu_block_stride)); self.block_idx = int(block_idx)`.
  - Construction: `self.sgu_W = None; self.sgu_alpha = None` (stubs for attribute-lookup safety). When `use_gmlp_sgu=True` AND `self.block_idx % self.gmlp_sgu_block_stride == 0`:
    - `self.sgu_W = nn.Parameter(torch.empty(d_model, d_model))` then `self.sgu_W.data.normal_(mean=0.0, std=0.02)` — raw `Parameter` + inline `.data.normal_` so RNG state stays aligned with the no-flag path (same discipline as 163's `v_mix_conv_weight` at line 1859-1861).
    - `self.sgu_alpha = nn.Parameter(torch.tensor(float(gmlp_sgu_alpha_init)))` — shape `[]` scalar, init −10 so `sigmoid(−10) ≈ 4.5e-5` ⇒ contribution is `~0` in fp32 at step 0.
  - The total of 4 blocks × 4,096 = 16,384 params (+1.74% of 0.94M) matches the idea's budget.
- `models/layers.py` — `MultiHeadAttention.forward`: insert a new branch in the post-merge / pre-W_O region, immediately AFTER the `use_v_mix_conv` block (around line 4837) and BEFORE the 168 av-output-carry stash:
  ```python
  if self.sgu_W is not None:
      # 201 — Degenerate gMLP SGU on attention output, pre-W_O.
      z = attn_output.mean(dim=1, keepdim=True)            # [B, 1, d_model]
      z = F.gelu(z)
      z = z @ self.sgu_W                                   # [B, 1, d_model]
      z = z.expand(-1, seq_len, -1)
      alpha = torch.sigmoid(self.sgu_alpha)
      attn_output = attn_output + alpha * z
  ```
  Composes multiplicatively with the 163 conv (the conv reads the same `attn_output` input) and additively with the multiplicative scalars (160 head-gain, 168 av-carry, 191 token-attn-gain — all run before or after this branch and the W_O linear sums them).
- `models/layers.py` — `TransformerBlock.__init__`:
  - Add `use_gmlp_sgu: bool = False, gmlp_sgu_block_stride: int = 3, gmlp_sgu_alpha_init: float = -10.0, block_idx: int = 0` to the signature, sibling of the 163-`use_v_mix_conv` kwargs.
  - Pass them through to `MultiHeadAttention(...)` in the `self.attention = MultiHeadAttention(...)` call (around line 5563).
- `models/llm.py`:
  - Add `self.use_gmlp_sgu = getattr(config, "use_gmlp_sgu", False)` + the other two flag attrs (sibling of `self.use_v_mix_conv`).
  - In the `transformer_blocks` list-comprehension (line 1055-1057), convert the `_` placeholder to a real index by enumerating: `for i in range(n_unique)`, then pass `block_idx=i` to each `TransformerBlock(...)`. This keeps `tie_layer_groups=1` and `n_unique = n_layers` semantics intact (block 0..11); with `tie_layer_groups>1` the same index `i` aliases the same block weights, which is fine because the SGU stride at default 3 = 4 blocks (0, 3, 6, 9) is *within* a single tied group when `tie_layer_groups=1` and the tied-block group is unique-by-construction.
  - Pass `use_gmlp_sgu`, `gmlp_sgu_block_stride`, `gmlp_sgu_alpha_init` into the `TransformerBlock(...)` call alongside the existing 163-`use_v_mix_conv` kwargs.
- **Step-0 bit-identity when flag off**: the `self.sgu_W is None` and `self.sgu_alpha is None` stubs mean the branch is never taken, no `nn.Parameter` registered, no `.normal_()` consumed in RNG ⇒ byte-identical to the no-flag baseline. This is the same shape as 163 / 175 / 188.
- **Step-0 bit-identity when flag on**: `α = sigmoid(−10) ≈ 4.5e-5` ⇒ the additive contribution is `~0` in fp32 at step 0 ⇒ the W_O branch reads an `attn_output` that is bit-identical to the no-flag path. The `sgu_W.data.normal_(std=0.02)` init is the only RNG-consuming construction; since it's gated on `block_idx % stride == 0` it consumes RNG only for 4 of 12 blocks, but the no-flag path does NOT consume this RNG (it skips the `sgu_W = nn.Parameter(...)` entirely), so **the on-path RNG state diverges from the off-path at construction time, but the *forward output* at step 0 is still bit-identical** because the SGU contribution is ≈ 0. This is the same construction-time-RNG-divergence-but-forward-bit-identity pattern as 175-alibi (α-gate silence at init) — flagged here for the self-check.

## Control
- **Control**: `Tiny1M3MConfig` — plain tiny1m3m baseline. Cache val reference per `autoresearch/baseline-cache.json`: ~6.4216 on the Vast V100 box.
- **Treatment**: `Tiny1M3MGMLPSGUConfig` — `use_gmlp_sgu=True, gmlp_sgu_block_stride=3, gmlp_sgu_alpha_init=-10.0` (4 of 12 blocks, α init −10).
- **Seed**: 42 (one seed only — never multi-seed, per protocol).
- **Tier**: tiny1m3m (12L × 4H × 64d, 0.94M params, 3M tokens).
- **Schedule**: 201 runs *after* 163 lands. The daemon owns queue ordering, but the plan states the dependency explicitly. If 163 ships null, 201 is the next-in-line "global vs local post-attn" follow-up.

## Cost
- **Params Δ**: 4 blocks × (d_model² + 1) = 4 × (64² + 1) = 4 × 4,097 = 16,388 extra params (+1.74% of 0.94M). The 4 α scalars are negligible. `gmlp_sgu_block_stride` is a config int (no learnable params).
- **FLOPs Δ**: per forward, per SGU-enabled block: one `mean(dim=T)` (B·d_model T-adds) + one `gelu` (B·d_model) + one `d_model × d_model` matmul on a `[B, 1, d_model]` vector (B·d_model² muls) + one broadcast add (B·T·d_model). The d_model² matmul is the dominant term: 4·4² = 4·16 = 64k muls × B = ~512k muls per step at B=8, T=2048. Net Δ < 0.01% of the ~8.6G baseline QK matmul. Memory: one transient `[B, 1, d_model]` vector per SGU-enabled block per step (negligible).
- **Wall-clock Δ**: < 1% slowdown from the SGU branches; well within the ±0.04 noise band.
- **Construction-time RNG**: 4 × `sgu_W.data.normal_()` calls. Diverges from the no-flag path's RNG state (the no-flag path doesn't even construct the Parameter), but the forward output is bit-identical at step 0 because `α ≈ 0` ⇒ additive contribution is numerically 0. (Same pattern as 175-alibi-slopes; flagged here for the self-check.)

## Run
- **Command (on the box)**: `python _arq_201-mlp-token-mixer.py` (the daemon's queue calls this with the args baked into the stub).
- **Tier**: tiny1m3m.
- **Seed**: 42.
- **Expected wall-clock**: ~6-7 min (on par with the ~6 min baseline; the SGU cost is sub-1%).
- **Pass/fail bar** (from `idea.md`):
  - **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
  - **NULL**: `|trt_val − ctrl_val| < 0.01`.
  - **DRIFT**: `trt_val > ctrl_val + 0.01`.
- **Predicted magnitude (per idea.md mechanism)**:
  - **Primary prediction: NULL (|Δval| < 0.01).** Attention's softmax is already a content-based global token mixer at 0.94M; a per-block-stochastic *position-based* (mean-pool ignores content) gate has no clear axis the optimizer can exploit at this scale. 175-alibi, the other post-attn-pre-W_O lever (per-channel positional), already closed WIN at Δ ≈ −0.005; 201 is the per-channel *content-agnostic* sibling and is unlikely to bind tighter than 175.
  - **Long-shot WIN (Δval ∈ [−0.005, −0.015])**: 201's mean-pool + W_g channel-mix is shape-preserving across scales (a per-channel broadcast is the same shape at 0.94M and 135M), so if it binds here it should bind at the ladder. The α-trajectory (4 scalars over training) is the load-bearing diagnostic for "lever bound vs lever didn't bind."
  - **DRIFT risk (Δval ∈ [+0.01, +0.05])**: a sigmoid gate that grows past 1.0 effectively *adds* a per-channel scalar broadcast to every attention head's output, which can interact poorly with the existing 175-alibi gain site. Bounded by the α = sigmoid(·) parameterization (α < 1 always) and the 4-of-12 sparsity (8 of 12 blocks remain at α ≈ 0).
- **Diagnostic**: log α per SGU-enabled block per checkpoint to `evidence.md` as a 4-row table (block_idx, step, α, val_loss). This is the load-bearing readout when val_loss sits inside the noise band — distinguishes "lever didn't bind" (α stays near 0) from "lever bound but in a different axis than val_loss" (α grows but val_loss doesn't follow).
- **Artifact**: `_arq_201-mlp-token-mixer.py` (top-level `C = Tiny1M3MGMLPSGUConfig`) + `autoresearch/ideas/201-mlp-token-mixer/run.json`.
