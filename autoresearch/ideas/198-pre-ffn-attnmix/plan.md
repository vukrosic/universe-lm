# Plan — 198 Pre-FFN Attention Mixing

## Flag
- `use_pre_ffn_attn_mix: bool = False` and `pre_ffn_attn_mix_init: float = -10.0` on `LLMConfig` (in `configs/llm_config.py`, right after `use_mish_glu` at line 373, before `use_conv_ffn`). Default OFF ⇒ baseline path bit-identical (no `nn.Parameter` registered, no forward branch taken, `MinimalLLM.__init__`'s `getattr` falls through to `False`). When ON, each `TransformerBlock` registers a 0-dim scalar `pre_ffn_attn_mix_gamma_raw = nn.Parameter(torch.tensor(pre_ffn_attn_mix_init))` and the pre-norm2 path mixes the raw attention output into the FFN input: `ffn_in = norm2(x + sigmoid(γ_raw) · attn_out_raw.detach())`.

## Change
- `configs/llm_config.py`:
  - Add `use_pre_ffn_attn_mix: bool = False` and `pre_ffn_attn_mix_init: float = -10.0` to `LLMConfig` (right after `use_mish_glu`).
  - Add `@dataclass class Tiny1M3MPreFFNAttnMixConfig(Tiny1M3MConfig)` with `use_pre_ffn_attn_mix: bool = True` and `pre_ffn_attn_mix_init: float = -10.0` (subclasses the plain `Tiny1M3MConfig`, no other flags touched). Mirror the docstring style of `Tiny1M3MSwigluFFNConfig` (the closest sibling: A/B vs `Tiny1M3MConfig`, val 6.4216 cached, val 6.3988±0.04 box-baseline).
- `models/layers.py` — `TransformerBlock.__init__`:
  - New kwargs `use_pre_ffn_attn_mix: bool = False`, `pre_ffn_attn_mix_init: float = -10.0` (next to `use_swiglu_ffn` / `use_mish_glu`).
  - When ON, register `self.pre_ffn_attn_mix_gamma_raw = nn.Parameter(torch.tensor(float(pre_ffn_attn_mix_init)))` and `self.use_pre_ffn_attn_mix = True`; else `self.pre_ffn_attn_mix_gamma_raw = None` and `self.use_pre_ffn_attn_mix = False`. Mirror the `ReZero` scalar-parameter registration at lines 6449–6452 (1-line lazy construct, no RNG consumption).
- `models/layers.py` — `TransformerBlock.forward`, pre-norm branch (around line 6936):
  - Capture the **raw** attention output `attn_out_raw = self.attention(self.norm1(x), ve, ...)` **before** any `use_layerscale` / `use_layer_scale` / `use_sub_ln` / rezero / dropout wrapping. This is the natural location: the spec defines `attn_block(x)` as the unmodified attention output, and we want the γ gradient to be cleanly tied to FFN-side loss (per the `.detach()` discipline below).
  - In the pre-norm2 path, replace `ffn_in = self.norm2(x)` with
    ```python
    if self.use_pre_ffn_attn_mix:
        # γ·attn_out_raw.detach(): detached so γ's gradient is
        # cleanly tied to FFN-side loss only (no gradient through
        # the attention path's Q/K/V/O projections at step 0).
        # sigmoid-parameterized → γ∈(0,1) bounded; init γ_raw=−10
        # → γ≈4.5e-5 → step-0 perturbation is ~4.5e-5 in fp32.
        pre_mix = torch.sigmoid(self.pre_ffn_attn_mix_gamma_raw) * attn_out_raw.detach()
        ffn_in = self.norm2(x + pre_mix)
    else:
        ffn_in = self.norm2(x)
    ```
  - **Placement pin (from `review.md` r1):** use placement **(A)** — the mix is added *inside* the norm2 input (so it's renormalized by RMSNorm). This matches the spec's plain reading `ffn_input = attn_residual + γ·attn_block(x).detach()` and the reviewer's `ffn_in = norm2(x + sigmoid(γ) · attn_out.detach())` recommendation. Placement (B) `ffn_in = norm2(x) + sigmoid(γ)·attn_out.detach()` (mix outside RMS) is rejected per the reviewer's analysis: it changes the mix's effective magnitude by `1/RMS(x+mix)`, which is an uncontrolled dynamic — we want the lever to be the optimizer's γ, not an RMS-induced rescale.
  - Scope: **pre-norm path only.** The post-norm and parallel-block paths are alternative architectures (off by default; not the baseline); adding the mix to those paths is out of scope for this A/B. A user combining `use_pre_ffn_attn_mix=True` with `use_post_norm=True` or `use_parallel_block=True` would have the lever silently shadowed on those paths — documented in the flag docstring.
- `models/llm.py` — `MinimalLLM.__init__`:
  - Read `self.use_pre_ffn_attn_mix = getattr(config, "use_pre_ffn_attn_mix", False)` and `self.pre_ffn_attn_mix_init = float(getattr(config, "pre_ffn_attn_mix_init", -10.0))` (place next to `use_mish_glu` at line 588).
  - Pass both kwargs to the standard `TransformerBlock` constructor at the second site (line 1381+, alongside `use_swiglu_ffn`/`use_mish_glu`) **and** the YOCO upper-half block constructor at the first site (line 994+, alongside `use_swiglu_ffn`/`use_mish_glu`). Both sites mirror.
- Step-0 bit-identity: `sigmoid(−10) ≈ 4.5400e-5` ⇒ `pre_mix ≈ 4.5400e-5 · attn_out_raw` ⇒ at init the perturbation to `x` is on the order of `4.5e-5 · O(1) ≈ 4.5e-5` in fp32. The expected `max|x+pre_mix − x|` is `~4.5e-5`, well below the `1e-5` reviewer-tightened fp32 noise target **when** the `attn_out` tensor's elementwise magnitude is `O(1)`. The reviewer noted this is *fp32-noise bit-identical* not literally bit-identical — same convention as 188/206/201/205. **Self-check at plan-time**: confirm the cached baseline `val_mean=6.3988±0.04` reproduces within band when the flag is off (the daemon owns the baseline; my plan-time check is to construct the model on CPU and verify `torch.allclose` between the `use_pre_ffn_attn_mix=True` first-forward and the no-flag first-forward with a tolerance of `atol=1e-3, rtol=1e-3` — both should produce O(4.5e-5)-magnitude deltas on `ffn_in`).

## Control
- **Control (this run's ctrl)**: `Tiny1M3MConfig` (the plain baseline, cached val_mean `6.3988±0.04` per `autoresearch/baseline-cache.json` box `5b8a7fea8963`, prior `val_runs=[6.4112, 6.3934, 6.3919]`, n=3). The daemon owns the baseline.
- **Treatment**: `Tiny1M3MPreFFNAttnMixConfig` (the new `use_pre_ffn_attn_mix: bool = True` config, subclasses `Tiny1M3MConfig` with no other flags touched).
- **Seed**: 42 (one seed only — `feedback-one-seed-only`).
- **Tier**: tiny1m3m (0.94M params, 3M tok, single seed 42).

## Cost
- **Params**: +12 scalars (1 per block × 12 blocks), +0.0013% of 0.94M. Negligible.
- **FLOPs**: +1 elementwise multiply-add (sigmoid + multiply) per block per forward on a `[B, T, d_model]` tensor. At tiny1m3m `B=2, T=2048, d_model=64, 12 blocks`, that's `2·2048·64·12 = 3,145,728` fp32 ops/forward — negligible (<0.01% of total forward FLOPs).
- **Memory**: 12 floats total. Negligible.
- **Compile path**: no change. The lever is a tensor add before `norm2`, fully compatible with the existing pre-norm forward graph; no manual attention path forcing needed.

## Run
- Command (GPU box): `_arq_198-pre-ffn-attnmix.py` with seed 42 and `--warmup false` (the daemon's `_box_smoke.py` runs the build on CPU first; the actual GPU run is launched by `queue-daemon.sh` reading `autoresearch/ideas/198-pre-ffn-attnmix/run.json`).
- Tier: tiny1m3m, seed 42. Expected wall-clock: ≤12m (`job_timeout=12m` in `run.json`).
- Val read: from `autoresearch/remote-results/<run-dir>/trt_*.log` — search for the final `val` line emitted by `train.py`.
- Champion reference: `Tiny1M3MConfig` (the baseline, val_mean `6.3988±0.04` per `autoresearch/baseline-cache.json`).
- Pass/fail bar (from `idea.md` / reviewer):
  - **WIN**: `trt_val ≤ 6.3988 − 0.01 = 6.3888` AND clears the two-ctrl rule (Δ clears the noise band `±0.04`).
  - **NULL**: `|trt_val − 6.3988| < 0.01` (i.e., the FFN's residual-stream input is sufficient; the intra-block attention-to-FFN mix doesn't bind).
  - **DRIFT**: `trt_val > 6.3988 + 0.01 = 6.4088` (γ opens and the FFN's input gets a noisy perturbation that hurts the loss).
- Diagnostic when NULL: log `pre_ffn_attn_mix_gamma_raw` per block per checkpoint to `evidence.md` as a 12-row table — the load-bearing diagnostic for whether γ is *opening at all* during training (a γ frozen at -10 is the same lever being SILENT; a γ growing toward 0 is the lever being TRIED and found useless).
