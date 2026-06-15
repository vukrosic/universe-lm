# Plan — 174 xPos Exponential Decay on RoPE

## Flag
`use_xpos: bool = False` — single boolean added to `LLMConfig` (configs/llm_config.py, in the attention-lever cluster near `use_rov` / `use_moa` / `use_rebased_attn`). Default OFF → baseline path bit-identical. Treatment subclass `Tiny1M3MXPosConfig` sets `use_xpos: bool = True`.

## Change
- `models/layers.py`
  - `MultiHeadAttention.__init__` (already accepts `use_xpos: bool = False`, line 1170): the kwarg, the parameter slot (`self.xpos_gamma = nn.Parameter(torch.zeros(1))` at line 1893, gated on `use_xpos`), and the forward branch at lines 2644-2647 (`K *= exp(-xpos_gamma * t)` after RoPE + GQA repeat + k_gain) are **already in place** from a prior worker (code visible in current file). This implementation pass **leaves them untouched** because the git diff shows they were staged but the treatment config + MinimalLLM plumbing were missing.
  - `TransformerBlock.__init__`: add `use_xpos: bool = False` to the kwarg signature (in the same cluster as `use_rov`, `use_moa`, `use_rebased_attn`). Forward `use_xpos=use_xpos` into the inner `MultiHeadAttention(...)` call.
  - `YOCOLlamaBlock` (models/yoco.py:69) inherits from `TransformerBlock` and passes `*args, **kwargs` straight through — no change needed.
- `configs/llm_config.py`
  - `LLMConfig`: add `use_xpos: bool = False` to the lever cluster (after `use_rebased_attn` / before `mega_beta` at line 793-811).
  - Add `Tiny1M3MXPosConfig(Tiny1M3MConfig)` with `use_xpos: bool = True`.
- `models/llm.py`
  - `MinimalLLM.__init__`: read `self.use_xpos = getattr(config, "use_xpos", False)` (mirrors the `use_rov` / `use_rebased_attn` / `use_moa` pattern). Pass `use_xpos=self.use_xpos` into **both** the `TransformerBlock(...)` call (lower-half path, ~line 644) and the `YOCOLlamaBlock(...)` call (upper-half path, ~line 926).

Step-0 byte-identical: at init `xpos_gamma = 0` ⇒ `g_t = exp(0·t) = 1` for all `t` ⇒ `K = K · 1 = K` ⇒ K magnitude unchanged ⇒ attention scores unchanged ⇒ forward graph bit-identical to the 500k-base RoPE baseline at step 0 (max-abs-diff = 0.0).

## Control
- **Control**: `Tiny1M3MConfig` (`configs/llm_config.py:1926`), val baseline 6.4216 (current `autoresearch/baseline-cache.json` 6.4447 ± 0.0244).
- **Treatment**: `Tiny1M3MXPosConfig` (this idea), only difference `use_xpos=True`.
- **Seed**: **42, single seed** (per protocol — never sweep).
- **Tier**: tiny1m3m, 3M tokens.
- **Bracket**: the daemon owns the two-ctrl baseline bracket (per RUN-CONTRACT); the treatment run is single.

## Cost
- **Params Δ**: +12 scalars (one per MHA at n_layers=12). 0.94M → 0.94M+12 ≈ +0.001%.
- **FLOPs Δ**: one `exp(-γ·t)` + one broadcast multiply per layer on K `[B,T,H,D]`. Trivial (≪1% of attention FLOPs).
- **Memory Δ**: +48 bytes total (12 scalars × 4 bytes fp32). Zero activation memory delta (broadcast, no extra tensor allocation).
- **Wire site**: placed **after** RoPE + GQA repeat + per-head k_gain so the decay broadcasts uniformly over heads (matches the paper's "decay the rotated K" reading).

## Run
- **Tier**: tiny1m3m, seed 42.
- **Entry**: `/Users/vukrosic/my-life/llm-research-kit-scaling/_arq_174-xpos-decay.py` (deterministic self-contained treatment, defines top-level `C = Tiny1M3MXPosConfig`).
- **Command** (via queue daemon):
  ```
  /venv/main/bin/python _arq_174-xpos-decay.py --warmup false \
    --config_class __main__.C
  ```
  Daemon owns baseline bracket separately.
- **Expected wall-clock**: matches baseline tier (~12-15 min on Vast V100).
- **Pass/fail bar** (from idea.md): **NULL** band `|Δ| ≤ 0.01`, **DRIFT** > +0.01, **PASS** ≤ -0.01. The idea's stated Δval band is `[-0.003, -0.020]` — modest because locality is already well-served by 009-FIRE-PE, but a clean signal at the upper end.

## Pass/fail bar
- **PASS** ≤ -0.01 (Δval vs cached baseline).
- **NULL** `|Δ| ≤ 0.01` — locality-prior decay redundant with FIRE / 500k RoPE base at this tier; closes the "exponential decay" axis.
- **DRIFT** > +0.01 — fail loud.
