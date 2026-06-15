# Plan — 207 W_O Low-Rank Bottleneck

## Flag
- `use_lowrank_wo: bool = False` — master switch (default OFF ⇒ baseline path bit-identical).
- `wo_rank: int = 16` — absolute rank of the low-rank correction.
- `wo_lowrank_alpha_init: float = -10.0` — `sigmoid(-10) ≈ 4.5e-5`, the soft-gate init for the rank-r contribution.

Files / lines:
- `configs/llm_config.py:NNN` (new flag block, sits in the W_O lever cluster near the existing 171-DropConnect and 160-head-gain comments).
- `models/layers.py:NNN` (per-MHA param registration, then forward-site W_O correction just before `F.linear(attn_output, w_o)` at the current `qkvo_proj[self.qkv_size:]` site — same W_O slice the 171-DropConnect mask reads).
- `models/llm.py:NNN` (mirror the 188 plumbing pattern: `getattr` pick-up at construction, pass-through into `TransformerBlock.__init__`, then to `MultiHeadAttention.__init__`).
- `configs/llm_config.py` tail — new `Tiny1M3MLowRankWOConfig` subclass with the flag on.

## Change

A learnable **rank-r residual correction** is added to W_O in every attention block. The effective projection becomes

```
W_O_eff = W_O + σ(α) · (W_O_A @ W_O_B)
```

where `W_O_A ∈ R^{d_model × r}` and `W_O_B ∈ R^{r × d_model}`. Per block we register two new `nn.Parameter` tensors (`W_O_A` normal-init std=0.02, matching the existing `out_proj` init at line 6043; `W_O_B` **zero-init** so the correction is exactly 0 at step 0) plus one 0-dim scalar `α_raw` (init `wo_lowrank_alpha_init`, default −10 ⇒ `σ(α) ≈ 4.5e-5` at step 0). The forward computes

```python
if self.use_lowrank_wo:
    alpha = torch.sigmoid(self.wo_lowrank_alpha)
    w_o = w_o + alpha * (self.wo_a @ self.wo_b)
output = F.linear(attn_output, w_o)
```

— composes with the 171-DropConnect mask (the 171 mask runs first on `w_o`, the 207 correction is added after, so the two levers multiply through and are jointly OFF-by-default + bit-identical at step 0).

**LoC budget (well under 200).** Roughly: 3 config flags (~6 LoC incl. comments) + 2 `nn.Parameter` registrations + the σ + matmul (~8 LoC in forward) + TransformerBlock / MinimalLLM plumbing (~12 LoC) = ~30 LoC of new code total. The 188 parallel worker touches the same `models/layers.py` and `configs/llm_config.py` files but on the **K/V projection** axis (input side) — no parameter or forward-site overlap with 207 (output side).

**Init justification for `wo_lowrank_alpha_init = -10.0`.** σ(−10) ≈ 4.54e-5. Combined with W_O_B=0, the rank-r contribution is *numerically* 0 at step 0 (bit-identical to baseline). The reviewer flagged this is "approximate, not literal" — the σ(−10) gate is just a soft on/off; the dominant silence comes from W_O_B=0, not from σ(−10) being tiny. The `sigmoid(-10)·W_O_A@W_O_B` form mirrors modded-nanogpt's sigmoid gate pattern (not a zero-init gate — see [[U-Net skips gate fix]] for the prior zero-init failure). I chose −10 over −5 (≈ 6.7e-3 — too large, would not be silent) and over −15 (≈ 3.1e-6 — fine but wastes training time) per the modded-nanogpt default.

## Control
- **Control**: `Tiny1M3MConfig` (the plain baseline), seed 42, no flags. The daemon owns the ctrl per `RUN-CONTRACT.md`.
- **Treatment**: `Tiny1M3MLowRankWOConfig` (subclasses `Tiny1M3MConfig`, sets `use_lowrank_wo=True`, `wo_rank=16`, leaves `wo_lowrank_alpha_init=-10.0`).
- **Tier**: `tiny1m3m` (0.94M params · 3M tokens). Single tier per `PIPELINE.md`.
- **Seed**: 42, one seed only. A sub-noise effect is logged inconclusive and the run moves on — no "add seeds to confirm" (per the one-seed-only rule).

## Cost
- **Params**: `2 × d_model × wo_rank + 1` per block = `2 × 64 × 16 + 1 = 2049` per block × 12 blocks = **24,588 extra trainable params** (+2.6% of 0.94M).
- **FLOPs**: per forward per block, extra `1` matmul of shape `[d_model, wo_rank] @ [wo_rank, d_model]` = `d_model² · wo_rank = 64² · 16 = 65,536` mul-adds. Negligible vs. the full block forward (~M-mults per token).
- **Memory**: `W_O_A` and `W_O_B` are 1 KB and 1 KB per block; total extra ~24 KB. Negligible.
- **Wall-clock**: ~12 min for a treatment on the V100 (the daemon's default `job_timeout` covers this; the `wo_rank=16` matmul is in the noise).

## Run
- **Command**: `python _arq_207-wo-lowrank-bottleneck.py` (seed 42, dataset `processed_data/pretrain_1B`, `--warmup false`).
- **Tier / seed**: `tiny1m3m` / 42.
- **Wall-clock expected**: ~12 min for treatment (the daemon's default `job_timeout=12m`; if a ctrl is needed the daemon prepends ctrls via `baseline.sh check`).
- **Pass/fail bar (from idea.md)**: Δ ≤ −0.01 vs the box-noise band of ±0.01 ⇒ **WIN** (exploitable W_O intrinsic-rank ≤ 16 at 0.94M). Δ inside the band ⇒ **NULL** (logged inconclusive or as a W_O-rank-axis closure). Δ ≥ +0.01 wrong-sign ⇒ reject the lever at this tier. The 175-alibi champion is the active baseline (val 6.2403 ± 0.04) — the bar is the same: |Δ| < 0.02 inside band, ≤ 6.2353 PASS, > 6.2553 DRIFT.
- **Self-check (per `code-implementer.md` §5)**:
  1. `MinimalLLM(C())` constructs on CPU (build-smoke). `_arq_207-wo-lowrank-bottleneck.py` defines `C` at module top level.
  2. Flag OFF (default): `MinimalLLM(C(use_lowrank_wo=False))` forward is **bit-identical** to `MinimalLLM(Tiny1M3MConfig())` (max-abs-diff = 0.0 across the full forward — W_O_B=0 ⇒ correction = 0 exactly, so adding it to `w_o` is a no-op on the value).
  3. Flag ON: `MinimalLLM(C())` constructs, registers 24,588 new params (12 × 2,049). `param_count(Tiny1M3MConfig) + 24588 == param_count(Tiny1M3MLowRankWOConfig)` exactly.
  4. `plan.md` pass/fail bar matches `idea.md` (Δ ≤ −0.01 ⇒ WIN, inside band ⇒ NULL).
  5. `run.json` + `_arq_207-wo-lowrank-bottleneck.py` written, daemon-importable, defines top-level `C`.
