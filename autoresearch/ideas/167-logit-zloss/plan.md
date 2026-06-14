# Plan — 167 logit-zloss

## Flag
- `use_z_loss: bool = False` (default off) and
  `z_loss_lambda: float = 0.0` (default 0.0) on `LLMConfig`.
  - Fields: `configs/llm_config.py` (new block, sibling of `use_value_residual` /
    `use_av_output_carry` in the residual-stream / output-head group).
  - Wiring already present in `training/trainer.py:1464-1465, 1548-1552,
    1654-1658, 1599, 1702, 1706` — the trainer reads both via
    `getattr(config, ..., 0.0 / False)` so adding the LLMConfig fields
    with defaults `False` / `0.0` is drop-in compatible (no other
    config has to change). The `use_z_loss` AND `z_loss_lambda > 0`
    guard means the term is exactly 0 whenever either is off —
    baseline path bit-identical.
- A/B subclass: `Tiny1M3MZLossConfig(Tiny1M3MConfig)` (newly added at end
  of `configs/llm_config.py`), `@dataclass`-decorated with
  `use_z_loss: bool = True` and `z_loss_lambda: float = 1e-4`.
  Imported directly from `configs.llm_config` by
  `_arq_167-logit-zloss.py` — same `C`-import pattern as 162/165
  (avoids the dataclass-inheritance pitfall that breaks bare
  `class C(Tiny1M3MConfig): use_z_loss: bool = True`).

## Change
- `configs/llm_config.py` — added `use_z_loss: bool = False` and
  `z_loss_lambda: float = 0.0` to `LLMConfig` with a doc block
  documenting the PaLM-style mechanism, the `logsumexp` math, the
  baseline-bit-identical guarantee, and the train-only contract.
  Added `Tiny1M3MZLossConfig` at the file's tail, `@dataclass`-
  decorated, `use_z_loss=True`, `z_loss_lambda=1e-4`. This is the
  class the daemon imports as `C` for the build-smoke and run.
- `training/trainer.py` — unchanged. The trainer already has the
  full z-loss wiring at lines 1464-1465 (config reads), 1548-1552
  (AMP branch), 1654-1658 (CPU branch), 1599 (AMP accumulation),
  1702 (CPU accumulation), 1706 (per-step `z_loss_val` logging).
  Both the `if use_z_loss / else logits.new_zeros(())` branches
  produce a 0-dim scalar — the `loss = ce_loss + ... + z_loss + ...`
  accumulation is unchanged in shape when the flag is off or on.
- `models/layers.py` — unchanged. Z-loss operates purely on the
  output logits produced by the existing head; no model-side
  wiring needed.
- `models/llm.py` — unchanged. Same as above.

Step-0 identity (flag OFF): `use_z_loss = False` or `z_loss_lambda = 0.0`
⇒ the `else logits.new_zeros(())` branch fires ⇒ `z_loss = 0`
exactly ⇒ `loss = ce_loss + 0 + ...` is bit-identical to the
no-z-loss baseline. `z_loss_val = 0.0` per step.

Step-0 identity (flag ON at λ=1e-4): the trainer reads the
`logits` produced by the standard head, computes
`z = logits.logsumexp(dim=-1).pow(2).mean()`, and adds
`1e-4 · z` to the loss. At init with ~N(0, σ²) logits and
vocab_size=49152, `log(Z) ≈ 5.25`, `z ≈ 27.5`,
`λ·z ≈ 2.75e-3` — a small but non-zero auxiliary loss. Train-only;
eval stays plain CE.

## Control
- A: `configs.llm_config.Tiny1M3MConfig` (seed 42, flag OFF) — bare
  tier config. The daemon owns this control via
  `autoresearch/bin/baseline.sh`.
- B: `_arq_167-logit-zloss.py` (seed 42, flag ON) — same tier,
  `use_z_loss=True`, `z_loss_lambda=1e-4`. The `C` class is the
  build-smoke target.
- Tier: `tiny1m3m` (0.94M params, 3M tokens). Seed 42 only
  (one-seed-only rule).

## Cost
- Params: 0 (no new parameters — z-loss is computed from existing
  logits, the only new "weight" is the per-batch constant λ).
- FLOPs: 1·vocab_size·B·T elementwise exp/log reductions per step
  (`logsumexp` along dim=-1) — ~V=49152 multiplies + adds per
  token per step, ~negligible at tiny1m3m where the dominant cost
  is the attention matmul and FFN.
- Memory: 1·B·T float buffer for `z = logsumexp(...)` — a few
  hundred floats; activation memory unchanged.

## Run
- Artifact: `_arq_167-logit-zloss.py` (repo root) imports
  `Tiny1M3MZLossConfig as C` from `configs.llm_config` and
  dispatches `train_llm.main()` with `--config_class __main__.C
  --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.
- Descriptor: `autoresearch/ideas/167-logit-zloss/run.json` —
  `{"name": "167-logit-zloss", "arq_file": "_arq_167-logit-zloss.py",
  "job_timeout": "12m"}`.
- Daemon (`autoresearch/bin/queue-daemon.sh`): scp's the stub, runs
  the CPU build-smoke (`MinimalLLM(C())` constructs without error),
  then launches the run in the `arq` tmux.
- **Pass/fail bar** (per review.md action items 3,4):
  - **WIN (logit magnitude is binding at 0.94M):** treatment val ≤
    ctrl val − 0.005 (i.e. Δ ≤ −0.005, exceeds twice the box's
    measurement noise floor for a single-seed ±0.04 band).
  - **NULL (logit explosion not the binding constraint at 0.94M):**
    |treatment val − ctrl val| < 0.005. Null closes the z-loss
    axis at this tier.
  - **DRIFT (lever harmful):** treatment val ≥ ctrl val + 0.005.
  - Crash / NaN / OOM → `needs-recode` (round 1, inside budget).
- Reference: closed loss-shape axes 066-070 are recorded in
  `closed.md` (066/067/069 taste-rejected, 068/070 unlikelihood
  / MTP-head taste-rejected, 115-rdrop null closes R-Drop and
  names 066-070 as adjacent). All five target *target/prediction*
  softening; z-loss targets *logit magnitude* — orthogonal axis.
- Read-out: the daemon greps `Final Val Loss:` from the per-idea log
  (`/root/arq/logs/167-logit-zloss.log` on the box; the local
  equivalent is whatever path the queue-daemon assigns). Pass/fail
  is then arithmetically applied via `autoresearch/bin/baseline.sh
  verdict` against the recorded ctrl mean ± band.

## Self-check
- (a) Flag OFF reproduces the control — `use_z_loss=False` or
  `z_loss_lambda=0.0` ⇒ the `else logits.new_zeros(())` branch
  fires ⇒ total loss is byte-identical to no-z-loss baseline
  across all 92 steps (z-loss adds a 0 scalar via
  `+ z_loss` in the accumulation, which is the same code path
  the baseline uses — just with a zero addend).
- (b) Treatment path exercises the new code —
  `use_z_loss=True, z_loss_lambda=1e-4` ⇒ `z = (logits.logsumexp
  (dim=-1) ** 2).mean()` is computed and scaled by 1e-4 in the
  total train loss; `z_loss_val` is logged per step.
- (c) Build-smoke — `MinimalLLM(Tiny1M3MZLossConfig())`
  constructs on CPU without error (no `nn.Parameter` is added
  by the z-loss lever; same `MinimalLLM(C())` shape as the bare
  baseline).
- (d) Partition-function sanity — `logits.logsumexp(dim=-1)` on
  a synthetic `[1, 1, 4]` logits tensor matches a hand-computed
  `log(sum_v exp(logits_v))` to within fp32 noise
  (~1e-7 max-abs-diff). NOT `sum_v logits_v` (the spec's warning).