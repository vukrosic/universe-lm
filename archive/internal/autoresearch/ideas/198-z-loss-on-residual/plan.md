# Plan — 198 z-loss-on-residual

## Flag

- `use_residual_zloss: bool = False` and `zloss_coef: float = 1e-4` on
  `LLMConfig` (`configs/llm_config.py`, after the 167 block at lines 591-592).
  New subclass `Tiny1M3MResidualZLossConfig(Tiny1M3MConfig)` sets both to ON
  (`True` / `1e-4`). Defaults OFF ⇒ baseline path byte-identical.

## Change

- **`configs/llm_config.py`** (+~25 LoC)
  - Add two fields to `LLMConfig` with the same `getattr`-friendly docstring
    style as 167 (PaLM z-loss):
    ```python
    use_residual_zloss: bool = False
    zloss_coef: float = 1e-4
    ```
  - Add `Tiny1M3MResidualZLossConfig(Tiny1M3MConfig)` subclass with
    `@dataclass` decorator and `use_residual_zloss: bool = True,
    zloss_coef: float = 1e-4` — mirrors the 167 subclass pattern at
    `configs/llm_config.py:6493+`.

- **`models/llm.py`** (+~15 LoC)
  - In `MinimalLLM.__init__`, add the two `getattr(config, ...)` reads.
  - In `_run_post_embed`, just before `compute_logits` is called (right
    after `x = self.output_dropout(x)`), stash on the model:
    ```python
    if self.use_residual_zloss and self.zloss_coef > 0.0:
        # Per-token L2 norm of the final residual stream (after final
        # norm + output dropouts, before the LM head). Penalty is
        # `zloss_coef * log1p(||r||²).mean()`; squared-norm form is
        # `torch.log1p((x ** 2).sum(dim=-1)).mean()` which avoids the
        # extra `(B, T)`-shaped tensor that `.norm(dim=-1)` allocates.
        # The penalty is bounded below by 0 (log1p ≥ 0) and unbounded
        # above — quadratic for small `||r||`, logarithmic for large.
        # Step-0 at d_model=64, n_layers=12: `||r|| ≈ √(12·Var(r)) ≈
        # 3.5` ⇒ `log1p(12) ≈ 2.56` ⇒ `zloss_coef · 2.56 ≈ 2.56e-4`,
        # which is `O(1e-4)` — a tiny but non-zero correction at step
        # 0. The forward graph is otherwise unchanged; the LM head
        # still reads `x` directly. See `autoresearch/ideas/198-z-
        # loss-on-residual/idea.md` §Mechanism.
        self._residual_zloss = self.zloss_coef * torch.log1p(
            (x ** 2).sum(dim=-1)
        ).mean()
        # Detached form for per-step logging only — must NOT be the
        # autograd-graph-bearing tensor or `.item()` would leak the
        # graph. Trainer logs `res_norm_mean` and `res_norm_max` for
        # the falsification signature trace.
        self._residual_norm = (x ** 2).sum(dim=-1).sqrt()
    else:
        self._residual_zloss = None
        self._residual_norm = None
    ```
  - The existing flag path is unconditional when the flag is on. The
    `_residual_zloss = None` branch is what the trainer checks via
    `getattr(model, "_residual_zloss", None)` — when None, the trainer
    substitutes `logits.new_zeros(())` so the loss equation is unchanged.

- **`training/trainer.py`** (+~20 LoC across both branches)
  - At line ~1467 (right after the `use_z_loss = ...` line), add:
    ```python
    zloss_coef = getattr(config, "zloss_coef", 1e-4)
    use_residual_zloss = getattr(config, "use_residual_zloss", False) and zloss_coef > 0.0
    ```
  - In both the AMP branch (~line 1547) and the CPU branch (~line 1653),
    right after `entropy_reg_loss = _collect_entropy_reg(model)`, add:
    ```python
    residual_zloss = (
        model._residual_zloss
        if (use_residual_zloss and getattr(model, "_residual_zloss", None) is not None)
        else logits.new_zeros(())
    )
    ```
  - Add `+ residual_zloss` to the loss sum in both branches
    (lines 1599 and 1702).
  - Right after the `poly_loss_val = ...` detach (line 1715), add:
    ```python
    res_norm_mean = (
        model._residual_norm.mean().item()
        if getattr(model, "_residual_norm", None) is not None
        else 0.0
    )
    res_norm_max = (
        model._residual_norm.max().item()
        if getattr(model, "_residual_norm", None) is not None
        else 0.0
    )
    ```
  - Add `residual_zloss.detach().item()` for logging only (mirror of
    `z_loss_val`). Add `'rzl': f'{residual_zloss.detach().item():+.2e}'`
    to the pbar postfix dict.

- **`autoresearch/ideas/198-z-loss-on-residual/run.json`**: tiny
  descriptor `{ "name": "198-z-loss-on-residual", "arq_file":
  "_arq_198-z-loss-on-residual.py" }`.

- **`_arq_198-z-loss-on-residual.py`** (repo root): top-level
  `C(Tiny1M3MResidualZLossConfig)` import + `__main__` block that
  drives `train_llm.main()` with `--config_class __main__.C --seed 42
  --dataset_path processed_data/pretrain_1B --warmup false`. Mirrors
  `_arq_167-logit-zloss.py`.

Step-0 (flag OFF) byte-identity: the `if self.use_residual_zloss and
...` guard is False at the LLMConfig default of False, so the entire
stash branch is skipped; no `torch.log1p`, no `.sum(dim=-1)`, no
extra allocation. The forward graph is identical to the 167-off
baseline — same tensor ops, same RNG consumption, same gradient flow.
The trainer's `getattr(model, "_residual_zloss", None)` returns None
when the flag is off, so it substitutes `logits.new_zeros(())` and
the loss equation has an exact `+ 0.0` term.

Step-0 (flag ON): penalty is `1e-4 · log1p(12) ≈ 2.56e-4` per step at
tiny1m3m (d_model=64, n_layers=12, residual grows by O(√L)). This is
non-zero but small (~3 ppm of the step-0 CE loss ≈ 9.0) — not
byte-identical at step 0, but the contribution is below the
forward-graph noise floor. The pre-registered falsification signature
(`||r||_L2` trace at steps 0/100/500/1000/2000) is the right
diagnostic.

## Control

- **Control**: `Tiny1M3MConfig` (seed 42, flag OFF). The daemon owns
  this — we never ship a ctrl.
- **Treatment**: `Tiny1M3MResidualZLossConfig` (seed 42, flag ON with
  `zloss_coef=1e-4`). A/B vs the champion baseline.
- **Tier**: tiny1m3m (12L, d_model=64, 0.94M params).
- **Seed**: 42 (single seed — see §5 of the protocol).

## Cost

- **Params**: +0 (regularizer, no learnable parameters).
- **FLOPs**: ~2 extra per-token ops (square + sum + log1p + mean) per
  forward. At tiny1m3m (~10M activations), this is < 0.01% of the
  forward cost.
- **Memory**: one extra `[B, T]`-shaped float tensor (~32 KB at
  B=2, T=2048, fp32) for the residual norm stash. Negligible.
- **Backward**: identical extra cost — the penalty is a scalar so
  only one backward pass per step.

## Run

```bash
# Run via the daemon (autoresearch/bin/queue-daemon.sh), which prepends
# N≥3 ctrls only when baseline.sh check returns MEASURE. Standard
# training loop, 92-step horizon, seed 42.
python _arq_198-z-loss-on-residual.py
```

- **Tier**: tiny1m3m (~92-step horizon, val loss at the end).
- **Seed**: 42 (single seed).
- **Wall-clock**: ~3-4 minutes per run on Vast V100 (matches 167 and
  other regularizer A/Bs at this tier).
- **Pass/fail bar** (verbatim from idea.md):
  - **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule,
    AND re-runs `mean(||r||_L2) ≥ 5·√d_model = 40` at step 2k
    (falsifiable WIN signature).
  - **NULL**: `|trt_val − ctrl_val| < 0.01`. Sub-classify by the
    residual-norm trace at step 2k:
    - `||r||_L2 ≤ 3·√d_model = 24`: *symmetric axis closure* to 167
      (well-conditioned, lever cannot bind).
    - `||r||_L2 ≥ 5·√d_model = 40`: *steered NULL* (binding regime,
      optimizer ignores the gradient — would warrant a follow-up at
      `z_coef=1e-3`).
    - `24 < ||r||_L2 < 40`: *indeterminate* — NULL here is not a
      clean closure and warrants a follow-up at higher `z_coef`.
  - **DRIFT**: `trt_val > ctrl_val + 0.01`.
- **Falsification trace**: `residual_norm_mean` and `residual_norm_max`
  are logged per step in the pbar (`'rzl'`, `'rn'` keys). The
  post-run verdict reads `||r||_L2` at step 2k from these to classify
  the NULL into one of the three regimes above.
