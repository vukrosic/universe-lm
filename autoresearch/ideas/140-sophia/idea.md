---
id: 140-sophia
status: revising
round: 2
updated: 2026-06-14T04:32:13Z
transfer-risk: med
plain: An optimizer that uses the second derivative (curvature) of the loss to take bigger steps in flat directions and smaller steps in steep ones.
---

# 140 — Sophia Optimizer

## Source
Liu, Wang, et al. 2023, "Sophia: A Scalable Stochastic Second-order Optimizer", Stanford, arXiv:2305.14342. https://arxiv.org/abs/2305.14342

## Mechanism
Diagonal-Hessian-aware update with Hutchinson trace estimator.
- `m_t = β1 * m_{t-1} + (1-β1) * g_t`  (gradient EMA)
- `h_t = β2 * h_{t-1} + (1-β2) * h_hat_t`  (Hessian diagonal EMA, sampled every k steps)
- `update = clip(g_t, max=ρ) / max(h_t, ε)`  (preconditioned by inverse Hessian diagonal)
- `θ_t = θ_{t-1} - lr * (update + λ * θ_{t-1})`

Hessian diagonal is estimated via Hutchinson's trick: draw `u ~ N(0, I)`, compute `uᵀ H u ≈ uᵀ (∇g)` via one extra backward pass on a `g·u` scalar. Cost: ~2× backward time, but only every k=10 steps (so amortized ~1.1× backward).

## Design sketch (how it works + how to build it)
- New file `optimizers/sophia.py` with `Sophia(params, lr, betas, weight_decay, rho, hessian_update_freq)`. ~120 LoC.
- Add `use_sophia: bool = False` to `configs/llm_config.py`.
- In `trainer.py`, every `hessian_update_freq` steps, do an extra `loss.backward()` with a `g·u` scalar to populate `h_hat`. < 80 LoC integration.
- Identity at step 0: `m=0`, `h=0`. First update is `clip(g, ρ) / max(0+ε) ≈ g/ε` — large but bounded. Model output at step 0 (forward pass) is the baseline forward.
- Why a real lever, not a hyperparam: the Hutchinson estimator is the *only* way to get curvature information at sub-Quadratic cost. AdamW has no curvature signal; Sophia explicitly precondition by inverse Hessian diagonal, which is a different optimization path.
- Targets baseline failure: AdamW's `m/√v` treats all directions equally (modulo recent gradient variance), so it ignores the loss landscape's actual curvature. At 0.94M with only 92 update steps, a curvature-aware step might converge faster per step.

## Scale evidence
Paper trains 125M and 1.5B GPT-2 models; 2× faster wall-clock convergence than AdamW reported at 1.5B. 0.94M is below the paper's tested range, and Hutchinson is noisy at small scale (the diagonal Hessian has high variance when the model is tiny). Transfer risk: med — gain plausibly shrinks at sub-million params because the Hessian diagonal is over a 0.94M-dim space and is noisy per-step.

## Why it's worth a slot
Sophia is the simplest second-order optimizer that doesn't require full Hessian inversion — the next natural step after the AdamW variant wave (110–138). If it works at 0.94M, it's a paradigm shift for tiny training (curvature-aware at < 1M params). If it doesn't, it tells us 0.94M is too small for second-order methods to amortize their 2× backward cost, closing the second-order axis for Phase-1.

## Plan

### Files changed
- `optimizers/sophia.py` (NEW, ~200 LoC) — `Sophia` optimizer class with per-parameter `m_t` (gradient EMA), `h_t` (Hessian-diagonal EMA), and `update_hessian(h_hat_list, beta2)` method to ingest the Hutchinson sample. Per-element preconditioning `update = clip(g, ±ρ) / max(h, ε)` plus a per-parameter `update_clip` magnitude guard for the cold-start `h_t ≈ 0` case. Tracks `_step_count` so the trainer can fire the Hutchinson sample on a known schedule.
- `optimizers/__init__.py` — `from .sophia import Sophia` + add `'Sophia'` to `__all__`.
- `configs/llm_config.py` — add `use_sophia: bool = False` + 7 hyperparameters (`sophia_lr`, `sophia_beta1`, `sophia_beta2`, `sophia_eps`, `sophia_rho`, `sophia_hessian_freq`, `sophia_update_clip`). Defaults match the paper's 125M model (lr=6e-3, β1=0.965, β2=0.99, ρ=0.04, k=10). Add `Tiny1M3MSophiaConfig` that flips the flag and pins the paper defaults for the A/B run.
- `training/trainer.py` — top-level `from optimizers.sophia import Sophia`; add the `use_sophia` branch in the AdamW-replacement elif chain; add a Hutchinson block in `train_model` right after `torch.nn.utils.clip_grad_norm_` that (a) checks `sophia_opt._step_count % hessian_freq == 0`, (b) builds a local `adamw_params` list from `sophia_opt.param_groups[*]['params']` so the block does not depend on `setup_muon_optimizer()` scope, (c) saves the post-clip grads, (d) samples `u ~ Rademacher(±1)` per parameter, (e) computes the scalar `g·u` and runs a second backward to populate `p.grad = H·u`, (f) builds `h_hat = u · (H·u)`, (g) restores the original grads so `.step()` uses `g_t` not `H·u`, (h) calls `sophia_opt.update_hessian(h_hat_list, beta2)`.
- `_arq_140-sophia.py` (NEW) — A/B wrapper that subclasses `Tiny1M3MSophiaConfig` and invokes `train_llm.main()` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.

### Config flag name
`use_sophia: bool = False` (default off → baseline path bit-identical).

### Step-0 identity
With `use_sophia=False` (default) the `Sophia` class is never instantiated and the trainer uses plain `torch.optim.AdamW` unchanged — baseline path bit-identical. With `use_sophia=True` and `Sophia`, the first optimizer step has `m_0 = 0`, `h_0 = 0` (state lazily initialized in `.step()`). The Hutchinson block fires at `step=0` (when `_step_count=0`), so `h_t` is initialized to `h_hat_0` BEFORE the first `.step()` call. The update is `clip(g_0, ρ) / max(h_hat_0, ε)`, then the `update_clip=1.0` magnitude guard clips the per-element update to ±1.0 before the `lr` scale, so the first-step magnitude is bounded by `lr · 1.0` (same order as AdamW's first step). NOT bit-identical to AdamW's first step — the diagonal preconditioner IS the lever — but the magnitude matches by construction and the model does not diverge. The forward graph is unchanged, so step-0 `val_loss` (computed before any optimizer step) IS bit-identical to baseline.

### Run command
```
/venv/main/bin/python _arq_140-sophia.py
```
(matches the convention of the other `_arq_*.py` wrappers — they invoke `train_llm.main()` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`).

### Pass bar
- Final `val_loss` is the last `eval/milestone` entry in `metrics_history['val_losses']` / `logs/<run-name>/log.jsonl`.
- Compare against the locked tiny1m3m baseline cache at `autoresearch/baseline-cache.json` (`6.4394 ± 0.04`).
- `WIN` if `Δ = val_loss - baseline ≤ -0.01`.
- `NULL` if `|Δ| < 0.01`.
- `DRIFT` if `Δ > +0.01`.

### Re-code note (2026-06-14, round 1 → 2)
A previous GPU run failed with `ImportError: cannot import name 'Tiny1M3MSophiaConfig' from configs.llm_config` because the box was stale at `7a69c1a` and missing the `bd5adf5` commit that introduced the config class. **No local code change is required** — the class exists at `configs/llm_config.py:2084` and `optimizers/sophia.py` imports cleanly. The local smoke test passed: `MinimalLLM(Tiny1M3MSophiaConfig)` builds 949,056 params (bit-identical to `Tiny1M3MConfig`), forward at step 0 is bit-identical when seeded the same way, and `Sophia._step_count` increments correctly through `.step()`. The box must `git pull` (or fast-forward to a commit ≥ `bd5adf5`) before the next queue picks up `_arq_140-sophia.py`; otherwise the import will keep failing.

### How the final val loss is read
The trainer's `train_model` loop writes `metrics.json` to the run's `output_dir` after each eval milestone. The last entry in `metrics_history['val_losses']` (and the matching `val_perplexities`, `val_accuracies`) is the final raw `val_loss`; the runner compares that value against the pass bar above.
