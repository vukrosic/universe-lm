# Plan — 006 Schedule-Free AdamW

## Flag
- `configs/llm_config.py:395` (line after `use_soap_precondition_freq`)
  - `use_schedule_free_adamw: bool = False` — default OFF → baseline path
    bit-identical.

## Change

| File | Edit |
|---|---|
| `optimizers/schedule_free_adamw.py` | NEW (~150 LoC). `ScheduleFreeAdamW(Optimizer)`. 3-point iterate `(z, x, y)` where `y = (1-β1)·z + β1·x`. `p.data` holds `y` for forward/backward; `eval()` swaps to `x` (Polyak-Ruppert average) for evaluation; `train()` swaps back to `y`. Constant LR (no schedule); warmup steps control the internal `c` ramp (`c=1` during warmup, `c=1/(k-warmup_steps+1)` after). `eval()`/`train()` are idempotent on the `group["train_mode"]` flag. |
| `optimizers/__init__.py` | Add `from .schedule_free_adamw import ScheduleFreeAdamW` and `ScheduleFreeAdamW` to `__all__`. |
| `training/trainer.py:14-16` | Import `ScheduleFreeAdamW`. |
| `training/trainer.py:48-62` | Helper `_swap_optimizers_eval_mode(optimizers, mode)`. Duck-types on `hasattr(opt, "eval")`/`"train"` — no-op for non-SF optimizers (Muon, AdamW, Cautious, SOAP). |
| `training/trainer.py:160-209` | Gate in `setup_muon_optimizer`. If `use_schedule_free_adamw=True`, construct `ScheduleFreeAdamW(adamw_params, lr=adamw_lr, weight_decay=...)` (overrides `use_cautious_adamw`; SF has no sign-mask variant — orthogonal lever for a future co-test). |
| `training/trainer.py:531-535, 638-642` | Eval/train swap around `evaluate_model(model, val_loader, config)` calls (milestone and final). |
| `training/trainer.py:849-861` | Force `schedule_type='constant'` when SF is on (the optimizer's internal averaging handles late-training stabilization). |

Step-0 (flag OFF) — `use_schedule_free_adamw=False`, the AdamW path is unchanged (`torch.optim.AdamW` with the cautious gate). The chain is bit-identical to the current path.

## Control
- **Control**: V+q+SWA+HighRoPE + AdamW (1D + embedding + head) + Muon (2D hidden). Seed 42. Tier `tiny1m3m` (0.94M params, 3M tokens).
- **Treatment**: control + `use_schedule_free_adamw=True` → AdamW replaced by Schedule-Free AdamW (same params routed, but with iterate averaging and no external LR schedule). Seed 42. Tier `tiny1m3m`.

## Cost
- **Params**: 0 (re-routes the existing AdamW params).
- **FLOPs/step**: per SF param, ~1× Adam update on `z` + 1× EMA on `x` + 1× `y = (1-β1)·z + β1·x` reconstruction ≈ 1× AdamW. Plus the eval/train swaps, which are O(num_params) byte copies — paid at most every `eval_milestones` step.
- **Memory**: per SF param, 4 new state tensors: `z` (param-shaped), `x` (param-shaped), `exp_avg` (param-shaped), `exp_avg_sq` (param-shaped). For tiny1m3m the AdamW path is 407,488 params, so SF carries ~6.4 MB extra state (vs ~3.2 MB for AdamW's two state buffers).
- **Schedule cost**: forcing `schedule_type='constant'` removes the cosine/warmup-decay-to-zero LR decay. The SF optimizer's internal `c = 1/(k-warmup+1)` averaging takes over the late-training stabilization.

## Run

### Step 0 — smoke (gate)
CPU smoke test of `ScheduleFreeAdamW` + `setup_muon_optimizer` integration on `Tiny1M3MConfig`. Confirm:
- `use_schedule_free_adamw=False` → optimizers `[Muon, AdamW]`, AdamW manages 407,488 params.
- `use_schedule_free_adamw=True` → optimizers `[Muon, ScheduleFreeAdamW]`, SF manages 407,488 params.
- One forward+backward+optimizer.step completes cleanly.
- `sf.eval()` swaps `p.data` from `y` to `x` (verified via `torch.equal`); `sf.train()` swaps back.

### Step 1 — full A/B on `tiny1m3m`
```bash
# Control (baseline AdamW)
python train_llm.py --config tiny1m3m --seed 42 --out runs/tiny1m3m-sf-ctrl

# Treatment
python train_llm.py --config tiny1m3m --seed 42 --use_schedule_free_adamw True --out runs/tiny1m3m-sf-trt
```
Wall-clock: ~10-15 min each on a single A100 / T4. Pass/fail bar from `idea.md`:
- pass: treatment val ≤ 6.4206 (ctrl 6.4287, target Δ = −0.0081)
- fail: treatment val > 6.4287
- noise: |Δ| ≤ 0.005

### Step 2 — verdict (single seed, per pipeline hard rule)
Seed 42, single seed. |Δ| ≤ 0.005 is the noise band; a sub-noise result is logged inconclusive, not re-seeded.

## Self-check (before release to code-reviewer)
- `use_schedule_free_adamw=False` reproduces the control (no numeric drift) — confirm by inspecting the gate: with the flag off, the construction branch is the existing `torch.optim.AdamW(...)` path. Smoke shows `[Muon, AdamW]` with the same 407,488 AdamW params.
- The treatment path actually exercises SF — confirm by smoke: 10 optimizer steps, `state["step"]` advances, `k` group counter advances, eval swap toggles `group["train_mode"]` and changes `p.data`.
- `plan.md` pass/fail bar matches `idea.md` — both: pass ≤ 6.4206, fail > 6.4287, noise |Δ| ≤ 0.005.
