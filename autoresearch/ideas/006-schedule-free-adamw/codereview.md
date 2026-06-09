# Code review — 006 schedule-free-adamw

## r2 — 2026-06-09 — verdict: accept

### Re-review of the double-momentum fix
Read `optimizers/schedule_free_adamw.py` end-to-end against the r1 findings. The
double-momentum bug is **gone**.

- **State init (lines 122-125)**: only `z`, `x`, `v` are allocated. No
  `exp_avg`. No `bc1`. Confirmed at runtime — `sorted(state.keys())` is
  `['step', 'v', 'x', 'z']` after a real `.step()`.
- **z update (line 146)**: `z.addcdiv_(grad, denom, value=-lr)`. The RAW
  gradient is consumed directly into `z`. No first-moment EMA, no `bc1`.
  Matches `facebookresearch/schedule_free` `AdamWScheduleFree.step` and
  Algorithm 1 of Defazio et al. 2024.
- **Bias correction (lines 142-143)**: `bc2 = 1 - beta2 ** step_t`, applied to
  `v.sqrt() / math.sqrt(bc2)`. No `bc1` anywhere in the file. Correct.
- **Weight decay (line 136)**: decoupled, applied to `z` before the grad step.
  Matches the canonical convention.
- **c ramp (lines 100-105)**: `c=1` during warmup, `c=1/(k-warmup+1)` after.
  Matches Algorithm 1.

### Math verification (deterministic, torch tensors)
Ran a single step with a 2×2 param and 2×2 grad at `lr=0.01, betas=(0.9, 0.999),
eps=1e-8, weight_decay=0.1, warmup_steps=0`. The optimizer's `z`, `x`, and
`p.data` (= y) match a hand-computed reference to all printed digits. Notably,
`c=1` at k=0 makes `x_new = z_new` (Polyak-Ruppert with c=1 reduces to
identity-copy of z), and the optimizer reproduces this exactly.

### Eval/train swap
- `eval()` (lines 57-68): copies `state["x"]` into `p.data`. No stale
  `beta1 = group["betas"][0]` line — the r1 "dead code" finding is also fixed.
- `train()` (lines 70-83): reconstructs `y = (1-β1)*z + β1*x` and copies into
  `p.data`. Verified `torch.allclose` against the analytic `y`.
- `group["train_mode"]` flag guards idempotency — calling `eval()` twice is a
  no-op the second time.

### Plan consistency
- `plan.md:29` memory line already updated to "3 new state tensors: z, x, v"
  ≈ 4.8 MB. r1's third finding is closed.
- Pass/fail bar in `plan.md` (pass ≤ 6.4206, fail > 6.4287, noise |Δ| ≤ 0.005)
  matches `idea.md` exactly.
- Control / treatment match: control = V+q+SWA+HighRoPE + AdamW + Muon (seed
  42, tiny1m3m); treatment = same + `use_schedule_free_adamw=True`.

### Flag wiring (sanity)
- `use_schedule_free_adamw: bool = False` is a single boolean, default OFF.
  Smoke confirms construction: flag-OFF path uses the existing
  `torch.optim.AdamW(...)` branch; flag-ON path constructs
  `ScheduleFreeAdamW(adamw_params, lr=..., weight_decay=...)` with the
  same 407,488 AdamW-routed params.
- `setup_muon_optimizer` override note in `plan.md:16` ("overrides
  `use_cautious_adamw`") matches the gate logic — SF has no sign-mask
  variant, so the cautious gate is shadowed when SF is on. Acceptable,
  noted in `idea.md:25` as "orthogonal lever for a future co-test."

### Verdict
Canonical SF-AdamW. Implementation matches the paper and the reference
PyTorch code. Smoke build + step + eval/train swap all pass. No silent
HP drift, no reordering, no seed change, no LoC bloat (~150 LoC as planned).
Status → `needs-run`.

---

## r1 — 2026-06-09 — verdict: revise

### Bug — double momentum (beta1 used twice, faithful-to-spec violation)

**File:** `optimizers/schedule_free_adamw.py:114, 123, 131-136, 139`

**The divergence.** Canonical Schedule-Free AdamW (Defazio et al. 2024, arXiv:2405.15682; official `facebookresearch/schedule_free` `AdamWScheduleFree.step`) has **no first-moment EMA buffer**. The update is

```
denom   = sqrt(v_hat) + eps   # v_hat = v / (1 - beta2**t)
z      ← z - lr * g / denom   # g is the RAW gradient, not g_hat
x      ← (1-c)·x + c·z
y      ← (1-β1)·z + β1·x
```

`beta1` does **one** job: the y-interpolation that gives the iterate its momentum. The gradient is consumed directly. There is no `exp_avg`, no `bc1`, no `exp_avg.mul_(beta1).add_(grad, …)`.

The current implementation maintains BOTH an `exp_avg` first-moment EMA (line 114 init, line 133 update) AND a beta1 y-interpolation (line 145). The y/z interpolation is the *momentum* — adding a beta1-EMA on top of the raw gradient double-smooths the signal. The two mechanisms fight each other: the y-interpolation wants to give weight `β1` to the slow average, and the exp_avg also wants to give weight `β1` to the slow gradient EMA. The optimizer will not match the paper and will not match the official implementation's loss curves.

**The smoking gun on line 130 comment**: "Adam moment updates (gradient was computed at y = current p.data)" — if the gradient is computed at y (the schedule-free mechanism), there is no separate first-moment to maintain. The line betrays that `exp_avg`/`bc1` is AdamW muscle memory, not part of SF-AdamW.

**Fix.** Remove the first-moment buffer entirely. Replace lines 114, 123, 131-139 with:

```python
# In state init: drop exp_avg; keep only v (= exp_avg_sq).
state["v"] = torch.zeros_like(p)

# In step:
v = state["v"]
v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
bc2 = 1 - beta2 ** step_t
denom = v.sqrt() / math.sqrt(bc2) + eps
z.addcdiv_(grad, denom, value=-lr)   # raw grad, no exp_avg / no bc1
```

Drop `bc1` and the `adam_update` variable. Re-test on tiny1m3m after the change. Expected to make a non-trivial difference in val-loss trajectory because the current double-momentum is roughly equivalent to running SF-AdamW with `β1_eff ≈ 1 - (1-β1)² ≈ 0.99` for default `β1=0.9`, which is a meaningfully different optimizer.

### Minor — dead code in `eval()`

**File:** `optimizers/schedule_free_adamw.py:53`

`beta1 = group["betas"][0]` is computed but never used. The eval swap is a direct `p.data.copy_(state["x"])`. Remove the line. Cosmetic.

### Minor — memory footprint claim in `plan.md:29` is now wrong

**File:** `autoresearch/ideas/006-schedule-free-adamw/plan.md:29`

Plan claims "4 new state tensors: z, x, exp_avg, exp_avg_sq" ≈ 6.4 MB. After the fix, it's 3 (z, x, v) ≈ 4.8 MB. Update the plan to match — do not amend the spec, just the budget line.

### What's correct

- `eval()` / `train()` swap: correct. `p.data` holds y during step; eval copies `x` (the Polyak-Ruppert average) into `p.data`; `train()` reconstructs y from current z and x. State["x"] and state["z"] are stored directly so the swap is exact.
- Weight decay applied to z (line 127-128) — matches the canonical decoupled-WD-on-z convention.
- c ramp (line 95-98): `c=1` during warmup, `c=1/(k-warmup+1)` after. Matches Algorithm 1.
- Sparse-gradient raise (line 104-107) — fine, conservative.
- Bias correction on v (line 132, 135) — correct (`bc2` only, no `bc1` once the exp_avg is removed).
- Flag is a single boolean defaulting to OFF in `configs/llm_config.py`. Step-0 path is the existing `torch.optim.AdamW` branch and is bit-identical to baseline.

### Routing

Bug found, fixable in 1 pass. Status → `needs-recode` for the implementer. One round of revise before re-review (round 2).
