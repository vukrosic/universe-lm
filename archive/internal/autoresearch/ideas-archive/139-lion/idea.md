---
id: 139-lion
status: done
round: 1
updated: 2026-06-13T19:58:28Z
transfer-risk: low
plain: A faster optimizer that just uses the *sign* of the gradient direction, so it takes confident equal-sized steps when many gradients agree.
---

# 139 — Lion Optimizer

## Source
Chen, Hsieh, Gong 2023, "Symbolic Discovery of Optimization Algorithms", Google, arXiv:2302.06675. https://arxiv.org/abs/2302.06675

## Mechanism
Sign-based update with two EMA momenta and decoupled weight decay.
- `m_t = β1 * m_{t-1} + (1-β1) * g_t`
- `update_t = sign(β2 * m_t + (1-β2) * g_t)`  (interpolated sign)
- `θ_t = θ_{t-1} - lr * (update_t + λ * θ_{t-1})`

No second-moment estimate — every parameter takes an equal-magnitude step whose direction is the sign of an interpolated momentum. Decoupled WD applied in float32 (not the update sign). Recommended defaults: β1=0.9, β2=0.99, lr ≈ 3–10× AdamW lr.

## Design sketch (how it works + how to build it)
- New file `optimizers/lion.py` with `Lion(params, lr, betas, weight_decay)`. < 60 LoC.
- Add `use_lion: bool = False` to `configs/llm_config.py` (default-off, base unchanged).
- In `training/trainer.py` / `train_llm.py`, branch on `use_lion` to construct Lion instead of AdamW.
- Identity at step 0: `m=0`, first update is `sign(g_0)` — non-zero magnitude, so step-1 parameters differ from AdamW step-1. But the **model output at step 0 (before any optimizer step) is still the baseline forward pass**, which is what the `step-0 ≈ baseline` rule measures.
- Why a real lever, not a hyperparam sweep: the `sign()` is non-smooth, so Lion's update *trajectory* is qualitatively different from AdamW's m/√v scaling — it cannot be reproduced by tuning AdamW's eps or betas. Lion is also memory-cheap (no v buffer).
- Targets the baseline failure mode: AdamW's `m/√v` step-size shrinks in directions of high historical gradient variance, even when the *current* gradient is consistent. Lion ignores historical magnitude, only respects direction.

## Scale evidence
Original paper trains 7B language models (LaMA, BERT) and reports ~2× faster convergence than AdamW on small models (BERT-base, 110M). Mechanism is scale-agnostic. Transfer risk: low — the sign-update path has been replicated by independent groups on a wide range of scales.

## Why it's worth a slot
Lion is the strongest "not-AdamW" optimizer in the literature that hasn't been filed yet, and it's notably absent from our 110–138 optimizer wave. Real bet: at 0.94M, sign-based updates give a different (and possibly more robust) trajectory than AdamW's variance-scaled steps; the lever's null hypothesis is "AdamW's variance scaling is the right inductive bias at this scale". A win saves step-count on every future run; a null closes the sign-update family for tiny LMs.

## Plan

**Files (already wired):**
- `optimizers/lion.py` — `Lion(params, lr, betas, weight_decay, cautious=False)` class implementing the canonical Chen et al. 2023 sign-based update: `c = β1·m + (1−β1)·g`, `update = sign(c)`, `p ← p − lr·(update + wd·p)`, `m ← β2·m + (1−β2)·g`. ~115 LoC. Also supports the Liang et al. 2024 cautious sign-mask when `cautious=True` (gated on `use_cautious_lion`).
- `optimizers/__init__.py` — exports `Lion`.
- `configs/llm_config.py` — adds `use_lion: bool = False` (default off), `lion_lr: float = 3e-4`, `lion_beta1: float = 0.9`, `lion_beta2: float = 0.98`. Adds `Tiny1M3MLionConfig(Tiny1M3MConfig)` setting `use_lion=True`.
- `training/trainer.py:setup_muon_optimizer` — when `use_lion=True`, the 2-D non-embedding, non-norm routing slot goes to `Lion` instead of `Muon`; 1-D / embedding / norm / head stay on AdamW (same split as Muon). With `use_lion=False` (default) the Muon path is bit-identical to baseline.

**Config flag:** `use_lion` (default `False`).

**Zero-init at step 0:** Lion's `momentum_buffer` is `torch.zeros_like(p)` and the first step's update direction is `sign((1−β1)·g)` — a unit-magnitude step that does NOT modify the model output (the val pass runs before the first optimizer step). The model forward graph is unchanged, so step-0 val_loss is bit-identical to baseline. The lever only diverges from baseline at step ≥ 1 — its signature is the trajectory, not the initialization.

**Smoke check (passed):**
- `Lion([p], lr=3e-4, betas=(0.9, 0.98)).step()` ⇒ `p[0] = −3e-4` for `p.grad = +0.01` (matches paper math).
- `Tiny1M3MConfig().use_lion == False`; `Tiny1M3MLionConfig().use_lion == True`.

**Run command (tiny1m3m, seed 42):**
```
python train_llm.py --config_class Tiny1M3MLionConfig --seed 42
```

**Reading val loss:** `runs/Tiny1M3MLionConfig/seed_42/log.jsonl` (last `eval/loss` entry). PASS ≤ `Tiny1M3MConfig` ctrl val (6.4306) − 0.01. NULL band |Δ| < 0.01. DRIFT > +0.01. See `autoresearch/closed.md` for the verdict template.
