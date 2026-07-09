---
id: 141-adabelief
status: done
round: 1
updated: 2026-06-13T20:01:58Z
transfer-risk: med
plain: An optimizer that watches how much the gradient keeps wiggling around its average direction and uses that to decide each parameter's step size, instead of just the gradient's size like Adam does.
---

# 141 — AdaBelief Optimizer

## Source
Zhuang et al. 2020, "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients", NeurIPS 2020, arXiv:2010.07468. https://arxiv.org/abs/2010.07468

## Mechanism
Replaces AdamW's `v = E[g²]` with `s = E[(g − m)²] + ε`, where `m = E[g]`. The "belief" is the variance of the *residual* (g − m), not the squared gradient itself.
- `m_t = β1 * m_{t-1} + (1-β1) * g_t`
- `s_t = β2 * s_{t-1} + (1-β2) * (g_t − m_t)² + ε`
- `update = m_t / √s_t`
- `θ_t = θ_{t-1} - lr * (update + λ * θ_{t-1})`

Key intuition: when the current gradient `g_t` agrees with the momentum `m_t` (small residual), `s` is small → step is large (we trust the direction). When `g_t` disagrees with `m_t` (large residual), `s` is large → step is small. AdamW does the *opposite*: large `g²` makes `v` large, *shrinking* the step — which can be wrong when a large gradient is a *good* direction, not a noisy one.

## Design sketch (how it works + how to build it)
- New file `optimizers/adabelief.py` with `AdaBelief(params, lr, betas, weight_decay, eps)`. ~60 LoC. The only change vs AdamW is `s = (g - m)²` instead of `v = g²`.
- Add `use_adabelief: bool = False` to `configs/llm_config.py`.
- In `trainer.py`, branch on the flag. (Could even share code with AdamW via a class method swap.)
- Identity at step 0: `m=0`, `s=ε`. First update ≈ `g_0 / √ε` — large. Model output at step 0 (forward) = baseline.
- Why a real lever, not a hyperparam: the *scaling quantity* changes (variance-of-residual vs variance-of-gradient). This isn't reachable by tuning AdamW's eps or betas — it's a different decomposition of the gradient history. Even at 0.94M, the small-batch gradient is noisy, and "do I trust this gradient's direction relative to my running mean?" is a different question than "how big is this gradient?".
- Targets baseline failure: AdamW's `m/√v` can shrink the step on a strong, consistent direction (because `g²` is large), and inflate the step on a noisy, weak direction (because `g²` is small). AdaBelief fixes this swap.

## Scale evidence
Paper trains on CIFAR/ImageNet (CNNs) and Penn TreeBank (RNNs) — not LMs at scale. Independent replications show 1–3% gains on small LMs. 0.94M is below standard LM benchmarks. Transfer risk: med — the lever fires on noise-heavy regimes (small batch, low LR), which is exactly tiny1m3m.

## Why it's worth a slot
Real mechanism, well-cited, and notably absent from our 110–138 optimizer wave. Closest neighbor is 002-cautious-adamw (null at 0.94M), but cautious only masks the *sign* of `m ⊕ g`; AdaBelief changes the *denominator* of the step. A win would tell us variance-of-residual is the right scaling signal at 0.94M; a null would close the "denominator variants" axis and let us focus on numerator/optimizer-structure levers.

## Plan

### Files to change
- **NEW** `optimizers/adabelief.py` — `AdaBelief(Optimizer)` class implementing
  the paper's residual-variance denominator. Mirrors the structure of
  `optimizers/came.py` / `optimizers/radam.py` (per-param `exp_avg` and
  `exp_avg_sq` buffers, fp32 promotion for mixed-precision safety).
  ~60 LoC.
- `optimizers/__init__.py` — add `from .adabelief import AdaBelief` and append
  to `__all__`.
- `configs/llm_config.py` — add `use_adabelief: bool = False` plus
  `adabelief_lr`, `adabelief_beta1`, `adabelief_beta2`, `adabelief_eps`
  tuning knobs to the main `LLMConfig` dataclass (off by default).
- `training/trainer.py` — add one `elif getattr(config, "use_adabelief", False):`
  branch to the AdamW-bucket optimizer selection chain, importing
  `AdaBelief` at the top with the other optimizer imports.

### Flag name
`use_adabelief` (off by default → trainer uses `torch.optim.AdamW` unchanged).

### How it stays zero-init at step 0
At `t=1`: `m_1 = (1−β1)·g_0`, `s_1 = (1−β2)·(g_0 − m_1)² + ε
= (1−β2)·(g_0 − (1−β1)·g_0)² + ε = (1−β2)·β1²·g_0² + ε`. With β1=0.9 the
residual is `0.9·g_0` and `s_1 ≈ 0.081·g_0² + ε`. The update
`m̂_1/√(s_1) = g_0 / √(0.081·g_0² + ε) ≈ g_0 / |g_0|·(1/√0.081) ≈ 3.5·sign(g_0)`
— large but finite. This first-step displacement is the lever's signature
(same magnitude order as AdaShift/CAME, NOT bit-identical to AdamW at
step 0). The forward graph is unchanged, so the *pre-step-0 forward*
output is bit-identical to baseline; only the *first optimizer step*
differs.

### Run command (on Vast V100 box)
```bash
cd /root/universe-lm && /venv/main/bin/python train_llm.py \
  --config_class Tiny1M3MConfig \
  --override "use_adabelief=True" \
  --output_dir /root/universe-lm/runs/141-adabelief-tiny1m3m-42
```

### Reading final val loss
After training completes, parse the last `val_loss` from
`/root/universe-lm/runs/141-adabelief-tiny1m3m-42/log.jsonl`. The
delta vs the baseline `Tiny1M3MConfig` val loss (≈ 6.4216) determines
PASS/DRIFT/FAIL per the `autoresearch/PIPELINE.md` thresholds.
