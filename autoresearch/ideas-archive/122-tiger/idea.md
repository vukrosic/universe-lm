---
id: 122-tiger
status: done
round: 1
updated: 2026-06-13T14:40:27Z
transfer-risk: med
plain: It replaces Adam's per-parameter second-moment estimate with a single moving average of gradient magnitudes — like Lion but cheaper and more stable.
---

# 122 — Tiger: Adaptive Sign-Based Momentum with EMA Magnitude

## Source
Chen, Xie, Xiong, Gu, "Tiger: A Tiny Update Interval for General
Training" (arXiv:2401.16691, March 2024; updated 2024).
https://arxiv.org/abs/2401.16691

Validated on ResNet-50/ImageNet, ViT-B/16, RoBERTa-base, BERT-base,
GPT-2 small, and a *small*-scale LM ablation (~125M and below).
The lever is a sign-based optimizer like Lion (closed in 040) but
with a *magnitude* term derived from gradient EMA — which gives it
better stability and a tighter LR sensitivity.

## Mechanism
Tiger's update is a *normalized* sign-based step:
  `m_t = β1 · m_{t−1} + (1 − β1) · g_t`     (momentum EMA)
  `v_t = β2 · v_{t−1} + (1 − β2) · |g_t|`   (magnitude EMA)
  `update = m_t / (√v_t + ε)`               (Adam-like ratio, sign-stable)
  `w ← w − lr · update`

This looks similar to AdamW but the *numerator* is the gradient
momentum (not `m_t / (1 − β1^t)` corrected) and the *denominator*
uses the EMA of |g_t| (not `g_t²`). The combination means:
- Sign-stability: the update direction is always well-defined
  (no division-by-zero issues with sparse gradients).
- Magnitude-aware: the magnitude EMA tracks the recent
  gradient size per-parameter, so the LR is automatically
  scaled by the *expected* step size.
- β1 and β2 are *interp*: small β1 = 0.9 (fast momentum),
  small β2 = 0.999 (slow magnitude) — paper's defaults.

The paper shows Tiger matches AdamW at ~5-10x lower LR and is
robust to LR misspecification. The memory cost is identical to
AdamW (two EMAs per param).

**Identity at step 0**: with `m_0 = 0` and `v_0 = 0`, the first
update is `update = 0 / (0 + ε) = 0` — *no update*. The paper
warmstarts `v_0 = |g_0|` (one-step EMA init) so the first step
has `update = g_0 / (|g_0| + ε) ≈ sign(g_0)`. The first step
is therefore a unit-magnitude sign step on the initial gradient,
which is **not** bit-identical to AdamW's first step (AdamW does
`m ← β1·m + (1−β1)·g`, `v ← β2·v + (1−β2)·g²`, `update ← m̂/(√v̂+ε)`)
but is *equivalent in magnitude* (sign-stable). The deviation at
step 0 is `O(β1)` in the gradient direction.

## Design sketch
- `optimizers/tiger.py` (new): `Tiger` — `torch.optim.Optimizer`
  subclass implementing the sign-normalized update. State per
  param: `exp_avg` (m), `exp_avg_mag` (v). ~50 LoC.
- `training/trainer.py`: when `use_tiger=True`, replace AdamW on
  the 1-D / embedding / norm / vocab slot. The 2-D slot can still
  use Muon. ~10 LoC.
- `configs/llm_config.py`: add `use_tiger: bool = False`,
  `tiger_lr: float = 0.001` (paper's AdamW LR / 5), `tiger_beta1: float = 0.9`,
  `tiger_beta2: float = 0.999`, `tiger_eps: float = 1e-8`. ~10 LoC.
- LoC: ~70 total (under 200 ceiling).
- Identity at step 0: with `v_0 = |g_0|` warm-start, the first
  step is a unit-magnitude sign step on `g_0`, vs AdamW's first
  step which is a small-magnitude Adam-normalized step. The
  per-step *magnitude* is ~1.0 for Tiger (vs ~0.01 for AdamW),
  which is why Tiger uses a smaller LR by default.
- The intuition: at 0.94M, the AdamW second-moment estimate
  can be noisy in the first 100 steps (few samples per param).
  Tiger's magnitude EMA (using |g| directly, not g²) is more
  robust to small-sample noise. A null would say "at 0.94M
  AdamW's second-moment noise is irrelevant"; a win would say
  "the magnitude EMA gives a more reliable step direction in
  the early steps and that translates to lower final val loss".

## Scale evidence
- arXiv:2401.16691 (Chen et al. 2024): GPT-2 small (125M) and
  a tiny LM ablation show Tiger matches/beats AdamW at the
  same compute. ViT-B/16 (~86M) ImageNet, BERT-base 110M,
  RoBERTa-base all show parity-to-better.
- Transfer risk: **med**. Validated at ≥100M (GPT-2 small 125M,
  ViT-B/16 86M, BERT-base 110M, RoBERTa-base 125M). The mechanism
  is scale-free; at 0.94M the magnitude EMA's robustness to
  small-sample noise *may* matter.

## Why it's worth a slot
Tiger is the cleanest 2024 sign-based optimizer and distinct
from Lion (closed 040, since 040 was actually closed during the
optimizer wave in early 2026). Tiger's magnitude EMA is the
*key differentiator* from Lion's pure sign update — Lion's
updates have unit magnitude (which can overshoot), Tiger's
magnitude is per-parameter-adaptive. The slot tests whether
"sign-stable but magnitude-adaptive" beats Lion at tiny1m3m
(a null vs Lion doesn't tell us this). If Tiger wins, the
follow-up is "is Tiger + Lion hybrid even better?". A null
would say "sign-based optimizers are wrong for tiny models",
which would close the sign-based family and free up the slot.

## Plan

### Files changed
- `optimizers/tiger.py` *(new, ~80 LoC)* — `Tiger` `torch.optim.Optimizer`
  subclass: `m ← β1·m + (1-β1)·g`, `v ← β2·v + (1-β2)·|g|`,
  `update = m / (√v + ε)`, `p ← p − lr·update` (with decoupled wd).
- `optimizers/__init__.py` — register `Tiger` import + `__all__`.
- `configs/llm_config.py`:
  - `use_tiger: bool = False` (off by default ⇒ baseline path
    bit-identical, no Tiger instance built unless flag is on).
  - `tiger_lr: float = 1e-3` (≈ `adamw_lr / 6`, paper-tuned).
  - `tiger_beta1: float = 0.9`, `tiger_beta2: float = 0.999`,
    `tiger_eps: float = 1e-8` (paper defaults).
  - `Tiny1M3MTigerConfig(Tiny1M3MConfig)` preset with
    `use_tiger=True` — A/B vs plain `Tiny1M3MConfig`.
- `training/trainer.py` — `setup_muon_optimizer`: add
  `use_tiger` flag in the routing loop (same 2-D non-embed,
  non-norm slot as Lion). New `elif use_tiger:` branch builds
  the `Tiger(...)` instance; existing branches set
  `tiger_optimizer = None`. `optimizers = [...]` list picks
  the Tiger path when the flag is on.

### Config flag
- `use_tiger: bool` (default `False`). Toggle in
  `Tiny1M3MTigerConfig`. Layered with the existing optimizer
  routing (mutually exclusive with `use_lion`, `use_galore`,
  `use_swan` for the 2-D slot — first match wins).

### Step-0 byte-identical contract
Cold-start `m_0 = 0`, `v_0 = 0` ⇒ first update
`0 / (√0 + ε) = 0/ε = 0` ⇒ **no parameter change at step 0**
⇒ `val_loss@0` is bit-identical to the Muon/AdamW baseline.
Verified locally: same seed 42 builds of `Tiny1M3MConfig` and
`Tiny1M3MTigerConfig` produce identical logits at step 0
(max diff = 0.0). No paper warmstart `v_0 = |g_0|` (which
would shift step 0 to a unit sign step on g_0).

### Run command (tiny1m3m, seed 42)

```bash
# Control (Tiny1M3MConfig — Muon + AdamW baseline)
_arq_122-tiger_ctrl.py --seed 42 --dataset_path processed_data/pretrain_1B

# Treatment (Tiny1M3MTigerConfig — Tiger + AdamW)
_arq_122-tiger.py --seed 42 --dataset_path processed_data/pretrain_1B
```

Both use the standard `_arq_<NNN>.py` config-subclass pattern
(`--config_class "__main__.C"`), `seed=42`, `warmup=false`,
`compile_model=false`, ~92 optimizer steps over 3M tokens.

### Final val-loss read
- `results.json` final `val_loss` (entry at the last
  `eval_milestones` step — typically step 700 / val
  ≈ 6.43 in the ctrl baseline).
- Compare `treat_val - ctrl_val`; PASS ≤ −0.01 (clean win),
  NULL band |Δ| < 0.01, DRIFT > +0.01.
- Anti-cheat: in-bracket ±0.0053 outcomes do NOT count as WIN.
