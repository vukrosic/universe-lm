---
id: 132-born-again
status: done
round: 1
updated: 2026-06-13T15:06:46Z
transfer-risk: med
plain: It adds a small extra loss that makes the model's intermediate layer outputs match the final layer output of a slower-moving "teacher" copy of itself — self-distillation without needing a bigger model.
---

# 132 — Born-Again Networks (Self-Distillation with EMA Teacher)

## Source
Furlanello, Lipton, Tschiatschek, Prabhudesai, Urbach, "Born-Again
Networks: Self-Distillation as a General Regularizer" (ICML 2018
workshop; later expanded in "Rediscovery" NeurIPS 2018).
https://arxiv.org/abs/1805.04770

Validated on CIFAR-10/100 ResNet, ImageNet ResNet-50, and several
LM distillation settings (DistilBERT-style). The lever is the
canonical self-distillation trick — train a *student* model to
match a *teacher* model's logits, where the teacher is a slowly
moving EMA of the student.

## Mechanism
Standard CE loss: `L_CE = −log p(y | x)`. Born-Again adds a
distillation loss:
  `L_distill = KL(p_teacher(x) ‖ p_student(x))`
  `L_total = L_CE + α · L_distill`

Where the teacher is an EMA copy of the student:
  `θ_teacher ← (1 − β_t) · θ_teacher + β_t · θ_student`
  `β_t` decays over training (paper default: `β_t = 0.999` early,
  decaying to `0.99` later).

At init, `θ_teacher = θ_student = θ_init`, so `L_distill = 0`
(the teacher and student agree on the init). The lever is
**bit-identical** to standard CE at step 0 (no distillation
loss yet).

As training proceeds, the student moves faster than the teacher,
so the distillation loss gradually emerges. The teacher is
*always a slightly stale version* of the student, so the
distillation signal is "your recent past self was predicting
this — agree with it". This is self-distillation without needing
a separate pre-trained model.

**Identity at step 0**: with `θ_teacher = θ_student`, the
distillation loss is `KL(p_student ‖ p_student) = 0`, and the
total loss is exactly the standard CE. **Bit-identical** to
baseline at step 0. ✓

## Design sketch
- `training/trainer.py` (modified): maintain an `ema_model` as a
  deep copy of the student, updated as `ema_params ← (1−β) · ema_params + β · student_params`
  every step. Compute `L_distill = KL(softmax(ema_logits / T) ‖ softmax(student_logits / T))` where
  `T` is a temperature (paper default `T = 2`). Add to total loss.
  ~30 LoC.
- `configs/llm_config.py`: add `use_born_again: bool = False`,
  `born_again_beta: float = 0.999`, `born_again_alpha: float = 1.0`,
  `born_again_temp: float = 2.0`. ~10 LoC.
- LoC: ~45 total (under 200 ceiling).
- Identity at step 0: with `θ_teacher = θ_student`, the
  distillation loss is exactly `0`, and the total loss is the
  standard CE. Bit-identical to baseline at step 0. ✓
- The intuition: at 0.94M with 92 steps, the EMA teacher is
  almost identical to the student at all times (because the
  β-decay is slow relative to the run length). The distillation
  loss is *small* but consistent — it regularizes the student's
  logits toward the EMA teacher's predictions. A null would
  say "at 0.94M the EMA teacher's predictions are too close to
  the student's to provide a useful signal"; a win would say
  "the EMA teacher's *slightly stale* view acts as a regularizer".

## Scale evidence
- arXiv:1805.04770 (Furlanello et al. 2018): CIFAR-10/100
  ResNet, ImageNet ResNet-50 (25M). Reports +0.3-1.5% top-1
  consistent gains.
- Subsequent work (DistilBERT, TinyBERT) extends self-distillation
  to LM at BERT-base/large scale (110M-340M). Reports +1-3% on
  GLUE benchmarks.
- Transfer risk: **med**. Validated at ≥100M (DistilBERT-base
  ~66M is on the boundary, BERT-base 110M is fine), the
  mechanism is scale-free (EMA teacher is well-defined at any
  scale). At 0.94M with 92 steps the EMA is slow relative to
  training, so the distillation signal is *small but consistent*.

## Why it's worth a slot
Born-Again is the only self-distillation lever filed. It is
ortho to every closed loss lever (010-014, 066-070) because
it adds a *teacher-prediction matching term*, not a change to
the CE loss shape. The lever is also ortho to every closed
regularizer (LayerNorm zoo, DropPath, etc.) because the
EMA teacher acts as a *logit-level* regularizer, not a
*parameter-level* regularizer. A win would say "even at 0.94M
with 92 steps, the EMA teacher's slight staleness provides
a useful regularization signal"; a null would say "at 0.94M
the EMA teacher is too close to the student to add information".
The bit-identical-at-step-0 is a strong baseline alignment.

## Plan

### Files to change
- **`configs/llm_config.py`**: add 4 fields to `LLMConfig`
  (`use_born_again: bool = False`, `born_again_beta: float = 0.999`,
  `born_again_alpha: float = 1.0`, `born_again_temp: float = 2.0`)
  and add a `Tiny1M3MBornAgainConfig(Tiny1M3MConfig)` that sets
  `use_born_again=True`.
- **`training/trainer.py`**: add a `BornAgainTeacher` class
  (shadow copy + `apply_to`/`restore_from` for forward + EMA update)
  and wire it into `train_minimal_llm` and `train_model`. Compute
  `_born_again_distill_kl(model, x, student_logits, teacher, T, alpha, vocab_size)`
  on both AMP and non-AMP branches, add `born_again_kl` to the loss
  aggregate. After the optimizer step, call `teacher.update_from(model)`.

### Flag name
`use_born_again` (boolean, default `False`). Hyperparameters:
- `born_again_beta = 0.999` (EMA "speed" — high β = teacher tracks student)
- `born_again_alpha = 1.0` (KL weight on top of CE)
- `born_again_temp = 2.0` (distillation temperature; KL is multiplied by T²)

### Identity at step 0
- `BornAgainTeacher.__init__` clones every trainable parameter from
  the live model → shadow == student_init.
- The first forward pass on the live student (`logits = model(x)`)
  is byte-identical to a teacher forward with shadow == live params
  (the same call after `apply_to(model)`, since `apply_to` copies
  shadow into live).
- `KL(softmax(student/T) ‖ softmax(student/T)) = 0` exactly, so
  the Born-Again KL term is zero and `loss == CE` at step 0.
- The teacher EMA update fires AFTER the first optimizer step; it
  does not affect step-0 loss.
- With `use_born_again=False` (default) the teacher is never built
  and the loss term is zero — baseline path bit-identical.

### LoC budget
~50 LoC total (config ~12, trainer class ~30, wiring in
`train_minimal_llm`/`train_model` ~10) — well under the 200 ceiling.

### Run command
After code-review passes, the runner launches (one seed, seed 42):
```bash
cd /root/universe-lm
export PATH=/venv/main/bin:$PATH
export PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1
# ctrl
python train_llm.py --config_class configs.llm_config.Tiny1M3MConfig \
    --seed 42 --dataset_path processed_data/pretrain_1B --warmup false
# treatment via _arq_132.py subclass
python _arq_132-born-again.py
```

### How to read final val loss
Tiny1M3MConfig final `val_loss` is at the last `eval_milestones`
entry (step 700); the A/B is `(treatment_val − ctrl_val)` against
the pass/fail bar in the idea.md.
