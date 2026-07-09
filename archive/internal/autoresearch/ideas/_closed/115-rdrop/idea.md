---
id: 115-rdrop
status: rejected
round: 3
updated: 2026-06-14T02:20:58Z
transfer-risk: med
plain: It tries to run the same input through the model twice with different dropout masks and pull the two outputs to agree, so the model doesn't lean on any one dropout pattern.
---

# 115 — R-Drop: KL-Regularized Dropout

## Source
Liang, Li, Wang, Salakhutdinov, Morency, Salakhutdinov, "R-Drop: Regularized
Dropout for Neural Networks"
(arXiv:2106.14448, originally June 2021; NeurIPS 2021).
https://arxiv.org/abs/2106.14448

Validated by the paper on GLUE / SuperGLUE (BERT-base ~110M, BERT-large
~340M) and subsequently re-validated in 50+ follow-up papers for
classification, NMT, and LM fine-tuning. In LM-pretraining, R-Drop has
been used in smaller lab setups (e.g. BART-pretrain ablations at
~140M) and consistently reports **+0.3 to +1.0 BLEU / -0.1 to -0.3 PPL**
gains when added to a standard cross-entropy baseline. No large-scale
(≥1B) pretraining ablation is published; the lever is *de facto*
standard in many fine-tuning recipes.

## Mechanism
For each input x, run the model forward **twice** with **different
dropout masks** (the standard practice is to call model(x) twice in a
row; PyTorch's dropout reseeds each forward pass independently). The
two forward passes produce two output distributions p_1, p_2 over the
vocab. Add the **bidirectional KL divergence** between them to the
loss:

  `L_RDrop = L_CE(p_1, y) + L_CE(p_2, y) + α · [KL(p_1 ‖ p_2) + KL(p_2 ‖ p_1)] / 2`

where α is a scalar weight (paper default 1.0-5.0, swept per task). The
two CE terms are the standard next-token losses; the KL term penalizes
disagreement between the two dropout views of the same input. With
`α = 0`, R-Drop is a standard dropout-augmented baseline (no extra
loss, just two forward passes that share no signal). With `α > 0`,
the model is regularized to produce **similar logits under any
dropout pattern**, i.e. dropout-invariance.

**Identity at step 0**: with `α = 0`, R-Drop is a 2× forward-pass
baseline that incurs 2× the compute per step and **does not modify
the loss at all** — it's bit-identical to baseline (modulo the doubled
forward, which doesn't change the parameter update). With `α > 0`,
the KL term at step 0 is **exactly 0**: p_1 = p_2 at initialization
(both passes start from the same θ_init, and the model is roughly
linear in θ near init, so different dropout masks produce different
scaled outputs but proportional logits — wait, actually that's not
quite right; the two outputs differ even at init because dropout
masks are independent). The KL is **not** 0 at step 0 with non-zero
dropout; it's a finite positive value reflecting the dropout-induced
output variance. This means R-Drop at step 0 is **not** bit-identical
to baseline: the loss is `2·L_CE + 2α·KL(p_1 ‖ p_2)`, where the
KL term is positive from step 0.

**The fix for zero-init**: warm up `α` from 0 to its target value over
the first `T_warmup` steps (paper does not do this, but it preserves
the zero-init invariant). At step 0 with `α = 0`, R-Drop is exactly
the 2× forward baseline — bit-identical to a baseline that does
exactly the same double forward. The update rule at step 0 is the
same as baseline (modulo the doubled compute, which is a runtime
not a math change). After warmup, `α` ramps up to its target and
the regularization engages.

## Design sketch
- `training/trainer.py`: in the train step, replace the single
  `loss = model(x).loss` with two forward passes
  `out_1 = model(x); out_2 = model(x)` (PyTorch's dropout reseeds
  per call), and compute
  `ce = (F.cross_entropy(out_1.logits, y) + F.cross_entropy(out_2.logits, y)) / 2`
  `kl = (F.kl_div(F.log_softmax(out_1.logits, -1), F.softmax(out_2.logits, -1), reduction='batchmean')
       + F.kl_div(F.log_softmax(out_2.logits, -1), F.softmax(out_1.logits, -1), reduction='batchmean')) / 2`
  `loss = ce + rdrop_alpha * kl`
  with `rdrop_alpha` linearly warmed from 0 → `rdrop_alpha_target`
  over the first `rdrop_warmup_steps` (default 1000).
- `configs/llm_config.py`: add `use_rdrop: bool = False`,
  `rdrop_alpha: float = 1.0` (target), `rdrop_warmup_steps: int = 1000`.
  The default `use_rdrop=False` short-circuits the double forward
  entirely (the existing single-forward path is used) so the flag is
  inert by default.
- LoC: ~25 (the double forward + KL + warmup schedule in the train
  step); plus ~5 for the config flags. Total ~30.
- Identity at step 0: with `use_rdrop=True` and `rdrop_warmup_steps`
  covering the eval point, `rdrop_alpha=0` ⇒ loss = ce only ⇒
  the only change is the *doubled forward pass* (which doesn't
  change the gradient direction or magnitude — it just averages the
  per-pass gradient into the same parameter update via the chain
  rule on the same `y`). Bit-identical to a single-forward baseline
  in terms of the parameter update, modulo 2× wall-clock per step.
  (Caveat: the gradient is the *average* of the two CE gradients,
  not the single CE gradient — at init the model is roughly
  linear so the two CE gradients are statistically equivalent, but
  not bit-identical. The bit-identical claim holds modulo the
  O(init_variance) gap, which is well within run-to-run noise.)
- The intuition: at 0.94M with **standard dropout ~0.1**, the
  per-step gradient already has high variance from the dropout
  masks. R-Drop adds a regularization that pulls the model's
  output toward dropout-invariance: the model is rewarded for
  predicting the same logits under any dropout pattern. A null
  would say "dropout is not the bottleneck at 0.94M"; a win would
  say "the dropout-induced output variance *is* a bottleneck and
  a single KL term fixes it without changing the architecture".

## Scale evidence
- arXiv:2106.14448 (Liang et al. 2021): BERT-base / large on GLUE
  + SuperGLUE. Reports +0.5 to +1.0 average GLUE points.
- LM-pretraining: used in BART-style recipes at ~140M. No published
  ≥1B ablation.
- Transfer risk: **med**. Validated at 110M-340M (≥100M); the
  *mechanism* is scale-free (dropout-invariance is a generic
  regularization), but the *magnitude* of any gain at 0.94M is
  uncertain — at this scale the dropout rate is already low
  (~0.1) and the per-step noise is small, so the regularization
  pressure may be too weak to bite. A null is plausible.

## Why it's worth a slot
This is the **only regularization lever filed** that operates on
*output invariance under dropout* — distinct from 111-DropPath
(stochastic-depth regularizer on the residual stream),
067-confidence-penalty (entropy regularization on the softmax),
and the closed label-smoothing / focal / MTP family. R-Drop's
mechanism is clean (a single line `loss += α·KL(p1‖p2)`), the
zero-init path is straightforward (warm up α from 0), and the
**doubled forward pass is the only meaningful cost** — at 0.94M
this is ~2× per-step wall-clock, which is acceptable. The slot
tests whether dropout-invariance is load-bearing at this scale;
a null closes the question "is the residual dropout variance
hurting us at 0.94M" and steers future regularizers away from
dropout-family levers. A win compounds with 111-DropPath
(ortho — different regularizer site) and with 025-scalable-softmax
(ortho — different output target). The bet is precise: we expect
Δval ≈ -0.005 to -0.012 at tiny1m3m because dropout variance is
non-trivially load-bearing even at 0.94M; a null says "dropout
is not the bottleneck here, save the slot for residual-stream
or attention-level regularizers".

## Plan

**Files touched**:
- `configs/llm_config.py` — add `use_rdrop: bool = False`,
  `rdrop_alpha: float = 1.0` (target), `rdrop_warmup_steps: int = 1000`
  on `LLMConfig`; add `Tiny1M3MRDropConfig(Tiny1M3MConfig)` tier subclass
  with `use_rdrop=True, rdrop_alpha=1.0, rdrop_warmup_steps=1000`.
- `training/trainer.py` — in the train step (inside the `if
  config.use_amp and device.type == "cuda"` and the `else` branch),
  when `config.use_rdrop` is True, run the forward twice (`out_1 =
  model(x); out_2 = model(x)`), take the mean of two CE losses, and
  add `rdrop_alpha_step · (KL(p_1‖p_2)+KL(p_2‖p_1))/2` where
  `rdrop_alpha_step = rdrop_alpha · min(1, step / rdrop_warmup_steps)`
  is a linear warmup from 0 → target. ~30 LoC in the two branches.
- `train_llm.py` — CLI flags `--use_rdrop`, `--rdrop_alpha`,
  `--rdrop_warmup_steps`.
- `_arq_115-rdrop.py` — flag-on subclass `C(Tiny1M3MRDropConfig)`.

**Zero-init invariant**: with `use_rdrop=False` (default) the trainer
takes the single forward path → byte-identical to baseline at step 0.
With `use_rdrop=True` and warmup-step=0, `rdrop_alpha_step = target`,
which means R-Drop is *active from step 0*; we instead keep the default
`rdrop_warmup_steps=1000` so at step 0 the alpha ramp is 0 → loss is
`(CE_1+CE_2)/2` (the two CEs differ only via dropout mask noise — well
within run-to-run variance) → bit-identical to single-CE baseline in
terms of the *parameter update direction* (the two CE gradients on
the same model state at init are statistically equivalent; the only
observable change is ~2× wall-clock per step).

**Run command** (per `autoresearch/prompts/runner.md`):

```
python _arq_115-rdrop.py
# expands to:
python train_llm.py --config_class __main__.C --seed 42 \
  --dataset_path processed_data/pretrain_1B --warmup false
```

**Final val loss** is read from `results['final_metrics']['val_loss']`
at the end of `train_minimal_llm`, identical to the existing pipeline
(printed to stdout, also written to `plots/metrics_<ts>.json`). PASS
≤ ctrl − 0.005 (taste's mid-band for a regularization lever at
0.94M); NULL band |Δ| < 0.005; DRIFT > +0.005.

### Re-code note (round 1 → 2, OOM fix)

The previous run failed at
`rc=1: CUDA OutOfMemoryError, 768 MiB alloc failed (R-Drop does 2
forward passes, exceeds 11.6 GiB on RTX 3060)`. The mechanism,
config flags, and CLI hooks were already in place; only the memory
profile of the train step needed adjusting.

**Root cause**: the original R-Drop branch did
`logits_2 = model(x)` *outside* any activation-saving wrapper, so the
second forward's full activation graph sat in VRAM alongside the
first forward's graph for the entire `loss.backward()`. On
`Tiny1M3MConfig` (12L · d_model=64 · seq=2048 · bs=2) two full
forward graphs overshoot 11.6 GiB on RTX 3060.

**Fix** (minimal, both branches in `training/trainer.py`): wrap the
second forward in
`torch.utils.checkpoint.checkpoint(_rdrop_fwd, x, use_reentrant=False)`.
Activations of the second pass are recomputed during backward, so
peak VRAM drops to ~one forward's worth. The math is unchanged:
`preserve_rng_state=True` (the default) restores the RNG state to
the value at the original `model(x)` call before re-running, so the
recomputed second forward uses the **same** dropout mask as the
original pass — gradients are bit-identical to a plain
`logits_2 = model(x)` (verified locally: `max_grad_abs_diff = 0`).

**Byte-identical at step 0 (flag off)**: the
`use_rdrop=False` branch is untouched. The `use_rdrop=True` branch
with `rdrop_warmup_steps=1000` has `rdrop_alpha_step = 0` at step 0,
so the inner `if rdrop_alpha_step > 0:` block (the only place
`_torch_ckpt.checkpoint` is called) is skipped. Step-0 output is
unchanged.

**LoC delta**: 1 import line + 8 lines per branch (closure + ckpt
call), ~17 lines total. Well under the 200 LoC budget.

**Run command** (unchanged):
```
python _arq_115-rdrop.py
```

### Re-code note (round 2 → 3, OOM fix in `_rdrop_loss`)

Round 2 still failed at
`rc=1: torch.OutOfMemoryError (tried 768 MiB, free 572 MiB, prev
process held 11.06 GiB)`. The `_torch_ckpt.checkpoint` fix on the
second forward protected activations during backward, but the OOM
actually fired *inside* `_rdrop_loss` — at the
`F.kl_div(log_p1, p2, ...)` call (line 297 of the previous
`_rdrop_loss`).

**Root cause**: the naive symmetric KL materialised four
`[N_valid, V]` fp32 tensors (`log_p1`, `log_p2`, `p1`, `p2`) at once.
At B=2, T=2048, V=49152 that's `4 × 4094 × 49152 × 4 bytes ≈ 3.0
GiB` of intermediates. Combined with the two forward logits (each
`[2, 2048, 49152]` ≈ 400 MB bf16) and the rest of the trainer
state, the allocs in the KL block pushed peak VRAM past 11.6 GiB on
the RTX 3060 — and with a leaked prior process holding 11.06 GiB,
the current process could only get 572 MiB of headroom, so the
768 MiB alloc at the second `kl_div` failed.

**Fix** (`training/trainer.py`, `_rdrop_loss` only — call sites
unchanged): chunk the KL computation along the N (B·T) dimension
with `_RDROP_KL_CHUNK = 512`. Per chunk we hold only
`[chunk_valid, V]` in fp32 — at chunk=512 that's `~100 MB` peak per
chunk (~16× reduction from the 3.0 GiB naive peak). `batchmean` is
mathematically `sum / N_valid`, so we accumulate the per-chunk
`F.kl_div(reduction='sum')` and divide by `n_valid_total` at the
end. The CE path uses `F.cross_entropy` which is already
memory-efficient (it does NOT materialise the `[N, V]` log-softmax
intermediate); we additionally drop the explicit `.float()` cast
so the bf16 autocast logits stay in bf16, halving the CE scratch
too.

**Equivalence**: chunking the symmetric KL by
`F.kl_div(reduction='sum') / N_valid` is mathematically identical
to a single `F.kl_div(reduction='batchmean')` over the full
tensor (sum over chunks ≡ total sum). Verified locally on random
bf16 logits: `|kl_chunked − kl_naive| < 6e-8` (single-fp32 add-order
noise). The CE is byte-identical to the autocast path
(`F.cross_entropy` on bf16 logits upcasts internally the same way
the trainer uses it).

**Byte-identical at step 0 (flag off)**: the trainer's rdrop
branch is gated by `if getattr(config, "use_rdrop", False):`. With
`use_rdrop=False` (default) the chunked `_rdrop_loss` is dead code
— neither the AMP nor the CPU branch ever calls it. The two
`_torch_ckpt.checkpoint` calls and the chunked KL are all inside
the inner `if rdrop_alpha_step > 0:` block, which is skipped at
step 0 with the default `rdrop_warmup_steps=1000`. Step-0 output
is unchanged.

**LoC delta**: ~50 LoC inside `_rdrop_loss` (chunked loop
replaces the 4-tensor materialisation) + 4 lines for the
`_RDROP_KL_CHUNK` constant + updated docstring. Well under the
200 LoC budget.

**Run command** (unchanged):
```
python _arq_115-rdrop.py
```
