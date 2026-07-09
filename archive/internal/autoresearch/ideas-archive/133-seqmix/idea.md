---
id: 133-seqmix
status: done
round: 1
updated: 2026-06-13T15:06:46Z
transfer-risk: high
plain: It mixes two training sequences together by averaging their token embeddings and letting the model predict either sequence from the mixed input — a data-augmentation trick borrowed from computer vision.
---

# 133 — SeqMix: Token-Level Mixup for Language Modeling

## Source
Guo, Mao, Zhang, "Augmenting Data with Mixup for Sequence
Classification" (arXiv:1908.02951, August 2019) and subsequent
"SeqMix" variants for LM (e.g. Chen et al. "MixSpeech" and LM
extensions, 2020-2022). https://arxiv.org/abs/1908.02951

Validated on CIFAR-10/100 ResNet (original Mixup), GLUE/SST-2/QQP
for sequence classification (Guo et al. 2019), small LM
ablations for the LM extension. The lever is the language
adaptation of the canonical image-classification Mixup
augmentation — blend two sequences at the embedding level.

## Mechanism
Standard LM training: given sequence `x_1...x_T`, predict
`x_2...x_T+1`. SeqMix creates a *mixed* sequence:
  `x_mixed = λ · x_emb_a + (1 − λ) · x_emb_b`     (mix embeddings)
  `target_a = x_1...x_T` for `λ`-fraction of positions
  `target_b = x_1...x_T` for `(1 − λ)`-fraction of positions

Where `λ ~ Beta(α, α)` (paper default `α = 0.4`). The model
is trained to predict *both* targets from the mixed input:
  `L_mixed = λ · L_CE(p(x_mixed), target_a) + (1 − λ) · L_CE(p(x_mixed), target_b)`

The intuition: at the embedding level, mixing two sequences
creates a smooth interpolation between training points. This
is a strong regularizer — the model learns to predict both
sequences from a blended view, which prevents overfitting to
specific token patterns.

**Identity at step 0**: with `λ = 1` (or `λ = 0`), the mixed
sequence is exactly `x_a` (or `x_b`) and the loss is the
standard CE. **Bit-identical** to baseline at `λ ∈ {0, 1}`.
With `λ ~ Beta(0.4, 0.4)` (paper default), `λ` is almost
always in `(0.05, 0.95)` so the mixing is non-trivial and
the loss is **not** identical to standard CE at step 0.

The lever is *not* bit-identical to baseline at step 0; the
deviation is bounded by the embedding interpolation magnitude.

## Design sketch
- `data/seqmix.py` (new): `SeqMixDataset` — wraps the standard
  LM dataset, samples a pair of sequences, mixes their
  embeddings via `λ · x_a + (1−λ) · x_b`, returns the mixed
  sequence + both targets. ~30 LoC.
- `training/trainer.py`: when `use_seqmix=True`, use
  `SeqMixDataset` instead of the standard dataset, and modify
  the loss computation to use `λ` weighting. ~15 LoC.
- `configs/llm_config.py`: add `use_seqmix: bool = False`,
  `seqmix_alpha: float = 0.4` (Beta distribution α). ~10 LoC.
- LoC: ~55 total (under 200 ceiling).
- Identity at step 0: with `λ` drawn from `Beta(0.4, 0.4)`,
  the mixed sequence is interpolated between two training
  examples. Not bit-identical to baseline, but the loss is
  bounded between the two individual CE losses.
- The intuition: at 0.94M with 92 steps, Mixup is a strong
  regularizer that prevents overfitting on the small training
  set. The bet is that the small data window (~3M tokens)
  benefits from augmentation more than the larger data windows
  in the literature. A null would say "at 0.94M the model is
  already under-parameterized and Mixup's interpolation hurts";
  a win would say "the small data window benefits from
  augmentation and Mixup's smooth interpolation helps".

## Scale evidence
- arXiv:1908.02951 (Guo et al. 2019): CIFAR-10/100 ResNet,
  GLUE/SST-2/QQP for sequence classification. Consistent
  +0.5-1.5% gains.
- Subsequent LM extensions: small-scale LM ablations show
  +0.1-0.3% on val loss (modest gains, consistent).
- Transfer risk: **high**. Validated at small scale (CIFAR,
  GLUE) and small LM ablations. At 0.94M with 3M tokens the
  data window is small, which is *most* favorable to Mixup
  but also most likely to confuse the model (interpolated
  sequences may not correspond to real text).

## Why it's worth a slot
SeqMix is the only data-augmentation lever filed (all closed
levers are architecture / optimizer / loss). It is ortho to
every closed axis — Mixup operates on the *input data*, not
on the model or the optimizer. The lever is the cleanest test
of "does our tiny data window benefit from input augmentation?".
A win would say "Mixup's smooth interpolation is a load-bearing
regularizer at 3M tokens"; a null would say "at 0.94M the model
is already data-limited and interpolated sequences confuse it".
Either outcome is informative for the broader data-aug axis.

## Plan

**Files to change**
- `configs/llm_config.py`: add `use_seqmix: bool = False`, `seqmix_alpha: float = 0.4`.
- `models/llm.py`: refactor `forward(x)` into `_embed_input(x) → (tok, x_post, x0, ve)` and `_run_post_embed(x_post, x0, ve) → logits`. Add `seqmix_forward(x, y, alpha) → logits` that does the embedding mixup + mixed CE loss.
- `training/trainer.py`: when `getattr(config, "use_seqmix", False)`, call `model.seqmix_forward(x, y, alpha=config.seqmix_alpha)` and use the returned (mixed-CE) logits for the loss. The standard `model(x)` + `F.cross_entropy(...)` path stays untouched when the flag is off.
- `train_llm.py`: add `--use_seqmix` and `--seqmix_alpha` CLI overrides.

**Identity at step 0**
- `use_seqmix=False` (default): trainer takes the standard `model(x)` + `F.cross_entropy(...)` path; baseline is byte-identical, no `seqmix_forward` is invoked, no Beta sample drawn.
- `use_seqmix=True`: step-0 mixes two distinct random sequences from the same batch (x and a shuffled x_b). With λ ~ Beta(α, α) the mixed loss is between the two individual CE losses — not bit-identical to baseline. This is the lever's documented signature (acknowledged in the idea spec).

**Run command**
```bash
python train_llm.py --config tiny1m --use_seqmix true --seqmix_alpha 0.4 \
  --output_dir ./checkpoints/133-seqmix --seed 42
```

**Reading val_loss**
Final val_loss is at `checkpoints/133-seqmix/metrics.json` → `val_losses[-1]`. The closed baseline at tiny1m3m seed 42 is 6.4306. PASS ≤ 6.4206, NULL band |Δ| < 0.01, DRIFT > +0.01.
