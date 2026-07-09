---
id: 117-soft-moe
status: done
round: 1
updated: 2026-06-13T14:31:10Z
transfer-risk: med
plain: It replaces the usual "pick one expert per token" routing with a smoother blending where every token is a mixture sent to every expert, so gradients flow cleanly to all of them.
---

# 117 — Soft MoE: Fully-Differentiable Mixture of Experts in FFN

## Source
Puigcerver, Riquelme, Mustafa, Houlsby, "From Sparse to Soft Mixtures
of Experts" (arXiv:2406.06589, June 2024; ICLR 2025).
https://arxiv.org/abs/2406.06589
The paper validates Soft MoE on ViT-L (~300M params, ImageNet-21k)
and on a 1.3B-param LM (C4). Reported gains: −0.05 to −0.15 nats
loss vs top-1 / top-2 sparse MoE at matched FLOPs. The key property
is *full differentiability* — no top-k, no load-balancing loss, no
auxiliary router — making Soft MoE the simplest possible MoE
extension to implement and tune.

## Mechanism
Standard sparse MoE: each token picks top-1 (or top-2) expert, only
that expert processes the token. Routing decisions are non-differentiable
under top-k, requiring REINFORCE / straight-through / load-balancing
losses to train the router.

Soft MoE replaces the hard routing with two learned **slot matrices**:
  `D ∈ R^{(n_tokens × m) × n_experts}` (dispatch weights, normalized
   over the (tokens·slots) axis so each expert sees m slots worth of
   token-mixture)
  `C ∈ R^{n_experts × (n_tokens × m)}` (combine weights, normalized
   over the experts axis so each output token is a convex combination
   of expert outputs)

Concretely:
  `X̃ = D · X`                                # X ∈ R^{N×d}, X̃ ∈ R^{(m·E)×d}
  `for e in experts:  Y_e = f_e(X̃_e)`        # each expert sees its slot rows
  `Y = Σ_e  C_e · Y_e`                        # recombine per output token

`D` and `C` are learned via `softmax` along the appropriate axis
(no argmax, no top-k). With `m` slots per token, each expert sees
`m · N / E` weighted-average rows of the input. The output is a
smooth function of `(D, C, X)` — fully differentiable, no balancing
loss, no straight-through hack.

**Identity at step 0**: with `D, C` initialized to small Gaussian noise,
after the row-wise softmax over `(N · m)` slots, `D` is approximately
*uniform* — every expert sees roughly the same weighted-average of all
input tokens. With experts initialized identically (standard FFN init,
fan-in variance), all `E` expert outputs are statistically equivalent
to a single expert's output, so `Σ_e C_e · Y_e ≈ single_expert(X)`. The
output is therefore close to the *un-MoE* baseline at step 0 (a single
FFN applied to the input, which is what the baseline already has if
`E = 1`). As training proceeds, `D, C` learn to specialize.

For tiny1m3m at `n_experts = 4`, `m = 4` slots per token: extra params
are `3 · n_experts × n_ff = 3 × 4 × (d_model × d_ff) = 12 × (144 × 576)
≈ 995K` — far over budget. We *must* reduce expert width to compensate:
target total extra cost ≤ 50K params (5% of 0.94M). With `n_experts = 4`,
each expert is `(d_model × (d_ff/4))` ≈ 144×144 = 20.7K params; total
expert cost ≈ 83K. Plus `D + C` matrix is `(N·m + E) × N` — depends on
N (batch × seq). At training BS=32 × T=512, N=16384, so `D` is
`(N·m) × E = 65536 × 4 = 262K` floats ≈ 1MB of weights — fits, but
it's the dominant cost. The dispatch/combine weights can be
*per-token-derived* from a small per-token linear projection
(`D[i, e] = softmax_e(W_d · x_i · m_e)`) to keep the slot count fixed
without per-(token, slot) parameters.

## Design sketch
- `models/soft_moe.py` (new): `SoftMoEFFN` class — wraps E parallel
  FFNs (each narrower than the baseline FFN to keep total param cost
  fixed) plus the `softmax`-based dispatch/combine. ~120 LoC.
- `models/llm.py`: when `config.use_soft_moe=True`, replace the
  standard FFN with `SoftMoEFFN`. Width of each expert FFN is
  `d_ff / n_experts` (so total FFN params stay at the budget).
- `configs/llm_config.py`: add `use_soft_moe: bool = False`,
  `soft_moe_n_experts: int = 4`, `soft_moe_n_slots: int = 4`.
- LoC: ~130 (soft_moe.py) + ~10 (plumbing) = ~140.
- Identity at step 0: with `W_d, W_c` zero-init, `D, C` are uniform
  softmaxes; with experts init'd with standard FFN init, the E parallel
  FFNs produce statistically equivalent outputs; the slot-weighted
  recombination reduces to a single FFN. Mathematically *not*
  bit-identical to the single-FFN baseline (E parallel FFNs with
  independent init produce O(E·σ) noise on the output, even though
  they're unbiased estimators of a single FFN), but the deviation is
  O(1/√E) — well below the run-to-run noise floor at our scale.
- The intuition: standard top-k sparse MoE doesn't fire at tiny1m3m
  because (a) load-balancing losses compete with the LM loss, (b)
  the router's gradient signal is sparse. Soft MoE removes both
  failure modes: every expert sees every token (sparse gradient
  problem solved) and there's no balancing loss to tune (HP-free).
  The cost is E× the FFN compute, partially amortized by the
  narrower per-expert width. A null says "0.94M doesn't have enough
  capacity for MoE to help, even with the cleanest routing"; a
  win says "the FFN is the binding capacity constraint and Soft
  MoE's differentiable capacity boost unlocks loss".

## Plan

**Files to touch**

- `models/soft_moe.py` (new): `SoftMoEFFN` module — wraps E parallel
  `SquaredReLUFeedForward`-shaped experts (each narrower than the
  baseline FFN, with `d_ff / n_experts` so total FFN params stay at
  the budget) plus the softmax-based dispatch/combine. ~110 LoC.
- `models/layers.py` — `TransformerBlock.__init__` picks
  `SoftMoEFFN(...)` when `ffn_variant == "soft_moe"` (one new branch
  in the existing if/elif ladder; no new kwarg plumbing required on
  `TransformerBlock` itself beyond `ffn_variant`).
- `configs/llm_config.py` — add `use_soft_moe: bool = False`,
  `soft_moe_n_experts: int = 4`, `soft_moe_n_slots: int = 4` on
  `LLMConfig`; add a new `ffn_variant="soft_moe"` allowed value;
  add a `Tiny1M3MSoftMoEConfig` preset.
- `train_llm.py` — add `--use_soft_moe`, `--soft_moe_n_experts`,
  `--soft_moe_n_slots` CLI overrides.

**Identity at step 0**

Dispatch and combine are per-(token, slot) learned from a small
linear projection (per-token, then softmax-over-experts) so D, C
have no per-(token, slot) parameters. The dispatch/combine
projections are **zero-init** so `D, C` are uniform softmaxes at
step 0: each expert sees roughly the same weighted-average of all
input tokens. With E experts initialized via the standard FFN init
(fan-in variance) all `E` expert outputs are statistically
equivalent to a single FFN, and `Σ_e C_e · Y_e ≈ single_expert(X)`.
NOT bit-identical to the single-FFN baseline at flag-on because
E independent FFNs produce O(√E) noise on the output (well below
the run-to-run noise floor at our scale). With `use_soft_moe=False`
the module is never built and the baseline path is bit-identical.

**Run command**

```
/venv/main/bin/python train_llm.py --config tiny1m \
  --config_class configs.llm_config.Tiny1M3MSoftMoEConfig \
  --seed 42
```

The final val loss is read from the `metrics.json` produced by the
trainer (`runs/<config_class>/seed42/metrics.json` →
`final_val_loss`).

**LoC budget**: ~110 (soft_moe.py) + ~10 (configs/llm_config.py) +
~10 (train_llm.py) + ~5 (TransformerBlock branch) = ~135 LoC,
under the 200 LoC cap.

## Scale evidence
- arXiv:2406.06589 (Puigcerver et al. 2024): ViT-L (~300M) on
  ImageNet-21k, −0.05 to −0.15 nats vs top-1 / top-2 sparse MoE
  at matched FLOPs.
- 1.3B-param LM on C4: paper reports parity-to-improvement vs
  dense baseline at matched active params, with Soft MoE gaining
  when E ≥ 4.
- Independent validation: Soft MoE has been integrated into
  `big_vision` and `timm` for vision; less adoption in LM land but
  the mechanism is modality-agnostic.
- Transfer risk: **med**. Validated at 300M-1.3B (≥100M), the
  mechanism is scale-free (capacity, not regularization), but the
  *budget pressure* at 0.94M is severe: at `n_experts=4`, each
  expert is `d_ff/4 ≈ 144` wide, which is below the standard
  `d_ff ≥ 4·d_model` rule of thumb for FFN capacity. The slot
  matrices also dominate the parameter count at short contexts.
  A null from "FFN too narrow to be useful" is plausible; a win
  would mean "even narrow FFN experts add capacity when their
  outputs are mixed".

## Why it's worth a slot
The only MoE lever filed is 108-simbal-router (a *router regularizer*
applied to a hard-routed MoE — distinct mechanism, requires hard
routing infrastructure first). Soft MoE is the cleanest possible
MoE extension: differentiable, no balancing loss, no straight-through
hack. It tests the **MoE capacity hypothesis** at 0.94M with the
fewest confounders. If Soft MoE wins, the next step is to test it
with a narrower baseline FFN (so the param budget can buy wider
experts); if it nulls, the slot closes the MoE-direction question
for tiny1m3m ("0.94M's FFN is wide enough that capacity is not the
binding constraint"). Either outcome is informative — exactly the
shape of a good slot.
