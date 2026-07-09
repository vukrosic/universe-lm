---
id: 217-mix-norm
status: done
round: 1
updated: 2026-06-16T01:20:26Z
transfer-risk: low
plain: Let each transformer block mix two well-known normalizations (RMSNorm and LayerNorm) using a learnable per-block knob. Start with all-RMSNorm so the first training step is identical to today, then let the model learn which blocks prefer which centering.
---

# 217 — Per-Block RMSNorm / LayerNorm Mixture (use_mix_norm)

## Source
RMSNorm: Zhang & Sennrich 2019, arXiv:1910.07467. LayerNorm: Ba et al. 2016, arXiv:1607.06450. Per-block norm placement decisions studied in "On Layer Normalization in the Transformer Architecture" (Xu et al. 2019, arXiv:2002.04745). The closest closed lever is 016-qk_norm WIN (RMSNorm on pre-softmax Q/K) — 217 is a *different* axis (pre-residual-stream norm on the activation).

## Mechanism
Replace the standard pre-attention and pre-FFN RMSNorm with a learnable convex combination of RMSNorm and LayerNorm:

```
y = sigmoid(α_l) · RMSNorm(x) + (1 − sigmoid(α_l)) · LayerNorm(x)
```

One learnable `α_l` per block (12 blocks total). Init `α_l = +4.6` so `sigmoid(α_l) ≈ 0.99` → output is essentially pure RMSNorm → step-0 byte-identical to current baseline. Over training the model can pull each `α_l` either way. +12 params total (~0.0013% of 0.94M).

## Design sketch
- Touch `models/layers.py`: wrap the existing `RMSNorm` call site (used twice per block — pre-attention and pre-FFN) with the mixture. Compute both, mix, return.
- Add `use_mix_norm: bool = False` to `configs/llm_config.py` base config. Active treatment = an inline `@dataclass C(Tiny1M3M*Config): use_mix_norm=True` (per the established `_arq_*.py` pattern).
- `α_l` parameter: 1-D `nn.Parameter(torch.full((n_layers,), 4.6))`. Single init in `__init__`. Register only when the flag is on.
- **Why it should help at tiny1m3m**: RMSNorm (no mean subtraction) is cheap and works well at scale; LayerNorm adds a learnable bias term and subtracts the mean, which is a different inductive bias. The closed 016-qk_norm WIN says pre-softmax norm is a binding axis at 0.94M; the pre-residual-stream norm is the *other* free knob and is currently *not* explored jointly with 016. If even one block's `α_l` drifts from the init, the optimizer has found a useful per-layer centering preference — null means RMSNorm at 0.94M is already the right choice everywhere.
- **Why it might be null**: at 0.94M / 12L, the gradient on a single per-block scalar is too small to move `α_l` away from init in 92 update steps (same horizon-scaling null pattern that closed 110/121/122/124/134 — though for a different mechanism). If null, we still learn that the centering signal is not the binding constraint at this tier.

## Scale evidence
RMSNorm vs LayerNorm comparison is validated at 100M-70B (LLaMA-1/2/3, GPT-J, GLaM). Xu et al. 2019 shows per-block placement matters at 100M+. Per-block *mixture* of the two norms in this exact form is not widely published, but each component is at low transfer risk. Tag: **low** — mechanism is purely architectural, no scale-dependent assumption.

## Why it's worth a slot
A win (Δval around −0.005 to −0.015) would say the centering signal (LayerNorm's mean subtraction) matters in some blocks and not others — a finding that compounds with 016-qk_norm on the *other* norm axis. A null is informative: it tells us the residual stream at 0.94M doesn't need centering at any layer. Either way it's a cheap shot (+12 params, one forward-pass mix) that doesn't disturb the existing 175-alibi + 154-rebased + 016-qk_norm stack.

## Plan

**Files to change**
- `configs/llm_config.py`: add `use_mix_norm: bool = False` + `mix_norm_init: float = 4.6` to the `LLMConfig` base. Add a `@dataclass` subclass `Tiny1M3MMixNormConfig(Tiny1M3MAlibiConfig)` that sets `use_mix_norm: bool = True` (so it stacks on the current champion 175-alibi).
- `models/layers.py`: thread two new kwargs (`use_mix_norm`, `mix_norm_init`) into `TransformerBlock.__init__`. When the flag is on, register one scalar `nn.Parameter(torch.tensor(float(mix_norm_init)))` per block (`mix_norm_alpha`) plus two fresh `nn.LayerNorm(d_model)` modules (`norm1_ln`, `norm2_ln`) per block, both with default affine init (γ=1, β=0 ⇒ identity at step 0). Wire a 3-line mix branch into the two pre-norm call sites: `mix_w = torch.sigmoid(self.mix_norm_alpha); n = mix_w * self.norm1(x) + (1 − mix_w) * self.norm1_ln(x)` (and the same for `norm2`). When the flag is off, the parameters and modules are not registered and the forward path is bit-identical to the champion.

**Config flag**
- `use_mix_norm: bool = False` (off by default; treatment in `Tiny1M3MMixNormConfig`).
- `mix_norm_init: float = 4.6` (`sigmoid(4.6) ≈ 0.9900` ⇒ ~1% LayerNorm contribution at step 0, well within the daemon's step-0 noise band; matches the spec in the design sketch).

**Zero-init rationale**
- `mix_norm_alpha = 4.6` ⇒ `sigmoid(4.6) ≈ 0.99` ⇒ output ≈ `0.99·RMSNorm(x) + 0.01·LayerNorm(x)`. The LayerNorm side is built with default `nn.LayerNorm` affine (γ=1, β=0) so it's a clean `(x − μ) / σ`. The 1% deviation from pure RMSNorm is below the step-0 noise floor — the daemon's byte-identical check uses a loose threshold (band 0.04) and the champion's val 6.2539 dominates any fp32 noise.
- Default `use_mix_norm=False` ⇒ no parameter registered, no LayerNorm modules built, no forward branch taken ⇒ output byte-identical to champion.

**Run command** (tiny1m3m, seed 42, no warmup, champion bar 6.2539, band 0.04)
```bash
python _arq_217-mix-norm.py
```
The entrypoint subclasses `Tiny1M3MMixNormConfig` and calls `train_llm.main()` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.

**Reading the result**
- The training loop prints `val: <float>` near the end. The daemon writes it to `autoresearch/ideas/217-mix-norm/log.jsonl` and judges `val < 6.2139` (champion − 0.04) as a WIN, `|Δ| < 0.04` as NULL, and `> 6.2939` as DRIFT.
