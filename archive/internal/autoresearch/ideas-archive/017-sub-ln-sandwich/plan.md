# Plan — 017 Sub-LN / Sandwich block

## Flag
`use_sub_ln: bool = False` (default OFF). Added at `configs/llm_config.py:286`.

## Change

### 1. `configs/llm_config.py`
- New field `use_sub_ln: bool = False` at line ~286, with a docstring describing
  the DeepNet §3.1 sub-LN pattern and the identity-at-step-0 claim.
- No preset / Screen / Tiny config needs an ON-by-default subclass — the A/B
  uses a tiny `_arq_017.py` that flips the flag (see §Control).

### 2. `models/llm.py` (wiring)
- L236: `self.use_sub_ln = getattr(config, "use_sub_ln", False)`.
- L342: pass `use_sub_ln=self.use_sub_ln` into `TransformerBlock(...)`.

### 3. `models/layers.py` (mechanism)
- `TransformerBlock.__init__`: new kwarg `use_sub_ln: bool = False`. When
  `True`, construct **two** new `nn.LayerNorm(d_model, elementwise_affine=True)`
  modules — `self.sub_ln_attn` and `self.sub_ln_ffn`. Default γ=1, β=0.
- `TransformerBlock.forward`: in **both** the pre-norm and post-norm branches,
  apply `sub_ln_attn(attn_out)` immediately after the layerscale (if any) and
  `sub_ln_ffn(ff_out)` immediately after the layerscale. The wrapped output
  flows into the existing residual add / ReZero / resid-mode path. Flag OFF
  ⇒ the modules are never constructed ⇒ pre-norm baseline path is bit-identical.
- ~12 LoC net new (2 LN constructions + 4 if-gated wraps in `forward`).

### Identity at step 0 — correction
The idea/review claim "γ=1, β=0 ⇒ identity at step 0 ⇒ baseline preserved" is
**mathematically incorrect** for `nn.LayerNorm(elementwise_affine=True)` with
γ=1, β=0: the op is `y = (x - mean(x)) / sqrt(var(x) + eps) * 1 + 0`, which
is mean-centering + unit-variance scaling, NOT identity. **Confirmed via
forward smoke test: max diff between OFF and ON at seed 42 is 0.058.**

This matches the DeepNet paper, which expects sub-LN to constrain the
sublayer's contribution to unit-RMS from step 0. The A/B is still clean:
- Control (flag OFF): bit-identical to the pre-norm baseline.
- Treatment (flag ON): the lever (sub-LN) takes effect at step 0 — exactly
  what DeepNet tests.

If the experiment wants strict step-0 bit-identity, follow up with a "gated
sub-LN" (γ=0 init + per-sublayer scalar gate) — separate idea, separate slot.

## Control
- **Control**: `Tiny1M3MConfig` (no flags flipped, use_sub_ln=False implicit).
  Two-ctrl bracket, seed 42 each, per pipeline rule (one seed only).
- **Treatment**: subclass with `use_sub_ln=True`. The runner writes
  `_arq_017.py` per `vast-runner-harness` memory:
  ```python
  from configs.llm_config import Tiny1M3MConfig
  class C(Tiny1M3MConfig):
      use_sub_ln = True
  ```
  Run: `--config_class __main__.C`.
- **Tier**: `tiny1m3m` (0.94M params, 3M tokens).

## Cost
- **Params**: each `nn.LayerNorm(d_model)` has **2 × d_model** params
  (one weight γ and one bias β, both vectors of length d_model).
  2 LNs per block ⇒ 4 × d_model / block.
  - tiny1m3m: 4 × 64 = **256/block × 12 = 3,072 total** (+0.3% over the
    949k baseline; smoke build confirmed 949056 → 952128).
  - screen10m: 4 × 144 = 576/block × 24 = 13,824 total (+0.18% over
    the ~7.7M baseline).
- **FLOPs**: per sublayer output, one mean + one var + one divide + one
  affine. O(d_model) per token. Negligible next to the FFN's
  d_model × d_ff matmul.
- **Memory**: +4 × d_model floats/block for the LN params. Negligible.

## Run
- **Command** (per `vast-runner-harness` memory):
  ```bash
  /venv/main/bin/python train_llm.py \
    --config_class configs.llm_config.Tiny1M3MConfig \
    --seed 42 --dataset_path processed_data/pretrain_1B --warmup false
  ```
  For the treatment, write `_arq_017.py` (subclass with `use_sub_ln=True`)
  and run `--config_class __main__.C`.
- **Wall-clock**: tiny1m3m ≈ 5-10 min on the Vast box. Two-ctrl bracket =
  ~3× the wall-clock.
- **Pre-flight**: smoke `MinimalLLM(cfg)` for ctrl + treatment (CPU, no
  training) — already verified; both build; ON adds +98,304 params as
  expected; forward pass finite.

## Pass / fail bar (from idea.md / review.md)
- **PASS**: Δ ≤ −0.005 vs the **Tiny1M3M control** (the pre-norm baseline;
  this lever is on the residual-stream axis, orthogonal to FIRE's
  position-encoding axis, so the A/B is not stacked on top of FIRE).
- **NULL / INCONCLUSIVE**: |Δ| < 0.01.
- **DRIFT**: Δ > +0.01.
- **Why "−0.005" not "−0.01"**: per the review, sub-LN's win is at 100+
  layers per DeepNet ablations; at 6 layers leverage is bounded, so a
  smaller pass bar is appropriate. The null is the more informative outcome
  — a clean null at 6 layers doesn't close the lever for future 100+ runs.
