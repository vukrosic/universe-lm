---
id: 134-mega-ema
status: done
round: 1
updated: 2026-06-13T15:07:53Z
transfer-risk: med
plain: It replaces the standard attention's value matrix with a learned moving average of the values — so each attention layer can both look at recent context and "remember" an exponentially-smoothed summary of older tokens.
---

# 134 — Mega: Moving Average Equipped Gated Attention (EMA on V)

## Source
Ma, Zhou, Yi, Xu, Lv, Yang, Zhang, Li, "Mega: Moving Average
Equipped Gated Attention" (arXiv:2209.10655, ICLR 2023).
https://arxiv.org/abs/2209.10655

Validated on WikiText-103 LM, GLUE classification, and a long-
range-arena benchmark for long-context modeling. The lever is
a single architectural modification — add an EMA of the values
to each attention layer's value matrix.

## Mechanism
Standard attention: `Q, K, V` are all computed from the input
projection `xW`. Mega adds an *exponential moving average* of
the input:
  `u_t = EMA(x_t) = β · u_{t−1} + (1 − β) · x_t`
  `V_mega = concat(V, u)`     (extend V with the EMA)
  `attn = softmax(Q · K^T / √d) · V_mega`     (attention over
                                                raw + EMA values)

The intuition: standard attention uses the *current* token's
value vector. Mega augments this with a *smoothed* value vector
that captures recent context. The EMA acts as a "memory" of
the recent past — when the current token's value is noisy or
uninformative, the EMA provides a smoothed alternative.

The paper combines this with a *gated* output:
  `output = gate(Q, K) · attn · V_mega`     (gated attention)

Where `gate(Q, K) = σ(Q · K^T / √d)` is a sigmoid gate (similar
to the closed "gated attention" 024 lever, but applied in a
different position).

**Identity at step 0**: with `β` initialized to `0` (or near-0),
the EMA `u_t = x_t` (no smoothing). The Mega value matrix
becomes `V_mega = concat(V, x_t)`, which is *not* identical to
standard `V`. The lever is **not** bit-identical to baseline
at step 0.

With `β = 0`, Mega collapses to gated attention (which closed
024 has shown is a positive lever). With `β = 1`, Mega collapses
to constant `u_t = u_0` (no signal). The PASS bar is defined
at intermediate `β` (paper default `β = 0.9`).

## Design sketch
- `models/mega.py` (new): `MegaAttention` — wraps standard
  attention with an EMA on the value input. The EMA is computed
  via `u_t = β · u_{t−1} + (1 − β) · x_t` with a learnable
  per-channel β. ~60 LoC.
- `models/llm.py`: when `use_mega=True`, replace standard
  attention with `MegaAttention`. The Q, K projections are
  unchanged; V is computed from `concat(xW_v, EMA(xW_v))`. ~15 LoC.
- `configs/llm_config.py`: add `use_mega: bool = False`,
  `mega_beta: float = 0.9`, `mega_gate: bool = True` (apply
  sigmoid gate). ~10 LoC.
- LoC: ~85 total (under 200 ceiling).
- Identity at step 0: with `β = 0`, the EMA is the current
  token's value (no smoothing). The Mega output is
  `gated_attn(V, V)` which is *similar to but not identical*
  to closed 024's gated attention.
- The intuition: at 0.94M, the EMA acts as a "temporal smoother"
  on the value matrix. The bet is that the smoothed values
  reduce noise in the attention output, especially for the
  small-context 3M-token training window. A null would say
  "at 0.94M the attention already captures the relevant context
  and the EMA is redundant"; a win would say "the EMA smooths
  out per-token noise and the smoothed values give a better
  attention output".

## Scale evidence
- arXiv:2209.10655 (Ma et al. 2022, ICLR 2023): WikiText-103 LM
  (Transformer-XL scale ~250M), GLUE classification (BERT-base
  110M), long-range-arena (LRA) for long-context modeling.
  Reports +0.5-1.0 perplexity points on WikiText-103 vs
  Transformer-XL baseline.
- Transfer risk: **med**. Validated at ≥100M (WikiText-103
  Transformer-XL is ≥250M, GLUE BERT-base 110M), the lever
  is scale-free (EMA is well-defined at any depth). At 0.94M
  with 12L and 3M tokens, the EMA acts as a "smoothing" on
  per-token noise — relevant.

## Why it's worth a slot
Mega is the only *value-side* attention modification filed
(closed attention levers are all on Q, K, or the attention
*pattern*; Mega operates on V). It is ortho to every closed
attention lever (SWA, GQA, NSA, Diff Attn, Gated Attn 024,
Scalable Softmax 025). The lever also adds a *temporal
smoothing* that no closed lever provides. A win would say
"value-side smoothing is the missing piece and Mega should
be added to the architecture"; a null would say "at 0.94M
the attention pattern already captures the relevant context
and the EMA is redundant". Either outcome is informative —
neither axis (value-side, temporal-smoothing) has been
probed in our pipeline.

## Plan

- **Files touched:**
  - `configs/llm_config.py` — add 3 flags to `LLMConfig`:
    `use_mega: bool = False`, `mega_beta: float = 0.9`,
    `mega_use_input: bool = True`. Add `Tiny1M3MMegaConfig(Tiny1M3MConfig)`
    preset with `use_mega=True` (the A/B treatment subclass).
  - `models/layers.py` — extend `MultiHeadAttention.__init__` and
    `forward` to wire the EMA on the V stream. `mega_beta` is a
    learnable per-channel scalar in `[0, 1]` parametrized as
    `β = sigmoid(raw)` (raw init 0 → β = 0.5 at step 0). The EMA
    state `u` is computed via `u_t = β·u_{t-1} + (1-β)·x_t` over
    the *input* to the V projection (the pre-projection residual),
    then BOTH the raw input and the EMA are projected through
    shared W_V and concatenated into `V_mega = concat(V_raw, V_ema)`
    of shape `[B, T, 2·kv_size]`. The AV product is then split:
    the attention weights softmax over the doubled key dimension
    and the resulting output is `o_h = (a·V_raw + a·V_ema) ·
    W_O`. The standard 2-head V split lets us keep SDPA without
    per-head reshape. At step 0 β=0.5 ⇒ `u_t = 0.5·u_{t-1} + 0.5·x_t`
    ⇒ the EMA is half-smoothed; the concat means V is NOT
    identical to baseline. Per the idea: "With β=0, Mega collapses
    to gated attention (which closed 024 has shown is a positive
    lever). With β=1, Mega collapses to constant u_t = u_0." We
    use β=0.5 at step 0 (between those two extremes, the natural
    "smoothed" starting point). The lever is explicitly NOT
    byte-identical to baseline at step 0 — the design sketch
    documents this and the closed 024 precedent justifies the
    "β=0 collapse to gated attention" path.

- **Config flag name:** `use_mega` (plus `mega_beta`,
  `mega_use_input`).

- **Step-0 init:** per-channel `raw = 0` ⇒ `β = σ(0) = 0.5` at
  step 0 (paper default is 0.9; we start at 0.5 — the natural
  "half-smoothed" position — and let the optimizer find the
  operating point). NOT byte-identical to baseline at step 0
  (the concat doubles V width). With `use_mega=False` the EMA
  is never built and the baseline forward is bit-identical.

- **Run command (treatment):**
  ```bash
  python _arq_134-mega-ema.py
  ```
  which sets `--config_class=__main__.C` where
  `C = Tiny1M3MMegaConfig`, `--seed 42`,
  `--dataset_path processed_data/pretrain_1B`, `--warmup false`.
  Or directly:
  ```bash
  python train_llm.py \
    --config_class configs.llm_config.Tiny1M3MMegaConfig \
    --seed 42 --dataset_path processed_data/pretrain_1B \
    --warmup false
  ```

- **Reading final val loss:** `plots/metrics_<TOKENS>_<TS>.json` →
  `final_metrics.val_loss` (same convention as every other idea;
  the runner pulls this from `remote-results/.../results.json`).
