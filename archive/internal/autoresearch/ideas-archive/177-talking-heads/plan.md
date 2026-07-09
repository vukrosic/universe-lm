# Plan — 177 talking-heads

## Flag
- `use_talking_heads_q: bool = False` — pre-softmax cross-head H×H logit mix.
- `use_talking_heads_out: bool = False` — post-softmax cross-head H×H value mix.
- Both default OFF on `LLMConfig` (already defined in `configs/llm_config.py`
  line 1094; `use_talking_heads_out` is read via `getattr` from
  `models/llm.py:685, 968` so it does not need to live on the parent for the
  lever to work).
- Treatment: a new `@dataclass class Tiny1M3MTalkingHeadsConfig(Tiny1M3MConfig)`
  in `configs/llm_config.py` with **both** flags True. Step-0 ⇒ identity
  `M @ x = x` on the cross-head axis ⇒ forward bit-identical to baseline
  (max-abs-diff = 0.0).

## Change
- `configs/llm_config.py` — append `Tiny1M3MTalkingHeadsConfig` (~5 LoC,
  no edits to `LLMConfig` itself). The parent already has
  `use_talking_heads_q: bool = False` at line 1094; the subclass overrides
  it to `True` and introduces `use_talking_heads_out: bool = True` (no
  conflict with parallel-AI diff hunks — they added `use_swiglu_ffn`,
  `use_entmax`, `use_dropconnect_wo` + warmup, xPos-decay, none near the
  talking-heads flag or this part of the file).
- No edits to `models/layers.py` — the talking-heads mechanism is already
  wired end-to-end (`MultiHeadAttention.__init__` lines 1072/1099, parameter
  inits at 1994 (`talking_heads_M` = `torch.eye`) and 2127
  (`talking_heads_out_M` = `torch.eye`), application via einsum at
  3244-3248 (pre-softmax) and 3274-3276 (post-softmax), and
  `use_talking_heads_q` plumbing into `_apply_logit_op` /
  `use_talking_heads_out` plumbing into `_apply_output_op` are all
  present). Identity init ⇒ step-0 byte-identical.
- Repo root — add `_arq_177-talking-heads.py` (treatment entry, top-level
  `class C` re-exporting `Tiny1M3MTalkingHeadsConfig`).
- `autoresearch/ideas/177-talking-heads/run.json` — daemon descriptor.

## Control
- **Control**: bare `configs.llm_config.Tiny1M3MConfig` (cached baseline
  val loss 6.4306 ± ~0.04 — see `autoresearch/baseline-cache.json`).
- **Treatment**: `Tiny1M3MTalkingHeadsConfig` (both flags True).
- **Tier**: `tiny1m3m` (0.94M params, 3M tokens, 12 layers, 4 heads).
- **Seed**: 42 (one seed only, per protocol).
- **Ablations (Q-only, Out-only)**: only if the joint run shows a real signal
  (WIN band). The single A/B at this gate is the joint form per the idea.

## Cost
- **Params Δ**: +2 × (H × H) = 2 × 4 × 4 = **+32 trainable params**
  (16 per lever × 12 layers = 192 in absolute, but H=4 means 16 each;
  2 levers × 16 = 32 per layer, 32 × 12 layers = 384 total). For
  reference, the baseline model is ~0.94M params ⇒ relative Δ ≈ +0.04%.
  Negligible memory.
- **FLOPs Δ**: two extra `einsum("bhst,hH->bHst", …)` and
  `einsum("bhtd,hH->bHtd", …)` per layer per forward. H=4 ⇒ matmul is
  effectively a 4×4×T² reduction and a 4×4×T×d_k reduction. At T=2048,
  d_k=16: ~131K extra muls per layer per fwd (≪ the 0.94M-param FFN).
  Wall-clock impact: undetectable in the 12m budget.
- **Memory Δ**: identity-init 4×4 matrices are stored as float32
  `Parameter` objects (one per layer per lever). 32 × 12 layers × 4 bytes
  = ~1.5 KB. Negligible.
- **No new dependencies**, no new files beyond the 3 (config entry,
  `run.json`, `_arq_*.py`).

## Run
- Command: `python _arq_177-talking-heads.py`
  (boots `train_llm.py` with `--config_class __main__.C`, `--seed 42`,
  `--dataset_path processed_data/pretrain_1B`, `--warmup false`).
- Tier: **tiny1m3m**, seed **42** (locked by protocol).
- Wall-clock budget: `12m` (`job_timeout` in `run.json`).
- **Pass/fail bar (copied from `idea.md`):**
  - **WIN**: `Δval = treatment_val − control_val ≤ −0.025` (clean PASS,
    outside the ±0.04 cache band, above the largest per-head-scalar
    null magnitude 0.0131).
  - **NULL**: `|Δ| < 0.015`.
  - **DRIFT**: `|Δ| > 0.04` wrong-sign ⇒ reject the cross-head-mix
    family at 0.94M.
- **Step-0 sanity**: identity init ⇒ `M @ x = x` literally, so the
  flag-on forward at step 0 must match the flag-off forward to all
  bits (max-abs-diff = 0.0). The build-smoke in §5 of the protocol
  exercises the construction; step-0 bit-identity is a property of
  the math, not the harness, so no extra harness is needed beyond
  the daemon's `MinimalLLM(C())` CPU smoke.
- **Ablations** (deferred): if the joint shows a real signal, the
  Q-only (`Tiny1M3MTalkingHeadsQConfig`-style) and Out-only
  (`Tiny1M3MTalkingHeadsOutConfig`-style) configs already exist in
  `configs/query_tiny_ablations.py` and
  `configs/attention_output_ablations.py`; both would re-enter as
  separate `177-q-only` / `177-out-only` ideas, not as ad-hoc reruns
  of this one.
