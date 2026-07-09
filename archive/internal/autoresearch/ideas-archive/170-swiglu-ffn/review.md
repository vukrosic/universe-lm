## r1 — 2026-06-14 — verdict: approve

- **Source real, mechanism sound.** Shazeer arXiv:2002.05202 is the canonical
  SwiGLU paper; LLaMA/Mistral/Qwen/Gemma/OLMo/PaLM all ship SwiGLU at 7B-540B
  (Touvron 2023, Chowdhery 2022). Authors plausible, citations resolve.
  Mechanism is structural (3-projection gated FFN vs 2-projection GELU), not
  an HP lever. ✓
- **Step-0 identity clean.** `W_gate = 0` ⇒ `silu(W_gate·x) = 0` ⇒ FFN output
  is exactly 0 at step 0 (residual stream carries only the attention sub-block).
  ReZero-style baseline; optimizer must grow the gate. Stable. ✓
- **Not a closed-axis duplicate.** `closed.md:112` (153-relu2-ffn) closes
  *activation-shape* (ReLU² inside 2-matrix FFN); 170 is *gating-structure*
  (3-matrix FFN with explicit elementwise gate ⊙). Mechanistically distinct.
  Adjacent FFN-side nulls (146-sparse-ffn, 156-moa, 157-conv-ffn, 158-gau,
  117-soft-moe, 118-MoD) are capacity-injection levers, not a 2-matrix→3-matrix
  structural swap. No closed-axis dupe. ✓
- **tiny1m3m only, seed 42.** Plan only references tiny1m3m (val ≈ 6.43 ± 0.04
  baseline-cache). No screen20m/ladder mentions. ✓
- **Implementable in <200 LoC.** Parallel worker diff (~95 LoC across
  `models/components.py` + `models/layers.py` + `configs/llm_config.py`) is
  well under the 200 LoC budget. New `SwiGLUZeroInitFeedForward` class +
  `use_swiglu_ffn` kwarg + cascading FFN branch ahead of `ffn_variant`. ✓
- **Transfer risk: low justified.** Mechanism is scale-invariant (per-input
  soft routing). Validated by every modern open LM at ≥7B; direct 1B-class
  validation in the original paper (T5 1.1B-3B). Low tag is correct. ✓
- **Param parity via Shazeer 2/3 trick.** `d_ff_swiglu = (2 * d_ff) // 3 = 170`
  for d_ff=256; new FFN params 32,640 vs baseline 32,768 (-0.4%, within noise).
  Math consistent across idea.md, taste.md, and components.py docstring. ✓
- **Finding 1 (minor, for reviser/implementer): explicit Δval pass/fail bar.**
  idea.md says "expected Δval ≈ -0.01 to -0.04 at tiny1m3m" but no explicit
  pass bar. Tighten to: **WIN = Δval ≤ -0.005 vs cached baseline (≈ 6.4394);
  NULL = |Δ| < 0.01; DRIFT = Δ > +0.01**. Matches the 016-qk_norm / 023-canon-conv
  convention used elsewhere in `closed.md`. (-0.01 expected is right at the
  noise floor; -0.04 expected is comfortably above.)
- **Finding 2 (minor, for implementer): mutual-exclusion assertion missing.**
  idea.md §Mechanism says "assert `not (ffn_variant == 'swiglu' and use_moe')`
  at the top of the FFN forward" — the parallel-worker's implementation diff
  does not include this assert. Add a one-line check at the top of the
  `use_swiglu_ffn` branch in `models/layers.py` so a later MoE-on + SwiGLU-on
  combo fails loud, not silent. Low priority (no current flag combo triggers
  it, but it's free insurance).
- **Finding 3 (procedural, for implementer): runner harness not yet present.**
  `autoresearch/ideas/_arq_*.py` files for other recent ideas exist
  (`_arq_153-relu2-ffn.py`, `_arq_154-rebased-attn.py`, etc.) but
  `_arq_170-swiglu-ffn.py` is not in the working tree yet. Implementer must
  create the runner mirroring the 153 pattern (build `Tiny1M3MSwigluFFNConfig`,
  call `config_class`, run `/venv/main/bin/python _arq_170-swiglu-ffn.py`).
- **Verdict: approve.** Sound, falsifiable, low transfer risk, clean identity
  init. Proceeds to code gate (round reset to 1).
