---
id: 170-swiglu-ffn
status: done
round: 1
updated: 2026-06-14T10:28:45Z
transfer-risk: low
plain: Swap the FFN's plain GELU activation for SwiGLU — a gate × value product used by LLaMA, Mistral, and PaLM — and start the gate matrix at zero so the FFN is silent on the first step.
---

# 170 — SwiGLU FFN (Gated Gated-Linear-Unit Replacement of GELU MLP)

## Source
- Shazeer, "GLU Variants Improve Transformer", arXiv:2002.05202 (2020) — the canonical
  SwiGLU paper. Shows consistent -0.1 to -0.4 PPL gains on T5-1.1B/1.6B/3B across
  GLU variants; SwiGLU tied with GeGLU as the best variant.
- Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023) — LLaMA
  uses SwiGLU FFN; LLaMA 2/3 retain it. So do Mistral, Qwen, Gemma, OLMo.
- Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways" (2022) — PaLM
  uses SwiGLU in every block at 8B-540B.
- Closest in-repo prior: 153-relu2-ffn (NULL at tiny1m3m, `closed.md:112`). 153 was an
  *activation-curvature* swap (ReLU² inside the same 2-matrix FFN); 170 is a *gating-
  structure* swap (3-matrix FFN with an explicit elementwise gate). The mechanism
  being tested is qualitatively different — the gate is a learnable per-input
  routing signal, not just an activation function. 153's null does NOT close 170.
- FFN-variant switch is not in `closed.md` axes (we've only tested activation-shape
  swaps via 153; gating-structure swap is a different lever).

## Mechanism
Baseline FFN: `y = W_down · GELU(W_up · x + b_up) + b_down` (two `d_model × d_ff`
projections, single GELU activation). SwiGLU FFN: `y = W_down · (silu(W_gate · x)
⊙ W_up · x)` — three `d_model × d_ff` projections and an elementwise gate ⊙.

`silu(z) = z · sigmoid(z)`. The gate signal `silu(W_gate · x)` is a smooth
learnable per-input soft switch; in the limit it lets the model turn subsets of
the d_ff hidden units on or off per token.

Param parity: with the standard 2/3 trick (Shazeer 2020, used in LLaMA), set
`d_ff_swiglu = round(2/3 · d_ff_baseline) = round(2/3 · 256) = 170`. New FFN
param count: 3 × `d_model × d_ff_swiglu` = 3 × 64 × 170 = 32,640 vs baseline
2 × 64 × 256 = 32,768. Effectively identical total params.

## Design sketch
- **Files**: `models/layers.py` (FFN module, ~30-50 LoC), `configs/llm_config.py`
  (add `ffn_variant: str = "swiglu"` branch and a `Tiny1M3MSwigluConfig`
  subclass that flips the flag).
- **Config flag**: `ffn_variant: str = "gelu"` (default) with new value
  `"swiglu"`. Branch in the FFN `forward`: if `"swiglu"`, allocate `W_gate`,
  `W_up`, `W_down` instead of the two-matrix pair.
- **Step-0 identity via zero-init**: initialize `W_gate = 0` (bias = 0). Then
  `silu(W_gate · x) = silu(0) = 0` for all `x`, so `silu(W_gate · x) ⊙ W_up · x = 0`,
  so the FFN output is **exactly zero** on step 0. This makes 170's step-0 a
  "ReZero-style" model — the residual stream carries the entire forward. The
  optimizer must then learn the gate; the FFN contribution ramps in gradually.
  At eval, the FFN contributes; at step-0, it doesn't. This is an *approximate*
  identity (not bit-identical to baseline, which has a non-zero GELU FFN at
  step-0) but the math is clean and the forward is stable.
- **Intuition (why it might lower val loss)**: gating is a per-input soft
  modularity prior — different tokens can route through different FFN
  sub-regions. LLaMA/Mistral/PaLM chose SwiGLU over GELU at 7B-540B specifically
  for this. The baseline weakness: plain GELU FFN treats every hidden unit
  identically; SwiGLU lets the model express "this token doesn't need units
  80-150" and "that token doesn't need units 30-90". The bet at 0.94M is that
  this is a structural advantage that holds even at small scale.
- **Mutual exclusion**: assert `not (ffn_variant == "swiglu" and use_moe)` and
  `not (ffn_variant == "swiglu" and ffn_variant == "sparse")` (no current
  sparse-ffn flag, but preempt the case) at the top of the FFN forward.

## Scale evidence
- SwiGLU is the standard FFN choice in 7B-540B open-weight models (LLaMA 1/2/3,
  Mistral, Qwen, Gemma, OLMo, Falcon). Shazeer's original 2002.05202 validates
  at T5 1.1B/1.6B/3B. The mechanism is the most-cited FFN improvement in
  modern open-source LMs.
- **Transfer risk: low** (≥100M validated directly by every modern open LM
  release; 1B-class direct validation in the original paper).

## Why it's worth a slot
The bet: SwiGLU's gating structure is a *stronger* inductive bias than plain
GELU — not just a different activation shape (which 153's null closed), but
a different connectivity. We expect Δval ≈ -0.01 to -0.04 at tiny1m3m (smaller
than LLaMA's gains at 7B but in the right direction). A null would tell us
gating structure doesn't bind at 0.94M/3M tokens and the FFN-activation axis
is fully closed (153 + 170 both null); a win would tell us gating is a real
lever and the 0.94M tier is large enough for the gate to learn. The
zero-init gate makes this a clean, safe test.

## Plan

**Files to change**
1. `models/components.py` — add `SwiGLUZeroInitFeedForward` class
   (mirrors existing `SwiGLUFeedForward` but with `gate_proj.weight = 0` init).
   `d_ff_swiglu = (2 * d_ff) // 3` for param parity (Shazeer 2/3 trick;
   `int(2*256/3) = 170`, matches idea.md math). New FFN param count:
   3 × `d_model × d_ff_swiglu` = 3 × 64 × 170 = 32,640 vs baseline
   2 × 64 × 256 = 32,768 (≈ 0.4% smaller, well within harness noise).
2. `models/layers.py` — add `use_swiglu_ffn: bool = False` kwarg to
   `TransformerBlock.__init__` and add a new branch in the FFN construction
   cascade that runs **AHEAD** of the existing `ffn_variant == "swiglu"`
   branch (so this lever wins when both flags are on). The branch builds
   `SwiGLUZeroInitFeedForward(d_model, (2*d_ff)//3, dropout)` and stashes
   `self.use_swiglu_ffn = True`.
3. `configs/llm_config.py` — add `use_swiglu_ffn: bool = False` to `LLMConfig`
   (default off, so baseline path is bit-identical when off) and add a new
   `Tiny1M3MSwigluFFNConfig(Tiny1M3MConfig)` subclass that flips the flag on.
   Also pass `use_swiglu_ffn` through to `TransformerBlock` in `MinimalLLM`.
4. `_arq_170-swiglu-ffn.py` — the runner harness. Mirrors the
   `_arq_153-relu2-ffn.py` pattern.

**Config flag**: `use_swiglu_ffn: bool = False` (default off, baseline path
untouched).

**Step-0 zero-init**: `SwiGLUZeroInitFeedForward` zeros `gate_proj.weight`
and `gate_proj.bias` at construction. Then `silu(W_gate · x + b_gate) =
silu(0) = 0` for all x, so `silu(gate) ⊙ (W_up · x) = 0` exactly, so the FFN
output is **exactly 0** at step 0 (better than baseline! the residual stream
carries only the attention sub-block at step 0). This is approximate-identity
to baseline (which has a non-zero GELU FFN at step 0), but the math is clean
and the forward is stable — same tolerance class as 153 / 159. Baseline
byte-identical when the flag is off: the new branch is never taken and
`ffn_variant` keeps its default cascade.

**Run command**
```bash
cd /Users/vukrosic/my-life/llm-research-kit-scaling
/venv/main/bin/python _arq_170-swiglu-ffn.py
```

**Final val loss readout**
- The runner prints `val_loss` to stdout at the eval milestones (every 25–100
  steps via `eval_milestones` on `Tiny1M3MConfig`); the GPU queue copies the
  final `val_loss` into `autoresearch/runs/170-swiglu-ffn/metrics.json` via
  the existing harness. Compare to baseline cached at
  `autoresearch/baseline-cache.json` (val ≈ 6.43 ± 0.04 on Vast V100).
