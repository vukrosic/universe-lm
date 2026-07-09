---
id: 060-ngpt
status: needs-plan
round: 2
updated: 2026-06-11T01:19:47Z
transfer-risk: med
---

# 060 — nGPT (hypersphere-normalized Transformer)

## Source
Loshchilov, Hsieh, Sun, Ginsburg, "nGPT: Normalized Transformer with
Representation Learning on the Hypersphere" (arXiv:2410.01131).

## Mechanism
Constrain every "vector that moves information" to unit L2-norm so the
token representation evolves on a hypersphere instead of an
unconstrained residual plane. This A/B runs the **normalization core
only** (see Spec); the paper's learnable eigen-rate updates and
per-vector scalar scales are explicitly dropped to fit the LoC budget
at tiny1m3m.

## Spec — nGPT-core (committed subset)

Treatment normalizes five vector sites to unit L2 along their feature
dim; ctrl has none of these.

| # | Site | Where | Op |
|---|---|---|---|
| 1 | Token-embedding rows | `models/llm.py` `MinimalLLM.__init__` + post-step hook | `W_emb ← W_emb / ‖W_emb‖₂` per row |
| 2 | Residual-stream state | `models/llm.py` `MinimalLLM.forward` block loop, after each `block(x, …)` | `x ← x / ‖x‖₂` per token |
| 3 | Per-head Q, K (post-RoPE) | `models/layers.py` `MultiHeadAttention.forward`, after `_manual_rope`/SDPA RoPE | `q ← q / ‖q‖₂`, `k ← k / ‖k‖₂` per head per token |
| 4 | MLP up/down rows | `models/layers.py` `TransformerBlock` MLP linear weights, post-step hook | row-wise unit norm on `W_up`, `W_down` |
| 5 | Output unembed rows | `models/llm.py` lm_head (tied with embedding — same op as site 1) | row-wise unit norm |

**Explicitly dropped** (to keep < 200 LoC and isolate the manifold
question from compensating HPs):
- Learnable per-block eigen-rates α_A, α_M (paper's controlled-step
  residual update `x ← x + α·(h − x)` with `h` normalized).
- Removing the `sqrt(d_k)` scale inside attention. We keep
  `softmax(QK^T / sqrt(d_k))`.
- Learnable per-vector scales `s_z`, `s_u`, `s_v` (the paper uses these
  to recover effective magnitude after every normalization).

**Stacking vs 016 QK-Norm (closed WIN):** ctrl `Tiny1M3MConfig` does
**not** enable 016 (016 lives in the separate `Tiny1M3MQKNormConfig`
subclass at `configs/llm_config.py:733`). nGPT-core *replaces* the
would-be 016 lever with a unit-L2 norm of Q/K **post-RoPE** (016 was
LayerNorm, **pre-RoPE**). The A/B therefore measures the full
hypersphere bundle against an unnormalized baseline — no QK-Norm
stacking ambiguity.

## Control

- ctrl: `Tiny1M3MConfig` (LEADERBOARD ≈ 6.4044/6.4091; see `closed.md`
  016/017 lines).
- trt: `Tiny1M3MNGPTConfig(Tiny1M3MConfig)` with one bool flag
  `use_ngpt: bool = True`; the code gate plans the actual flag wiring.
- **Not identity-initable** — hypersphere is a binary constraint, so
  the ctrl is *not* `use_ngpt=False` of the same checkpoint at step 0;
  it is the plain unconstrained Tiny1M3MConfig run. Taste flagged this
  (`taste.md:8`); the reviewer ratified.

## Learning rate

Keep base Muon/AdamW LRs from `Tiny1M3MConfig` unchanged. **Bet:**
unit-norming at tiny1m3m does not move the effective optimizer scale
enough to require rescaling; the paper's `s_z`/eigen-rates partly
compensate for the binary constraint, and we are dropping those, so a
silent LR mismatch is still less of a confound than introducing a new
LR knob that conflates LR-tuning with the manifold question. If the
treatment underflows visibly (loss diverging in the first ~25 steps),
the code gate may add a runner-side note, but the seed-42 A/B as
defined keeps base LR.

## Pass / fail bar

- Δval = trt_val − ctrl_val, both at the final eval milestone of
  `Tiny1M3MConfig` (seed 42, tiny1m3m box).
- **WIN**: Δval ≤ −0.01 (clears the ~±0.01 tiny1m3m box noise floor
  and the two-ctrl gap ≈ 0.0047 observed in 016/017 runs).
- **NULL**: |Δval| < 0.01 (inconclusive at this tier; do not "add
  seeds to confirm" — per the pipeline's seed-42-only rule).
- **FAIL**: Δval ≥ +0.01 (manifold constraint actively hurts at this
  scale; log to `closed.md`).
- Box-validation rule (from `PIPELINE.md`): if ctrl drifts > 0.01 vs
  LEADERBOARD, the box is bad — result not trusted, stays `needs-run`.

## LoC budget (target < 200, estimated ≈ 35)

| Site | File | Est. LoC |
|---|---|---|
| Flag + Tiny1M3MNGPTConfig subclass | `configs/llm_config.py` | ~10 |
| Embed + unembed row-norm (init + post-step hook) | `models/llm.py` | ~8 |
| Residual-state unit-norm in block loop | `models/llm.py` | ~3 |
| Q, K post-RoPE unit-norm in MHA | `models/layers.py` (~1294–1645 RoPE region) | ~6 |
| MLP up/down row-norm hook | `models/layers.py` `TransformerBlock` MLP | ~6 |
| **Total** | | **~33** |

MLP/embed/unembed row-norms run as a post-`optimizer.step()` hook
(re-normalize in place); the Q/K and residual-state norms run inside
`forward`. The code gate refines exact wiring.

## Scale evidence

Loshchilov et al. (arXiv:2410.01131) report that the normalized
Transformer reaches the same accuracy in **4×–20× fewer training
steps**, with the multiplier growing with sequence length. The abstract
confirms the 4–20× claim and the LM target; the specific model sizes
(0.5B and 1B on OpenWebText) are cited from common summaries of the
paper body rather than re-verified from the PDF in this gate.
transfer-risk: **med** — scale evidence is direct and large, but
mechanism is invasive (five vector sites) and the LoC-minimal subset
drops the paper's compensating scalar scales, so implementation risk
dominates scale risk.

## Why it's worth a slot

If hypersphere training helps here, then this codebase is spending
capacity on unconstrained residual-stream norms; if it fails, we learn
that the full normalized-Transformer geometry needs more scale/steps
than tiny1m3m offers (informs the 135M decision). The norm-cluster
siblings (051/052/055–059) each touch one tensor; nGPT-core is the
only end-to-end manifold-constrainer in the queue (per `taste.md:6`).
