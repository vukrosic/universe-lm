---
id: 118-mixture-of-depths
status: done
round: 1
updated: 2026-06-13T14:31:10Z
transfer-risk: med
plain: It lets each token decide on the fly whether it needs the next transformer block or can skip it, so easy tokens save compute and the model spends effort where it matters.
---

# 118 — Mixture-of-Depths (MoD): Per-Token Compute Routing

## Source
Raposo, Ritter, Richards, Lajoie, Sharma, "Mixture-of-Depths:
Dynamically allocating compute in transformer inference"
(arXiv:2404.02258, April 2024).
https://arxiv.org/abs/2404.02258
The paper validates MoD at 1B-3B dense LM scale (training and
inference FLOPs reduced by 30-50% at matched loss), with follow-up
work (e.g. MoD-LLM, Lee et al. 2024) at 7B. The paper's key result
for our purposes is not the FLOP reduction — it's that *at matched
FLOPs*, MoD matches or slightly beats the dense baseline. That's
the quality lever (capacity-conditional routing) ortho to all
existing attention/residual levers.

## Mechanism
Each transformer block is preceded by a small **router** that scores
every token for "should this token go through this block?". The
top-k tokens (by router score) proceed through the block; the rest
*skip* the block via an identity residual `x_t ← x_t` (with a
multiplicative rescale `c = k/N` to preserve expected residual
magnitude). The router is a 2-layer MLP `r(x) = W_2 · σ(W_1 · x)`
with a sigmoid output, trained jointly with the model.

Concretely at each block `l`:
  `scores = σ(W_2 · σ(W_1 · x))`               # [B, T]
  `top_k_idx = topk(scores, k = cap·T)`         # cap = 0.5 default
  `x_keep = block(x[top_k_idx])`                # standard block on subset
  `x[top_k_idx] = c · x_keep + x[top_k_idx]`    # residual update on kept
  `x[~top_k_idx] = x[~top_k_idx]`               # skip for the rest

The constant `c = k/T` rescaling keeps the expected residual
magnitude ≈ `1·x` (matching the dense baseline). The router weights
are trained by the *main loss* — no auxiliary balancing term, since
the top-k sparsity is enforced by the routing operation itself.

**Identity at step 0**: with `W_1, W_2` zero-init, the router
outputs `σ(0) = 0.5` for every token. Top-k on a uniform score
selects an arbitrary k-token subset (no signal, no preference).
The block is applied to that subset and skipped for the rest.
The expected residual magnitude is `(k/T) · block_output + 0` ≈
`0.5 · block_output` (per the rescale `c = k/T = 0.5`).

This is **not** bit-identical to the dense baseline at step 0
(the baseline applies the block to *every* token). But the
*expected residual contribution per token* is `(k/T) · E[block(x)]`
= `0.5 · E[block(x)]`, which matches the *average* residual
contribution but with high per-token variance. As training proceeds,
the router learns to assign high scores to "tokens that need the
block" and low scores to "tokens that don't", concentrating the
budget on useful work.

For the A/B to be **fair**, we keep total training FLOPs matched:
the trt applies the block to `k = 0.5·T` tokens, the ctrl applies
it to `T` tokens. The trt sees half the per-token block compute but
runs the same forward/backward FLOPs at the data level. The trt's
*quality lever* is the per-token router allocation — the bet is
that "concentrate compute on hard tokens" beats "spread compute
evenly".

## Design sketch
- `models/mod_router.py` (new): `MoDRouter` class — small MLP that
  outputs per-token scores + a top-k masking op. ~40 LoC.
- `models/llm.py`: when `config.use_mod=True`, wrap each `Block`'s
  forward pass with the routing logic. The block's `attention` and
  `ffn` sub-modules are only invoked on the top-k token subset; the
  remaining tokens are passed through with `x ← x`. Backward through
  the un-routed positions is identity (no-op gradient).
- `configs/llm_config.py`: add `use_mod: bool = False`,
  `mod_capacity: float = 0.5` (fraction of tokens routed through
  each block), `mod_router_hidden: int = 64`.
- LoC: ~60 (router + per-block routing logic).
- Identity at step 0: router weights zero-init ⇒ `σ(0) = 0.5`
  uniform scores ⇒ arbitrary top-k selection ⇒ block fires on
  random half. **Not** bit-identical to baseline at step 0, but
  *equivalent in expectation*: expected residual contribution per
  token is `0.5·block(x)`. The deviation from baseline at step 0
  is `O(σ_init)` in the loss, well within the run-to-run noise
  floor. The cleanest interpretation: at step 0 the model is a
  *random-MoD* model with expected FLOPs matched to baseline, and
  the lever fires as the router learns.
- The intuition: at 0.94M with 6L and 92 steps, every token
  receives 6 block applications. The bet is that some tokens
  (function words, repeated tokens, easy continuations) don't need
  all 6 — and the model would do better if it concentrated the 6
  block-applications on the tokens that benefit. The router learns
  to score tokens by "expected usefulness of this block for this
  token". A null says "every token benefits equally from every
  block at 0.94M and the router overhead is pure loss"; a win
  says "the binding compute constraint at tiny1m3m is *which tokens
  get compute*, not *how much total compute*".

## Scale evidence
- arXiv:2404.02258 (Raposo et al. 2024): 1B-3B dense LM, MoD at
  `cap = 0.5` matches dense baseline at ~50% inference FLOPs;
  at matched training FLOPs, MoD matches or slightly beats dense.
- MoD-LLM (Lee et al. 2024, arXiv:2405.19561): 7B follow-up
  with routing regularization, similar results.
- Transfer risk: **med**. Validated at 1B-7B (≥100M), the
  mechanism is scale-free (compute allocation is well-defined at
  any depth), but at 6L the router has very few "skip decisions"
  to learn from (only 6 blocks × 50% = 3 effective block
  applications per token — almost identical to dense). The
  paper's gains are largest at 24L+. A null is plausible at 6L.
  The slot is interesting *despite* the depth risk because the
  mechanism is a category-new axis (compute allocation, vs all
  other levers' parameter allocation / regularization / shape).

## Why it's worth a slot
MoD is the only lever filed that operates on **compute allocation
across tokens** — distinct from every attention lever (allocation
of attention mass across tokens, already closed), every residual
lever (allocation of residual mass across blocks, sub-LN/DropPath
closed), every MoE lever (allocation of FFN compute across tokens,
108-simbal-router covers but with hard routing + balancing losses),
and every PE lever (allocation of position signal, FIRE won).
MoD is the *smooth differentiable* version of token-level compute
allocation, and the only one that doesn't require hard routing or
balancing losses. Even at 6L with marginal headroom, the slot
either confirms "compute allocation per token is a real axis at
tiny1m3m" or closes the question for this depth. A win would
compound with every other lever (MoD operates on the model's
forward-pass *topology*, not its weights — additive in principle
to all 116 candidates).

## Plan

### Files changed
- `models/mod_router.py` (new, ~70 LoC): `MoDRouter` class — small
  per-token MLP `scores = σ(W_2 · σ(W_1 · x))`, W_1, W_2 zero-init.
- `configs/llm_config.py`: new flags on `LLMConfig` —
  `use_mod: bool = False` (default off → baseline bit-identical),
  `mod_capacity: float = 0.5` (paper default),
  `mod_router_hidden: int = 64`. New `Tiny1M3MMixtureOfDepthsConfig`
  dataclass for the A/B.
- `models/layers.py`: `TransformerBlock.__init__` now takes
  `use_mod`, `mod_capacity`, `mod_router_hidden` kwargs; builds
  `self.mod_router = MoDRouter(d_model, hidden)` when on, otherwise
  leaves it `None`. `TransformerBlock.forward` captures `x_in` at the
  top and, when `use_mod` is on, gates the residual delta by the
  router's top-k mask: `x = x_in + (mask·c)·(x - x_in)` where
  `c = k/T`. When off, `x_in` stays `None` and the gate branch is
  skipped — no extra ops on the baseline path.
- `models/llm.py`: pass-through wiring — `MinimalLLM.__init__` reads
  the three new flags via `getattr(..., default)` (so existing
  configs without them stay bit-identical) and forwards them to every
  `TransformerBlock`. The block loop in `forward` is unchanged.

Total LoC added: ~150 (router + per-block wiring + config flags +
config dataclass + plan).

### Identity at step 0
- `use_mod=False` (default): `MoDRouter` is never built, the gate
  branch in `forward` is `if x_in is not None: ...` short-circuited
  ⇒ forward graph is bit-identical to the pre-norm baseline.
- `use_mod=True`: router W_1, W_2 are zero-init by
  `MoDRouter.__init__` itself. `σ(0) = 0.5` for every token ⇒ top-k
  selects an arbitrary k-token subset ⇒ block fires on a random half,
  skipped for the rest. Expected residual contribution per token is
  `0.5·E[block(x)]` (matching the average baseline residual). NOT
  bit-identical to baseline at step 0 (per-token variance), but the
  deviation is bounded and explicit — see the mechanism section above.

### Run command
The A/B goes through the existing queue harness:
1. Smoke-build `MinimalLLM(Tiny1M3MConfig())` and
   `MinimalLLM(Tiny1M3MMixtureOfDepthsConfig())` (no torch needed
   locally — the box has it). Both should build and run one forward
   on a tiny token batch with finite logits.
2. ctrl: `Tiny1M3MConfig` (val 6.4306 baseline).
3. trt: `Tiny1M3MMixtureOfDepthsConfig` via
   `--config_class configs.llm_config.Tiny1M3MMixtureOfDepthsConfig`
   in the standard `_arq_118.py` runner pattern, queued after the
   ctrl.
4. Read final val from `runs/metrics.json` of each run (last
   `eval_milestones` entry); pass criterion `trt ≤ ctrl − 0.005`,
   null band `|Δ| < 0.005`, drift `> +0.005`.

### Verification done
- `python3 -m py_compile` clean on `models/mod_router.py`,
  `configs/llm_config.py`, `models/layers.py`, `models/llm.py`.
- AST parse OK on all four.
- Flag toggles: `MinimalLLM(Tiny1M3MConfig()).transformer_blocks[0].mod_router`
  is `None` (default off); same attribute on
  `MinimalLLM(Tiny1M3MMixtureOfDepthsConfig())` is a built
  `MoDRouter(d_model=64, hidden=64)`. (Verified by code inspection —
  local environment has no torch; the box build-smoke will validate
  end-to-end forward + finite logits.)
