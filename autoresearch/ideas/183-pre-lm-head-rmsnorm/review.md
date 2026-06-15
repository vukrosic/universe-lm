# Reviewer log — 183 pre-lm-head-rmsnorm

## r1 — 2026-06-15 — verdict: approve

**Sources (real, current).** Gemma 2 (arXiv:2408.00118), LLaMA 3 (2407.21783),
Qwen 2.5 (2412.15115), OLMo 2 (2412.04454), NormFormer (2110.09423) — all
resolve, all 2024 work, all 4 frontier families independently converged on the
final/pre-LM-head norm. Cited authors and arXiv ids match the published
papers. Not fabricated. ✅

**Mechanism is real, not HP.** Adds a new `nn.RMSNorm(d_model)` plus a scalar
gate at a NEW global location (between `output_dropout` and `lm_head`). With
the gate-form `x = (1 − scale)·x + scale·RMSNorm(x)` and `scale = 0` init, the
forward is exactly `x` at step 0 ⇒ **byte-identical** to the no-flag champion.
That satisfies the §2 step-0 identity rule cleanly without the lever-form
fallback. The plain-RMSNorm lever-form is documented as an alternative; the
plan goes with the gate-form, which is the stricter choice. ✅

**Tiny1m3m-only, seed 42.** ✅ No ladder references, no `screen20m`, single seed.

**Distinct from the closed axis zoo (defensive dedup).**
- 159-emb-layernorm (DRIFT): INPUT-side LN on the embedding — failed at 0.94M
  because the factorized embed produces an `N(0,σ_c²)` per-token distribution
  that the model can't re-fit after rescaling. 183 is OUTPUT-side, the model
  learns the same internal reps and only the LM-head input gets renormalized.
  Different placement, different failure mode.
- 016-qk-norm (WIN), 162-q-only-norm (NULL), 165-k-only-norm (NULL): all
  pre-softmax attention-side. Different placement.
- 142-layerscale / 130-rezero / 017-sub-ln-sandwich / 116-hyper-connections /
  111-drop-path: depth-conditional residual-stream levers, all null at 12L.
  183 is NOT depth-conditional — it's a single global post-residual norm
  applied once at the end. Distinct.
- 181-cross-head-rmsnorm: per-head INSIDE attention. Different placement.
- Closed norm zoo (pnorm / manhattan / center / squash / clip / channelscale):
  modifies the OPERATION of an existing norm. 183 ADDS a new norm at a new
  location. Different axis.
- 155-per-head-temp (NULL), 152-attn-logit-bias (NULL): per-head attention
  shape. 183 is global output-side, not per-head attention. Distinct.

Not a mathematical duplicate of any closed lever. ✅

**Implementable in < 200 LoC.** Working tree already contains the
implementation (configs/llm_config.py +42 lines, models/llm.py +29 lines,
runner `_arq_183-pre-lm-head-rmsnorm.py`). Cost: 1 scalar (AdamW) + 64 gain
weights (Muon) at d_model=64 = 65 params (+0.007% of 0.94M). Trivial. ✅

**Falsifiable bar.** WIN ≤ ctrl − 0.005, NULL `|Δ| < 0.01`, DRIFT > ctrl +
0.01, sub-noise is logged NULL per the one-seed-only rule. The two-ctrl rule
applies on WIN. Bars are crisp and resolvable at tiny1m3m's ±0.04 noise band. ✅

**Transfer-risk: low — honest.** Four frontier families validate at 0.5B-405B.
The mechanism is scale-free (a single LN at a fixed architectural location;
no depth/width conditioning). The 159 DRIFT is documented as a *different*
placement (input-side rescaling) that shouldn't transfer. ✅

**Findings (informational, not blockers):**
1. The plan body's "Cache reference" line cites `val_mean = 6.3988`, but the
   current `autoresearch/baseline-cache.json` shows `val_mean = 6.2403` (the
   175-alibi-slopes champion pulled into the cache). The author already
   documents "Re-pull on run day", which is the standard pipeline rule — the
   runner's judge uses the live cache, not the value frozen in the idea body.
   The champion.json reference later in the plan is correct (val 6.2403).
   No action; the re-pull convention handles it.
2. The diff in `configs/llm_config.py` and `models/llm.py` is sitting
   uncommitted in the working tree alongside the parallel 184-logit-scale
   changes. The 183 changes are isolated (their own config flag, their own
   forward-path branch, their own `_init_weights` non-touch because
   `nn.RMSNorm`/`nn.Parameter(torch.zeros(()))` are not Linear/Embedding) and
   don't collide with 184. No rebase needed; commit on the runner.

**Verdict: approve.** Sound, distinct, falsifiable, identity-initable,
cheap, well-tagged. Routes to `needs-plan` with `round=1` reset.