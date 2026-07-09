# Code-review log — 004 retnet-retention

## r3 — 2026-06-09 — verdict: accept

**Round 3 cap — forced call. Blocking finding from r2 (3-seed protocol) is fixed. LoC accounting is now honestly stated, not hidden. Code unchanged from r2 (the r2 fix was a doc fix, not a code fix; the kernel + probe were already correct). Accepting v1 (kernel + synthetic probe) to `needs-run` for the pytest + probe pre-flight.**

**Re-check of r2 findings:**

- **🔴 3-seed protocol (r2 blocking) — FIXED.** `idea.md:39` now reads: "Seed 42, single seed, per the pipeline hard rule. |Δ| ≤ 0.10 is the noise band; a sub-noise result is logged inconclusive, not re-seeded. A null at |Δ| < 0.04 with seed 42 is *itself* the evidence the kernel doesn't catch up at this scale." `plan.md` Step 2 closes with: "The v2 PR's seed is 42, single seed, per the pipeline hard rule." Both doc fix lands. ✓
- **LoC accounting off by ~2× (r2 minor) — FIXED.** `plan.md` §Self-check now explicitly states: "kernel 106 + probe 87 + config ~12 = ~205 LoC production (excluding `tests/test_retention.py` 104 LoC, which is test code by convention and not counted in the budget). The 205 is at the original < 200 LoC ceiling the r1 review committed to, slightly over by ~5 LoC — within the 250 review ceiling. The v1 plan §"LoC estimate" originally said 150; that was an undercount (actual is 205 prod). The math, tests, and routing are unchanged; only the accounting is corrected." The implementer is honest about being 5 LoC over the original < 200 budget but under the 250 review ceiling — exactly the right disposition. ✓
- **Test invariant count off by one (r2 minor) — FIXED.** `plan.md` §Step 1 line 64 now says "Pass/fail = 4 invariants (no NaN, causal, per-head independence, γ-monotone-in-t)" — matches the actual `tests/test_retention.py` file (4 tests, not 3). ✓

**Identity re-check (no code changed, but noting for the runner):**
- `models/retention.py` — kernel never instantiated by the model in v1. `use_retention=False` (default) is the only path; `models/layers.py` has no `use_retention` consumer. Baseline path is bit-identical. ✓
- The v1 PR is a **kernel + probe + test**, NOT a downstream A/B. The "treatment" of any A/B is deferred to v2. This is the r2-authorized scope decision.

**Verdict:** `accept`. Status → `needs-run`. The runner executes the v1 pre-flight: `pytest tests/test_retention.py -v` + `python autoresearch/ideas/004-retnet-retention/probe.py`. If any of the 4 invariants fails, the runner files a `codereview.md` finding and the idea bounces back; if all pass, v1 closes and the v2 production-wiring PR becomes the next bet.

## r2 — 2026-06-09 — verdict: revise

**v1 ships a clean kernel + probe as scoped, but the seed protocol is malformed and the plan's LoC accounting materially understates the diff.**

**What I checked:**
- `models/retention.py` (106 LoC including docstring) — `RetentionKernel` with per-head learnable γ (sigmoid-mapped raw, init γ=0.99), log-space causal mask, no softmax, no `/sqrt(d_k)`. Math matches the RetNet parallel retention formula.
- `autoresearch/ideas/004-retnet-retention/probe.py` (87 LoC) — synthetic probe with 3 invariants (no NaN/Inf, causal, per-head independence) + an extra γ ∈ (0,1) check.
- `tests/test_retention.py` (104 LoC) — pytest version with **4** invariants (the 3 plan-committed ones + a decay-monotone-in-t check).
- `configs/llm_config.py:380-387` — `use_retention: bool = False` flag, on the line after `use_soap_precondition_freq`, with a v1-is-kernel-only docstring note.
- `models/layers.py` — UNTOUCHED in v1. ✓ (the v1 plan explicitly defers the `MultiHeadAttention` rewrite to v2)
- `models/llm.py` — UNTOUCHED. ✓
- The `use_retention` flag is *unused* in v1 (no consumer in the model). With `use_retention=False` (default), the kernel is never instantiated, no forward pass goes through it, the baseline path is bit-identical. ✓

**Findings:**

- **🔴 REVISE-BLOCKING — 3-seed protocol violates the ONE SEED ONLY rule.** `idea.md` §"Seed protocol" says "3 seeds (42/43/44). The lower 80% of the original expected range (|Δ| ≤ 0.04) is unresolvable on a single seed." `plan.md` §"Step 2 — v2 production wiring" re-commits the 3-seed protocol for the v2 PR. **PIPELINE.md hard rule:** "Every ablation runs at a **single fixed seed (42)** — never multi-seed. ... Any idea, plan, or review that asks for more than one seed is **malformed** — strip it down to seed 42 instead. Read a sub-noise effect as **inconclusive, not real**." **Fix:** delete the 3-seed paragraph from `idea.md` and replace with: "Seed 42, single seed, per the pipeline hard rule. |Δ| ≤ 0.10 is the noise band; a sub-noise result is logged inconclusive, not re-seeded." Update `plan.md` §Step 2 to the same single-seed commitment. The honest transfer story ("expect null or marginal at screen20m") is still true and unchanged — a null at |Δ| < 0.04 with seed 42 is *itself* the evidence the kernel doesn't catch up at this scale.

- **Plan §"LoC budget" understates the actual diff by ~2×.** Plan claims "kernel ~55 + probe ~60 + test ~30 + config ~5 = ~150 LoC total." Actual file sizes:
  - `models/retention.py`: **106 LoC** (not 55)
  - `autoresearch/ideas/004-retnet-retention/probe.py`: **87 LoC** (not 60)
  - `tests/test_retention.py`: **104 LoC** (not 30; the file has 4 invariants, not the 3 the plan listed)
  - `configs/llm_config.py` (4 lines + 7-line docstring for `use_retention`): **~12 LoC** (not 5)
  - **Total: ~309 LoC**, not 150. Even excluding the pytest test file (not part of the production diff, by convention), the production code is 106 + 87 + 12 = **205 LoC** — over the < 200 LoC budget the r1 review committed to. The original idea said "< 200 LoC" as a dedup criterion. The plan is now > 200 LoC. **Fix options (pick one):** (a) trim `retention.py` to the kernel math by removing the docstring-to-code duplication and the per-head independence test data shape comments (target ~70 LoC); or (b) bump the LoC budget in the plan/idea to "~200 LoC production + ~110 LoC test/probe" and acknowledge the v1 is at the original 200 ceiling, not the 150 the plan claimed. The math is fine either way; the plan just needs to match reality.

- **Test invariant count off by one (informational).** The plan §"Step 1" promises "3 invariants (no NaN, causal, per-head independence)." The actual test file has 4 (the 3 + `test_decay_monotone_in_t`). The extra test is valuable (it guards the kernel's decay behavior) — just update the plan to say 4, or drop the test if 3 is the budget. Not blocking.

- **Routing lever is a no-op in v1 (informational, already documented).** `use_retention` is added to `LLMConfig` but no code path reads it. v1 is a kernel + probe only; v2 will wire it into `MultiHeadAttention.forward`. The plan §"Self-check" and §"Cost" both acknowledge this. The "treatment" of any A/B is deferred to v2. This is a sound scope decision — the r2 review explicitly authorized it — but the reviewer note is: a v1 review cannot evaluate the `idea.md` pass/fail bar, only the kernel's correctness on the repo's tensor shapes. That's exactly what the v1 plan claims to do, and the kernel + tests do it. No code change needed.

- **`test_per_head_independence` allows a 3.0 raw shift to validate head isolation (informational).** The test perturbs `kernel_b.gamma_raw[2] += 3.0` (large enough to clearly differ). Sigmoid(0.9) ≈ 0.9 → sigmoid(0.9 + 3.0) ≈ 0.998. The Δ is large and the test will pass deterministically. Fine, just a note that the threshold (1e-5 vs 1e-3) is asymmetric on purpose.

**Plan ↔ idea consistency check:**
- pass bar: plan ≤ 4.5864, idea ≤ 4.5864 ✓
- fail: plan > 4.6364, idea > 4.6364 ✓
- noise: plan |Δ| ≤ 0.10, idea |Δ| ≤ 0.10 ✓
- control: V+q+SWA+HighRoPE on both ✓
- routing: "replace attention module only" on both ✓
- LoC budget: plan 150, idea "<200" — plan undercounted (see finding 2)

**Coordination check:**
- The untracked files `models/retention.py` and `tests/test_retention.py` are the v1 PR's own files — coherent with the plan, not stomped.
- `configs/llm_config.py` adds `use_retention` immediately after the SOAP flag block — adjacent placement matches the r1 spec, no clash with the parallel-Claude's other PRs.
- `git diff models/layers.py` shows 0 lines changed. ✓

**Hand-off to code-implementer:** the kernel is correct, the tests are good, the routing decision is honest. Strip the 3-seed protocol from `idea.md` and `plan.md` (the only blocking fix). Reconcile the LoC accounting (trim or re-state). Re-submit and re-review (round 3, the cap — next pass must be `accept` or `reject`).

## r1 — 2026-06-08 — verdict: revise
(implementation review — code was not yet at this round; placeholder for completeness)

## r0 — (none)
