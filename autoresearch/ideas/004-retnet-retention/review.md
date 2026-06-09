# Review log — 004 retnet-retention

## r2 — 2026-06-08 — verdict: approve

**All 6 r1 findings addressed. Ready for plan.md (with the LoC budget as an active watch-item).**

- **LoC breakdown (r1 finding).** Realistic split: kernel 80 + integration 80 + trainer 40 = ~200 LoC, at the budget edge. Explicit fallback if integration exceeds 250 (split into kernel-only PR + integration PR, or downscope to a synthetic probe). v1 ships parallel + chunkwise; recurrent (inference) is a stub. ✓
- **Routing committed (r1 finding).** "RetNet replaces the attention module only." Per-block list: swapped = `MultiHeadAttention` in `models/layers.py`; kept = V-embed, Q-gain, K-gain, SWA, HighRoPE, LM head, embeddings, FFN, norms. Cleanly isolates "softmax vs retention." ✓
- **Transfer argument (r1 finding).** Dedicated section: O(N) memory/compute of the retention kernel is the scale-invariance argument for 135M+; honest caveat that softmax is very well-tuned at this scale (4.6364 baseline) and the kernel may not catch up. Bottom line: "expect null or marginal at screen20m; expect a compute-advantage at 135M+ if the kernel works at all." ✓
- **Field-moved caveat (r1 finding).** Dedicated section "Field-moved caveat — why RetNet is the right probe": RetNet = most-cited linear-attention baseline with clean math; Mamba = different mechanism (state-space); GLA/RWKV-7 = closer cousins but not in stack. RetNet represents the class. ✓
- **Noise vs expected Δ (r1 finding).** Tightened to "−0.04 to −0.06; lower values are below the single-seed noise floor." 3-seed protocol (42/43/44) committed. Realistic null outcome (|Δ| < 0.04) is still useful. ✓
- **Closed-list migration rule (r1 finding).** Explicit: "A null at screen20m → close the lever, re-test only if a 135M+ compute grant materializes. Do NOT file a Mamba variant as a re-probe of the same lever." ✓

**Hand-off to code-implementer:** promote to `plan.md`. Watch-item: the 250-LoC integration ceiling. If the integration PR trends over 250, split before merging rather than accumulating. Pre-flight (single synthetic run to confirm the kernel works on the repo's tensor shapes) is recommended but not blocking — the bf16 pre-flight precedent from 003 doesn't apply here (no eigendecomp).

## r1 — 2026-06-08 — verdict: revise

**Sound mechanism, clean dedup, honest about staleness — but the doc has 4 uncommitted decisions and one blocker: the LoC estimate is unrealistic for a working attention rewrite, and the transfer argument is required (not optional) per reviewer §2.**

**5-check sweep:**
- **Source real:** Sun et al. (Microsoft) arXiv:2307.08621, Jul 2023. Code at aka.ms/retnet. ✓ (but 2.5 years old — see field-moved caveat)
- **Mechanism is structural:** replacing softmax attention with a retention kernel (learnable per-head decay γ, position-dependent mask, no softmax) is architectural, not a HP lever. ✓
- **Not already closed:** not in `closed.md`; closed axes include MHA/GQA/MLA, NSA/diff-attn/hybrid, multiscale/parallel/attn sink — none of which is "linear attention with retention." ✓
- **< 200 LoC:** doc claims "<200" for the kernel. ⚠️ — the kernel is small; the *integration* with the existing model scaffolding is the LoC driver. See finding below.
- **Falsifiable bar with real control:** V+q+SWA+HighRoPE 4.6364. Pass ≤ 4.5864, fail > 4.6364, noise band |Δ| ≤ 0.10. ✓
- **Transfer argument:** doc flags the risk honestly but does not provide the argument. **Required** for promotion past tiny1m3m. ⚠️ — see finding below.

**Findings (must be addressed before `needs-plan`):**

- **LoC estimate is the kernel only, not the integration.** Doc says "Single Q/K/V projection + custom retention kernel — <200 LoC." That's the new code, not the diff. The actual work includes: (a) integrating with the existing RoPE application point in `models/layers.py` (where does γ × RoPE(Q) go?); (b) deciding whether the recurrent (inference) path ships now or just parallel+chunkwise; (c) the chunkwise path needs a custom CUDA kernel or a careful Triton one for any speedup at all. **Fix:** break down: kernel body ~80 LoC, model integration ~80 LoC (attention block, KV cache stub, RoPE hook), trainer hooks ~40 LoC (no GQA reshape, no flash-attn path). Total realistic: ~200 LoC, **at the budget edge.** If integration exceeds 250, the idea should be split (kernel-only PR + integration PR) or downscoped to a probe (run a tiny synthetic to verify the kernel works, don't try to land it as the production attention).
- **Routing decision not committed.** "Decide before implementing: does RetNet REPLACE the V+q+SWA+HighRoPE stack, or only the attention module (keeping V-embed etc.)? Default: replace attention only." **Fix:** state "replace attention module only" as a line, not a question. V-embed and q_gain/k_gain are orthogonal to the attention math; preserve them. SWA + HighRoPE baseline stays; attention block is swapped. This is the right call.
- **Transfer argument — required, not optional.** Doc says "A 10M win may NOT transfer 25M→135M" and "Paper's strongest results are 7B+." That's the risk, not the argument. **Fix:** add one paragraph. The mechanism: retention is O(N) memory and O(N) compute, so it has a compute advantage at 135M+ where attention is O(N²) memory. *That* is the scale-invariance argument. The risk is that small-scale softmax attention is so well-tuned (the repo has a tight V+q+SWA+HighRoPE baseline at 4.6364) that the retention kernel can't catch up at 10M-20M. The honest transfer story is: "expect null or marginal at screen20m; expect a compute-advantage at 135M+ if the kernel works at all."
- **Field-moved caveat — needs justification, not just an asterisk.** Doc flags "Mamba, GLA, RWKV-7" as newer but files RetNet anyway. **Fix:** add a sentence explaining *why* RetNet is the right probe: (a) RetNet is the most-cited linear-attention baseline with a working impl and clean math, (b) Mamba has a state-space model underneath (different mechanism, not a fair comparison in the "softmax vs retention" question), (c) GLA/RWKV-7 are closer cousins but the repo has neither in stack. If we're testing the *class* of linear-attention alternatives, RetNet is a representative.
- **Noise band vs expected Δ.** Doc's expected Δ is −0.02 to −0.06; noise band is |Δ| ≤ 0.10. The lower 80% of the expected range is unresolvable on a single seed. **Fix:** tighten expected to "−0.04 to −0.06" or commit to a 3-seed protocol. Realistic outcome at this scale: null (|Δ| < 0.04) and the result is still useful (it tells us retention works at the kernel level, just not at the screen20m signal level).
- **No `closed.md` migration rule for linear-attention nulls.** If this lands as a null at screen20m, the doc should state whether to close the lever, file a Mamba variant, or just leave it. **Fix:** add one line: "Null at screen20m → close and re-test only if a 135M+ compute grant materializes; don't file Mamba as a re-probe of the same lever."

**Hand-off to reviser:** all 6 findings are fillable in <30 min. The honesty about the field-moved caveat is good — keep it, but justify the choice. The transfer argument is the most important one; the rest are doc-completeness.

## r0 — (none)
