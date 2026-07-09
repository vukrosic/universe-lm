## r1 — 2026-06-16 — verdict: accept

**Leverage.** A 0-param, init-time-only fixed global scalar — cheapest lever in the pipeline. If it wins, it's a 0-cost init prior with 200L+ theoretical backing (DeepNet) and 100M+ empirical backing (Primer's learned form). If it nulls, it cleanly closes the *fixed-form* axis (separate from the *learned per-block* axis). Either way the slot pays for itself.

**Non-obviousness / novelty.** The depth-conditional residual-scaling family has 5 closed levers (017 Sub-LN-sandwich, 111 drop-path, 116 hyper-connections, 130 ReZero, 142 LayerScale) — all *learned* per-block or per-channel. 197 is the **first fixed global scalar** in that family. That's a real mechanism split, not a tweak: learned = "let the optimizer specialize per block"; fixed = "impose the theoretical prior and force the network to operate in the bounded regime from step 0." Two different bets, two different failure modes.

**Crisp bet.** "Bounded residual magnitude (O(1) throughout) eliminates depth-amplified activation/gradient drift, even at 12L." The mechanism is well-posed: at L=12, α = 1/√24 ≈ 0.204, so per-component residual growth drops from ~3.5× to ~0.7×. The LM head sees a well-conditioned input regardless of depth.

**Information value.** High. Two reasons:
- The lever has *no optimizer freedom* (no per-block learned scalar), so a null is unambiguous — not "optimizer didn't use the lever" but "this prior doesn't help at this tier." A null closes the *fixed-form* axis at 0.94M with no equivocation.
- A win is 0-cost and theoretically motivated — exactly the kind of lever that compounds into the 135M recipe.

**Portfolio fit.** The broader family is crowded (196 was just rejected as "5th in family"). But 196 was cross-block residual-mixing (different mechanism). The depth-conditional *init* family at this slot count is 017/130/142 — three closed learned forms. 197 is the first fixed form. The axis is closed, the *form* is not. Acceptable diversification, not a re-tread.

**Niche fit.** Clean: 0 new params, init-time only, mechanism (not HP), identity-able (single global scalar is a definable prior), tiny1m3m-runnable (just a multiply on attn_out / ffn_out). ✓

**Transfer risk.** low — correct. The lever is validated at 200L+ (DeepNet, arXiv:2203.00555) and the *learned* form at 100M+ (Primer, So et al. 2021). At 12L the absolute magnitude of the depth-drift being fixed is small (sqrt(12) ≈ 3.5×, vs sqrt(200) ≈ 14×), so the lever's *expected effect* at our tier is correspondingly smaller — but the *direction* is theoretically supported.

**One concern (noted, not blocking).** The 0.204 scale is aggressive at 12L — the residual stream is starved of signal. If 0.204 is too small, the model may under-converge and DRIFT hard. That's a clean signal (the lever is too coarse for this depth), not a code/plan problem. The A/B remains informative either way.

**Verdict.** Accept. Reset to round 1 for the definition gate's budget.
