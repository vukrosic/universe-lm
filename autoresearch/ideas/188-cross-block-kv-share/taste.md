## r1 — 2026-06-15 — verdict: accept

**Bet (one sentence):** Adjacent attention blocks redundantly re-learn similar K/V projection subspaces; a per-block learnable blend toward the previous block's projection (sigmoid α init ≈ 0) regularizes depth and improves val loss.

**Why the slot is earned:**
- **Genuinely new axis.** In-repo V-family is residual-stream/output level (021 V-residual, 168 AV-output-carry, 164 Q-residual, 186 within-block V time-axis). 188 mixes the **W_K / W_V projections themselves** — that's projection-level, not stream-level. Universal Transformers (Dehghani 2019, <100M) do this globally & hard; 188 does it adjacent-only & soft with a learnable scalar. Spirit-share with UT, mechanically distinct from the closed in-repo V-side variants.
- **Identity-init clean.** `α_raw = -10` ⇒ `sigmoid ≈ 4.5e-5` ⇒ `W_eff ≈ W_self` at step 0 (forward graph unchanged up to fp32 noise). Passes baseline-gate trivially. Two scalars per block × 12 = 24 params (0.003% of 0.94M) — negligible.
- **Self-correcting under transfer risk.** At larger scales, if parameter sharing hurts, α can fall toward 0 and the model collapses back to the baseline. The lever only pays the sharing tax if training finds it useful — no forced commitment to UT-style tying.
- **Information value is high regardless of outcome.** A win at 0.94M would justify carrying to 135M (a real depth-regularization lever). A clean null would localize V-side cross-block gains to the *residual stream only* (consistent with 168-AV-carry null), and would close the projection-sharing axis at this tier — both outcomes are useful.

**Caveats (not blocking, noted for the review gate):**
- Family is crowded with V-side/KV-side variants. 188 is the 5th, but the projection-level axis is genuinely orthogonal.
- `detach()` on `prev_W_K` is essential (gradient must not flow back through the previous block's projection, else this becomes a depth-recurrent net and behavior diverges sharply).
- transfer-risk: med is honest. UT-style gains validated at <100M; the soft-α formulation is plausible at 135M but unproven.

**Verdict:** `accept` → routes to `needs-review`. Round reset to 1 for the definition gate's own budget.
