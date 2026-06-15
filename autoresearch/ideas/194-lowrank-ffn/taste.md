## r1 — 2026-06-15 — verdict: revise

**Taste gap (one sentence):** the FFN-shape axis is *exhausted* at 0.94M (6+ closed nulls — 146, 153, 157, 158, 170, plus 117/118/145 on the MoE sub-axis), and the pitch itself predicts null ("A null at 0.94M is expected") — that's not a bet, it's a coin-flip asking for a scarce slot.

**Findings:**

- **Portfolio fit fails.** Closed.md reads "FFN capacity levers don't bind at this tier" four times over (146, 156, 158, 170). Closed nulls on FFN-side shape: 146-sparse-ffn (Δ=+0.0057), 153-relu2-ffn (Δ=−0.0053), 157-conv-ffn (Δ=−0.0078), 158-gau (Δ=+0.1095 DRIFT), 170-swiglu-ffn (Δ=−0.0170). Closed nulls on FFN-MoE capacity: 117-soft-moe, 118-MoD, 145-expert-choice. The protocol is explicit: "5th optimizer-momentum variant in a row is a `revise` (diversify) even if each is individually fine" — we are well past the 5th on FFN.
- **Author hedged their own bet.** "Why it's worth a slot" closes with "A null at 0.94M is expected (FFN-capacity levers don't bind); a win would mean the FFN has specific low-rank structure that the optimizer can leverage." Asking for a slot for a *predicted* null is a taste red flag — the experiment is null-by-default and a surprise win is the only informative outcome, but surprise wins on a closed-out axis are not load-bearing evidence.
- **The mechanism is technically distinct but the binding constraint is the same.** Low-rank FFN correction is not the same lever as sparse-MoE or SwiGLU gating — it's a residual additive rank-r path. But all three push on the same *binding constraint* (FFN effective capacity at 0.94M/12L/d_model=64), and the closed list shows that constraint isn't binding. The bit-identical-at-step-0 trick is correct (good engineering) but doesn't rescue the bet.
- **Transfer evidence is med with no <100M win.** "No published *training-from-scratch* low-rank FFN win at <100M" — that's a red flag for a tier-strategy that lives or dies on tiny1m3m. LoRA is an *adaptation* lever, not a from-scratch lever; the mechanistic analogy is loose.
- **What would make this `accept` on re-pitch.** Two of the following:
  1. **Reframe as the terminal FFN-axis falsification test** with a sharp pre-registered number — e.g., "If effective FFN rank at step 2k is <32, a rank-16 correction wins; commit to that and treat null as axis-closure." Make the null *informative* by naming what it kills.
  2. **Drop the FFN-side mechanism and propose the same low-rank correction on a different sub-block** (attention Q/K/V/O projection, the residual stream itself, or the embedding-to-hidden lift) where the axis isn't already closed. The mechanism is fine; the placement is the problem.
  3. **Diversify away from the FFN family entirely.** The active queue (188, 189, 190, 191, 192, 193, 195) is concentrated on attention-shape — but the closed list shows attention-side closures too. A truly fresh axis (init scheme, optimizer-side, data-side, normalization topology) would beat yet another structural lever.

**Verdict routing:** `needs-repitch` — re-pitch with one of the three framings above. Round resets to 1 for the miner.
