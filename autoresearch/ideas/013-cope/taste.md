## r1 — 2026-06-09 — verdict: accept
- **Leverage**: real. 009 FIRE PE just won big (Δ −0.064/−0.082) at tiny1m3m, but FIRE is *fixed-decay on absolute position*. CoPE is *content-conditional position* — orthogonal inductive bias. A clean A/B against the same ctrl teaches us whether the FIRE win was about the fixed-decay kernel specifically, or about position-encoding flexibility in general.
- **Niche fit**: identity-safe (probe p init near 0 → near-uniform positions → RoPE-like at step 0), tiny LoC (~40), no compute cost. Drop-in swap for RoPE in `models/attention.py`.
- **Information value either way**: yes. Win → content-conditional lever. Null → FIRE's mechanism is the one that matters; CoPE adds nothing.
- **Risk noted, not blocking**: the hypothesis Δ is small (−0.005 to −0.02), and at the low end it sits inside our ~0.04 run-to-run variance. The definition gate should set PASS bar at the upper end of the hypothesis (≤ −0.01) so a sub-noise "win" doesn't get called a win. Flag for the reviewer; not a taste issue.
- **Portfolio**: 010 (PolyLoss) and 011 (Cautious-Lion) in the same batch is fine — 013 is position-architecture, not loss or optimizer. No crowding.
- Crisp bet: "we expect −0.005 to −0.02 because content-conditional position encodes in-context structure (rare-word spacing, list boundaries) that fixed-decay position does not." Good.
