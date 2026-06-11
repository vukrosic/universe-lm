# Taste log — 038 SWAN

## r1 — 2026-06-11 — verdict: accept
- Leverage: stateless gradient normalization+whitening as the matrix-param update; if it works, signals "geometry > memory" for the 135M recipe — a recipe-shaping result, not a rounding-error lever.
- Information value: clean either way at tiny1m3m — win = stateless beats Muon's Newton-Schulz; null = Muon's specific orthogonalization route matters; either outcome is loggable.
- Novelty / portfolio fit: among the 031–040 optimizer queue this is the most mechanistically distinct — most others (031 Adam-Mini, 032 ADAMS, 033 Sophia, 034 Adan, 036 LAMB, 039 Apollo, 040 Adafactor) are adaptive/curvature variants of Adam; SWAN is the only stateless one. Closed optimizer wins so far (011 cautious-Lion, 015 Moonlight-Muon-RMS) are stateful — accepting one stateless bet diversifies the family rather than crowding it.
- Niche fit: pure optimizer swap, identity-able (set whitening to identity → ordinary SGD), no architecture change, transfer-risk:low matches source-scale evidence (LLaMA 350M / 1.3B).
- Crisp bet: stateless preprocessing (normalize + whiten matrix grads) on the Muon-routed params beats Muon's Newton-Schulz orthogonalization at fixed step count.
- **Carry-forward note for definition gate (not blocking, but must be tightened there):** the baseline already routes matrix grads to Muon, not AdamW — so the meaningful A/B is **SWAN-vs-Muon on matrix params** (not SWAN-vs-AdamW as a naive read of the paper might suggest). The reviewer should pin the control to current Muon+AdamW-on-scalars baseline so the delta isolates "whitening replaces Newton-Schulz", not "whitening replaces Adam-style adaptive memory".
