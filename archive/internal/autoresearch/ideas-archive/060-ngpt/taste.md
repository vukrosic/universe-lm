# Taste log — 060 nGPT

## r1 — 2026-06-11 — verdict: accept
- Leverage: geometry-level intervention (full unit-norm constraint on embeds + states + Q/K + MLP vectors), not a single-layer normalization tweak; paper's 4-20× step-efficiency claim should still leave a visible Δ in the ~92-step tiny1m3m budget if the mechanism fires at all.
- Information value: both outcomes log. WIN ⇒ residual is wasting capacity on unconstrained norms here; NULL ⇒ hypersphere geometry needs more scale/steps than the screen offers — feeds the 135M decision either way.
- Non-obviousness: distinct from closed/WIN levers — 016 qk-norm (WIN) normed Q/K only, 017 sub-ln-sandwich (null) re-stacked LN, 019 squash/DyT (closed) replaced norm op, 051-052 ScaleNorm/FixNorm (sibling queue) scale a single tensor. nGPT is the only one that constrains the *representation manifold* end-to-end.
- Portfolio fit: norm family is crowded right now (051/052/055/056/057/058/059 all at needs-taste alongside 060), but each is mechanistically distinct; nGPT is the most invasive and the most differentiated of the cluster — does not read as "the 5th momentum tweak".
- Niche fit: mechanism (not HP), runs at tiny1m3m. Caveat for the next gate: nGPT is **not identity-initable** — the hypersphere constraint is binary; the definition gate must accept that and design the A/B without an identity ctrl.
- Crisp bet: idea text already nails it in one sentence ("if it works, capacity is wasted on norms; if it fails, full geometry is too much for this codebase"). Pass-bar phrasing carries over cleanly.
- Transfer: paper evidence at 0.5B/1B is direct; mechanism is geometric, not regime-bound; transfer-risk: med is honest — implementation invasiveness is the real risk, not scale mismatch.
- Code-gate flag (for the next critic, not for me): touches embeddings, QK, MLP, residual update — likely the biggest LoC patch in the queue. Worth the slot if the spec stays minimal (skip learnable eigen rates, keep norm core only).
