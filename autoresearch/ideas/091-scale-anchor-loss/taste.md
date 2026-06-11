# Taste log — 091 Scale anchor loss

## r1 — 2026-06-11 — verdict: accept
- Tiny auxiliary-loss idea with a clean hypothesis about final residual scale.
- Distinct from the norm queue because it anchors magnitude without adding another normalization op.
- Worth testing because the implementation cost is tiny and the diagnostic value is high.

