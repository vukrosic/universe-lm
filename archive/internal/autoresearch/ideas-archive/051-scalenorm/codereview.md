# Code review log — 051 ScaleNorm

## r1 — 2026-06-11 — verdict: accept
- The scalar-gain norm is wired through `make_norm("scalenorm")`, the tiny1m3m config routes both residual norms to it, and the gain is truly scalar.
- Focused tests pass under `python -m pytest -q tests/test_scalenorm.py`, including the identity-at-init check and the config wiring check.
