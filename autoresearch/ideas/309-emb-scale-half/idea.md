---
status: done
---
# 309 — embedding_scale ×0.5 (8.0->4.0)

NEW axis: embedding scale (residual-stream init magnitude). Lower arm of the
bracket vs default 8.0=sqrt(d_model). Stacks on combo+×2.0LR champion. Seed 42.
A/B vs champion ~6.175. Held as DRAFT until 306 confirms (then flip to needs-run).
