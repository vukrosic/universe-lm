---
status: done
---
# 310 — embedding_scale ×2.0 (8.0->16.0)

NEW axis: embedding scale (residual-stream init magnitude). Upper arm of the
bracket vs default 8.0=sqrt(d_model). Stacks on combo+×2.0LR champion. Seed 42.
A/B vs champion ~6.175. Held as DRAFT until 306 confirms (then flip to needs-run).
