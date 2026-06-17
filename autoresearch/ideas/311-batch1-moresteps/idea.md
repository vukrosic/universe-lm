---
status: done
---
# 311 — batch_size 2->1 (2× optimizer steps)

NEW axis: batch-size / #optimizer-steps. Model is update-starved (LR×2 helped). bs=1 doubles steps at fixed 3M tokens. Combo champion base LR. Seed 42. A/B vs champ 6.1998.
