---
id: 083-root-optimizer
status: needs-plan
round: 1
updated: 2026-06-11T01:17:26Z
transfer-risk: low
---

# 083 — ROOT Optimizer

## Source
ROOT: Robust Orthogonalized Optimizer for Neural Network Training (arXiv:2511.20626). 2025.

## Mechanism
Start from the Muon-style orthogonalized update path, then add dimension-adaptive Newton-Schulz coefficients plus a proximal soft-threshold step to suppress gradient outliers. It can likely be wired as an optimizer wrapper around the existing orthogonalized update path, so the model code stays untouched.

## Scale evidence
The paper trains a 1B Transformer on a 10B-token trajectory and compares against a Muon baseline; both ROOT variants achieve lower training loss, and the main experiment uses a 100B-token sample. `transfer-risk: low` because the gain is already shown at 1B, though the extra orthogonalization machinery is more complex than plain Muon and could still be fragile at tiny1m3m.

## Why it's worth a slot
If ROOT beats Muon at our tiny screen, it is a direct upgrade to the optimizer family we already care about, not just another schedule tweak.
