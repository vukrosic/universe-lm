---
id: 092-seednorm
status: needs-plan
round: 1
updated: 2026-06-11T01:17:53Z
transfer-risk: med
---

# 092 — SeeDNorm (self-rescaled dynamic normalization)

## Source
Cai, Zhu, Liu, and Min, "SeeDNorm: Self-Rescaled Dynamic Normalization" (arXiv:2510.22777).

## Mechanism
Use RMSNorm as the base, but let the scale coefficient depend on the current input norm instead of being a static learned vector or scalar. The layer still normalizes, but it preserves and re-injects norm information in a data-dependent way.

## Scale evidence
The paper validates SeeDNorm across large-model pretraining and vision settings, with negligible efficiency impact and consistent gains over RMSNorm and LayerNorm. That makes it a strong candidate when the model needs more flexibility than a fixed gain but less disruption than removing normalization entirely.

## Why it's worth a slot
This sits between fixed-gain norms and norm-free activations. If tiny models care about input-dependent radial information, SeeDNorm should reveal it without forcing the whole stack into a new regime.
