---
id: 100-mud
status: needs-plan
round: 1
updated: 2026-06-11T01:18:41Z
transfer-risk: low
---

# 100 — MUD (momentum decorrelation)

## Source
Southworth and Thomas, "Beyond Muon: MUD (MomentUm Decorrelation) for Faster Transformer Training" (arXiv:2603.17970).

## Mechanism
Replace Muon's polar-decomposition orthogonalizer with a cheaper triangular whitening surrogate inspired by Gram-Schmidt and Gauss-Seidel.

## Scale evidence
The paper reports 10-50% wall-clock improvement over tuned AdamW and Muon, with roughly 1.3x-2.6x peak tokens/s in common settings and even larger gains on GPT-2 large. That makes it one of the strongest "same direction, cheaper path" optimizer papers in the batch.

## Why it's worth a slot
This is a direct follow-up to the Muon line, but with lower optimizer overhead. If the repo wants orthogonalized updates without paying polar-decomposition tax, this is the right next bet.
