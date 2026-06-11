---
id: 068-unlikelihood
status: rejected
round: 1
updated: 2026-06-10T23:27:37Z
transfer-risk: low
---

# 068 — Unlikelihood training

## Source
Welleck et al., "Neural Text Generation with Unlikelihood Training" (arXiv:1908.04319). Aug 2019.

## Mechanism
Add a negative loss on tokens or n-grams the model should avoid, so repeated or dull continuations are explicitly pushed down instead of merely not being rewarded. In this repo that can be implemented with a tiny set of sampled negatives per sequence, paired with the normal next-token CE.

## Scale evidence
The paper reports less repetitive, less dull text while maintaining perplexity, and better greedy/beam generations on dialogue and text-generation tasks. `transfer-risk: low` because it directly targets a common LM failure mode and was shown on real language tasks.

## Why it's worth a slot
Tiny models often fall into repetition before they lose perplexity; unlikelihood attacks that failure mode directly instead of hoping CE will learn around it.

