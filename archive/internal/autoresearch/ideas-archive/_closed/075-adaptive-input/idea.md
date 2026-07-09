---
id: 075-adaptive-input
status: rejected
round: 1
updated: 2026-06-10T16:51:58Z
transfer-risk: med
---

# 075 — Adaptive Input Embeddings

## Source
Adaptive Input Representations for Neural Language Modeling (arXiv:1809.10853). 2018.

## Mechanism
Split the vocabulary into frequency buckets, give frequent tokens more embedding capacity, and project each bucket back to `d_model`. Keep the current tied embedding as the base and add the adaptive branch with a zero gate so step 0 is unchanged.

## Scale evidence
The paper reports strong language-modeling results on WikiText-103 and Billion Word, with faster training and fewer parameters. transfer-risk: med - the evidence is directly LM-oriented, but the exact bucketed allocation still needs to prove itself in this tiny tied-head setup.

## Why it's worth a slot
This tests whether the model wants uneven embedding capacity by token frequency, which is a different question from changing the tokenizer or total embedding rank.
