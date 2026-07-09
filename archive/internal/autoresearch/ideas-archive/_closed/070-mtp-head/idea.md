---
id: 070-mtp-head
status: rejected
round: 1
updated: 2026-06-10T23:24:47Z
transfer-risk: low
---

# 070 — Multi-token prediction head

## Source
Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737). Apr 2024.

## Mechanism
Attach one or more auxiliary prediction heads on the shared trunk to predict the next few tokens in parallel, and add those losses to the main next-token CE. The implementation is small: a few extra heads, shifted targets, and a weighted sum of auxiliary losses.

## Scale evidence
The paper reports better downstream performance for 13B code and natural-language models, plus faster inference for 4-token prediction. `transfer-risk: low` because the method was validated on large LMs and is a direct auxiliary-head change.

## Why it's worth a slot
Multi-token heads test whether tiny1m3m benefits from richer supervision than pure teacher-forced next-token prediction.

