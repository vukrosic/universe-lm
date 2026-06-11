---
id: 082-diff-transformer
status: needs-plan
round: 1
updated: 2026-06-11T01:21:40Z
transfer-risk: med
---

# 082 — Differential Transformer

## Source
Differential Transformer (arXiv:2410.05258). 2024.

## Mechanism
Replace one attention map with two softmax maps per head and subtract them so common-mode attention noise cancels. This keeps the macro block layout intact while giving each head a built-in noise-cancellation path.

## Scale evidence
The paper reports improved language modeling and downstream performance on large language models, including long-sequence and retrieval behavior; public model artifacts for the method also point to a 1.3B comparison with lower loss, with a 7B instability caveat. `transfer-risk: med` because the gain is real at billion scale, but the stability caveat means it may not port cleanly to a 0.94M run.

## Why it's worth a slot
It directly attacks attention noise, which is the kind of small structural improvement that could survive into the 135M recipe if it is not just a bigger-model trick.
