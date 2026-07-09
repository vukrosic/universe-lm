---
id: 077-fnet
status: rejected
round: 1
updated: 2026-06-10T16:50:03Z
transfer-risk: med-high
---

# 077 — FNet Fourier Mixer

## Source
FNet: Mixing Tokens with Fourier Transforms (arXiv:2105.03824). 2021.

## Mechanism
Apply a Fourier transform along the token axis to create a global token mixer, then blend that mixer back into the residual stream with a zero-initialized gate so step 0 matches the baseline.

## Scale evidence
The paper reports 92-97% of BERT's GLUE accuracy, strong Long Range Arena results, and large speedups at standard sequence lengths. transfer-risk: med-high - the encoder evidence is strong, but causal LM transfer is the open question.

## Why it's worth a slot
This is a cheap global-mixing test: if FFT-style token mixing helps here, the model is bottlenecked by attention's quadratic structure more than by content matching.
