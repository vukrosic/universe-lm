---
id: 074-transformer-xl
status: needs-plan
round: 1
updated: 2026-06-11T01:21:40Z
transfer-risk: med
---

# 074 — Transformer-XL Memory

## Source
Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (arXiv:1901.02860). Dai et al., 2019.

## Mechanism
Split each 2048-token training sequence into **4 sub-segments of 512 tokens**. Run sub-segments in order within a single forward pass. After each sub-segment's attention, **detach and cache** the K/V projections of the last sub-segment (a 512-token ring buffer, refreshed per sub-segment). On the next sub-segment, **prepend the cached K/V** to the current K/V before the SDPA call, so attention spans 512 cached + 512 current = 1024 tokens. Positions are RoPE-extended (cache positions are 0..511, current are 512..1023) — no Shaw-style relative bias, no new positional machinery, no new learnable parameters. Step-0 identity holds: the *first* sub-segment of the *first* example sees an empty cache and is bit-exact to baseline; only sub-segments 2/3/4 deviate, so the lift is measured against the unmodified baseline on those positions.

## Scale evidence
Original paper improves WikiText-103, enwik8, PTB, and One Billion Word LM, plus longer effective dependency lengths (~3-4× the segment length). Largest published gain: 1B-word (~1B tokens, RNN-style LM head). At our tier (3M tokens, seq_len=2048, 733 steps) the mechanism has **not** been validated. transfer-risk: **med** — the published scale is much larger, the gain is real there, but at 3M tokens we have no published prior. The cache itself is 512 K/V vectors × n_layers × d_head × 2 (K+V) — adds ~`n_layers × 512 × 2 × d_head` floats to peak memory per layer, negligible vs. activations.

## Why it's worth a slot
**Crisp bet:** with SWA(512) as the only horizon and 4 sub-segments per training example, attaching a 512-token detached cache extends the *effective* context from 512 → 1024 at the segment boundary. We expect val-loss Δ ≈ **-0.01 to -0.04** because roughly 1-in-4 sub-segment boundaries in our wikitext-style data carry a real cross-boundary reference (a 2048-token sample with no boundary-crossing reference is the exception, not the rule, and a 512-token cache covers that gap with detached, gradient-free, O(1) extra params). Null result is informative: it would mean *cheap segment-recurrence doesn't pay at 3M tokens / seq_len=2048* and closes the entire segment-memory branch of the recipe tree.

**Why not a clone of 008/012/004:** 008-gated-deltanet and 012 are linear-attention *replacements* for softmax (substitute the attention kernel) — they fail at tiny1m3m because linear kernels don't carry enough signal at this scale. 004-retnet is the same family. **XL is structurally different**: it is an *additive cache* on top of standard softmax attention. The softmax, the RoPE, the FIRE-PE, and the SWA all stay in place — XL only extends the K/V sequence. So XL composes with the wins of 009 (FIRE, Δ -0.064), 021 (V-residual, Δ -0.034), and 023 (Canon-Conv, Δ -0.084), where 008/012/004 *replace* the kernel those wins sit on. The comparison isn't "does softmax-memory work at 3M tokens" (008/012/004 already answered that for linear softmax-replacements) — it's "does the segment-recurrence *pattern* (independent of kernel) fire at this tier."

**Tier-fit regime (named):** chunk_size=512, seq_len=2048, 4 segments per training example. At our data distribution, a wikitext sample of length L has ⌈L/2048⌉ forward passes and 4·⌈L/2048⌉ sub-segment boundaries. The cache lights up on sub-segments 2, 3, 4 of every pass. That's ~75% of compute seeing a non-empty cache — the regime is large enough to register a val-loss signal, but small enough that a null is also clean (it's not a sample-size issue).

**Single point of decision (one A/B, no knobs):** treat baseline as the unmodified transformer; treatment adds the K/V cache pass. Same RoPE base, same FIRE, same SWA window, same optimizer, same seed 42, same data. No new hyperparameters to sweep — chunk_size=512 is fixed by the seq_len / segment-count math.
