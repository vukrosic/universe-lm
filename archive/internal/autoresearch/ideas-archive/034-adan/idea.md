---
id: 034-adan
status: needs-plan
round: 1
updated: 2026-06-11T01:34:08Z
transfer-risk: low
---

# 034 - Adan

## Source
Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models (arXiv:2208.06677, 2022). Paper claims ~2× wall-clock speedup over AdamW for ViT/ConvNeXt/MAE and reports 1k→32k minibatch tolerance on language models (GPT-2, Transformer-XL, BERT).

## Mechanism
Replaces Muon on the 2D matrix-weight path. Uses an extra gradient-difference term `g_t - g_{t-1}` (Nesterov-style "look-ahead" on the gradient) with its own first/second moments, giving the update momentum that already accounts for *change* in the gradient — no extrapolated-gradient evaluation needed. In repo terms: a new `Adan` optimizer class wired into the 2D slot (`use_adan=True` flag in `LLMConfig`, mutually exclusive with `use_lion` and the default Muon path), AdamW path on 1D/embed/norm/head untouched.

## Slot
**Adan replaces Muon on 2D non-embedding, non-norm weights. AdamW path unchanged on 1D / embed / norm / lm_head.** This is the only slot where Adan is interpretable here: (a) AdamW-path-only at 0.94M means the lever mostly bypasses the small-vocab embed head, replicating 003-SOAP's failure mode; (b) Adan on 2D *and* Muon retired is a full optimizer swap — the most decisive test, and what the rest of the pitch sizes up to. Muon owning the 2D slot post-015-Moonlight-Muon-RMS is a WIN'd surface, so dethroning it requires a real bet, not a vibe.

## Hyperparameters (sized to tiny1m3m's 92-step run)
Adan defaults (β1=0.9, β2=0.92, β3=0.99) were tuned for ImageNet-scale runs (~50k+ steps). For our 92-step run:
- **β1 = 0.9** (gradient first moment, ~7-step half-life — fine, we see a step worth of momentum immediately).
- **β2 = 0.92** (gradient-difference momentum, ~12-step half-life — fine, the lever fires by step ~25).
- **β3 = 0.95** (second-moment EMA, ~14-step half-life — shortened from 0.99's ~70 steps so the moments are warm by mid-run, not at the very last step). 018-AdEMAMix was rejected on this exact axis (slow EMA half-life vs ~92-step run, 99% init-weighted); Adan ships the explicit fix.

## Scale evidence
Paper shows gains on GPT-2 1.5B-class pretraining (Table 6 in the arXiv report) and ViT/MAE at 100M+ scales, with minibatch tolerance 1k→32k. transfer-risk: **low** — published at ≥100M on language models, and the lever (momentum rule) is scale-invariant by construction. Tiny1m3m is the cheapest screen; the question is whether the mechanism is *informative* there, not whether it survives.

## Why it's worth a slot
**Bet:** val loss **−0.01 to −0.03** vs the Muon-only ctrl (`Tiny1M3MConfig`), bracket-band from Cautious-Lion 011. The most distinct mechanism in the 031–040 optimizer queue (vs Adam-mini's per-block grouping, Sophia's Hessian probe, AGC's grad-norm clip): Nesterov-style gradient differencing gives the update a one-step *look-ahead* on momentum direction without the dual-evaluation cost of classical Nesterov. With β3=0.95 the moments are demonstrably warm by step ~30, so a NULL is not a warm-up artifact. **What a NULL teaches:** the gradient-difference term adds no information at 92-step horizon — independent of EMA warm-up, independent of Muon's orthogonalization advantage, and orthogonal to what 001-cautious-muon / 002-cautious-adamw / 006-schedule-free-adamw NULL'd. That's three optimizer-momentum NULLs we'd be combining into one structural conclusion: Nesterov-style differencing is a wash on the matrix-weight path, and the action moves to non-optimizer levers.
