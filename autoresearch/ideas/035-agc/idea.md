---
id: 035-agc
status: rejected
round: 2
updated: 2026-06-10T23:50:40Z
transfer-risk: high
---

# 035 - Adaptive Gradient Clipping

## Source
High-Performance Large-Scale Image Recognition Without Normalization (Brock et al., arXiv:2102.06171, 2021). NFNet-F1/F2/F3/F4/F5 reach 83.2-86.5% top-1 on ImageNet without BatchNorm by replacing global L2 grad clip with per-parameter-unit AGC at threshold λ=0.01.

## Mechanism
Replace `torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)` at training/trainer.py:574 (and :873) with a per-parameter AGC step. For each parameter `w` with gradient `g` and `dim() >= 2`, compute `||g|| / ||w||`; if it exceeds λ, scale `g ← g · (λ · ||w|| / ||g||)`. Skip 1D parameters (biases, norm gains) — AGC's original prescription. λ is the only knob.

## A/B design (addresses taste r1 finding #2 — global clip collision)
- **Control**: current `grad_clip=1.0` global L2 (the running baseline).
- **Treatment A (clean replacement)**: AGC at λ=0.01, *replaces* global clip. 1D params still go through `clip_grad_norm_` on the 1D subset only.
- **Treatment B (threshold sweep)**: same as A, λ ∈ {0.001, 0.01, 0.05} — one config, three seeds-of-λ, report best. The sweep costs ~3× wall-clock but bounds the threshold-uncertainty.
- **No stacking** of AGC on top of global clip — the taste review correctly notes that at global 1.0, AGC almost never fires (info-free A/B). The clean replacement is the only way to measure the mechanism.

## Scale evidence (re-classified)
Brock et al. 2021 is the sole AGC primary source; NFNet evidence is vision-side, ≤550M params, and the *enabling conditions* (no BatchNorm, large batch 2048+, heavy AugMix) are all absent at tiny1m3m (LayerNorm-only, micro-batch ~16, no augmentation). LM pretraining literature (Hoffmann/Brown/Touvron/Su, modded-nanogpt) does **not** adopt AGC — global L2 clip at 1.0 is the universal default. **transfer-risk: high** (was: med). The mechanism plausibly exploits a vision-specific coupling between no-BN and large-step updates that the LM literature has not replicated.

## Failure-mode trace (addresses taste r1 finding #3 — "bad steps dominate run")
The two real instability events at tiny1m3m were 020-forgetting-attn (FoX, NaN at step ~400/732) and 022-softpick-attention (NaN, both runs). Inspection of `2026-06-10 arq` outputs shows both were **row-renorm blow-ups inside the attention block** — the NaN entered the forward pass via `q / q.norm(dim=-1, keepdim=True)` when `q.norm` collapsed, *before* `loss.backward()`. AGC operates on the gradient between `loss.backward()` and `optimizer.step()` — by then the parameter itself has already been corrupted by the divergent forward. AGC would not have caught either failure. The "bad steps dominate the run" framing is therefore load-bearing-but-wrong for our actual failure modes; AGC is the wrong lever for the instabilities we have on record.

## Why it's worth a slot
We expect **Δ ∈ [−0.01, +0.01] vs ctrl** — an informative null — *because* AGC's headroom-unlock mechanism depends on the no-BN/large-batch/strong-aug regime that tiny1m3m does not run, and global L2 clip at 1.0 is the de-facto LM standard that AGC was not designed to displace. The bet's value is *closing* the lever at this tier: if it nulls (the expected outcome), 035 enters the closed list alongside 001-006/010 and frees an optimizer-wave slot; if it surprisingly wins by ≥ −0.01, AGC is a 2-line swap at trainer.py:574 worth a phase-2 follow-up at 10M+. The run is informative either way — it converts an open "maybe" into a closed "no" with a measured A/B. Compared to the 11-idea optimizer wave (7 done, 5 null, 2 win), this is the right shape of question to ask *next*: a tier-bounded null that confirms or kills the lever.

## Plan-bar
- Bar: Δ ≤ −0.005 vs `grad_clip=1.0` ctrl at the best-of-3 λ, *at tiny1m3m/seed-42*. Below bar = informative null → close at this tier.
- Cost: ~5 min per AGC run on V100, ~3 thresholds + 1 ctrl = 4 jobs, ~20 min total.
