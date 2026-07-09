---
id: 007-sigmoid-loss
status: rejected
round: 3
updated: 2026-06-09T03:34:16Z
---

# 007 — Sigmoid loss for LM pre-training

## Source
Tian et al., Apple, 2023, "Sigmoid Loss for Language Model Pre-training", arXiv:2309.06965. (**Reviser note r1 + r2:** original miner-cited `arXiv:2309.06979` is the "Scaling Law for Strongly Correlated Token Frequencies" paper (different). Re-mine in r1 failed (web search API error). The r2 reviewer proposed `arXiv:2309.06965` and the author/title match; this reviser could not independently verify the ID this session (web search failed twice with API 400), so the ID is applied on the r2 reviewer's authority. **r3 reviewer: please confirm 2309.06965 against the arXiv listing before approving**; if it's wrong, reject at r3 per the r2 hand-off.)

## Mechanism
Replace the softmax cross-entropy LM head loss with a per-vocab-position sigmoid (binary cross-entropy summed over vocabulary), plus a z-loss regularizer on the logit magnitude: `L = sum_v BCE(logit_v, target_v) + z * logsumexp(logits)^2`. The gradient is bounded (sigmoid saturates at ±1, not at 0/1) and there's no implicit competition across vocab positions. Implementation: ~15 LoC swap in the loss head, no model-shape change.

## Tier, seed
**tiny1m3m, seed 42 only.** Per the new pipeline rules (🔴 one tier / one seed), this idea runs at tiny1m3m (0.94M params · 3M tokens, seed 42). No screen20m, no multi-seed, no seed sweeps. A sub-noise effect is **inconclusive, not real** — log it and move on.

## Pass / fail bar (tiny1m3m, V+q+SWA+HighRoPE control 6.4287)
- pass: tiny1m3m val ≤ 6.4237 (Δ ≤ −0.005)
- fail: tiny1m3m val > 6.4287
- noise: |Δ| ≤ 0.005 (single-seed, tiny1m3m) — treat as inconclusive
- expected Δ ≈ −0.005 to −0.02; lower values are below the noise floor
- control source: `autoresearch/queue.md` Remote run log row 1 (tiny1m3m ctrl, 1B data, T4, 6.4287)

## Z-loss coefficient (committed)
**`z_loss_lambda = 1e-4`**, matching the Apple paper's default and the existing repo precedent (OH1 in `training/trainer.py:316-320` already implements `z_loss = z_loss_lambda * (logits.logsumexp(dim=-1) ** 2).mean()` — this idea just lowers the flag from `0.0` to `1e-4` and swaps the CE for BCE in the loss head). The `use_z_loss` gate is already wired; only the lambda value changes.

## Why it's worth a slot
Standard softmax CE has a known pathology: the model must allocate mass to distractor tokens, and the gradient on the gold token is `1 - p(gold)` which is fine, but on negatives it never quite hits zero, dragging magnitude growth. Sigmoid loss decouples the targets so each vocab position has its own bounded gradient, and the z-loss term penalizes runaway logit scale. We expect a small val-loss improvement (~0.005–0.02 at tiny1m3m) at no compute cost and no architecture change — the bet is the loss head, not the model. A null still teaches us that softmax-CE-on-this-data is already in its basin, which is useful prior for the next loss-shape ablation (PolyLoss, ET loss, etc.).

## Transfer argument
The mechanism is **loss-head-local**: sigmoid BCE + z-loss don't touch the model architecture, don't introduce scale-dependent hyperparameters (z is a scalar), and the gradient is bounded at all vocab positions. The single tiny1m3m A/B is the *only* test this pipeline runs — there's no larger tier to validate against. A win at tiny1m3m is a real signal that the loss-head swap helps on this data; a null is a close on the lever at this magnitude. (Note: this is the only evidence the pipeline produces. The "transfer to 25M/135M" story is hypothetical, not testable here.)

## Wiring
Add `use_sigmoid_loss: bool = False` to `LLMConfig` (next to `use_z_loss` / `z_loss_lambda` in the OH1 block). The `z_loss_lambda` field is already defined; this idea sets it to `1e-4` when `use_sigmoid_loss=True` (and leaves the `use_z_loss`/softmax-CE path at `0.0`). In the training loop at `training/trainer.py:309-314` and `:340-344`, gate the loss swap: `F.binary_cross_entropy_with_logits(logits, target_oh) + z_loss if config.use_sigmoid_loss else F.cross_entropy(...)`. ~15 LoC (target = one-hot of `shift_labels` with `ignore_index=-100` → 0; ~5 LoC loss swap; ~5 LoC gate), bit-identical to baseline when `use_sigmoid_loss=False`.
