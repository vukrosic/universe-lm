---
id: 031-adam-mini
status: tasting
round: 2
updated: 2026-06-10T23:48:30Z
transfer-risk: low
---

# 031 - Adam-mini

## Source
Adam-mini: Use Fewer Learning Rates To Gain More (arXiv:2406.16793, Zhang et al., 2024).

## Mechanism
Use the paper's verbatim partition — not a custom one. AdamW keeps per-element `v` for Embed, Output (LM head), MLP (up/down), and all LayerNorm/RMSNorm tensors. The change is confined to the attention block: collapse `v` to **one scalar per head** for `W_q`, `W_k`, `W_v`, `W_o` (paper §3, "QK by head, V/O by head"). Concretely for tiny1m3m (`n_layers=12`, `n_heads=4`, `n_kv_heads=2`, `d_model=64`): Q gets `4×12=48` shared scalars, K gets `2×12=24`, V gets `2×12=24`, O gets `4×12=48` — 144 attention v-scalars total in place of ~49k per-element entries. Update rule is unchanged elsewhere; β₁, β₂, ε, LR all preserved → step-0 first-moment trajectory matches AdamW, identity-at-init on the loss curve.

## Scale evidence
Paper reports equal-or-better val loss vs AdamW from 39M (their smallest LLM ablation) through 13B across pretraining, SFT, and RLHF. transfer-risk: low — the loss-equivalence claim, not the throughput claim, is what carries: paper validates ≥39M which is 40× our 0.94M but the mechanism (per-head Hessian similarity) is width-independent at any depth, and Embed/MLP keep per-element `v` so the "different gradient scale across module types" failure mode never fires. (Throughput / memory savings are stripped from this pitch — at 0.94M optimizer state is ~7.5MB; per-step wall time is not what we A/B.)

## Why it's worth a slot
We expect Δval ∈ [−0.005, +0.005] at tiny1m3m (null band) — the paper's headline is equivalence, not improvement, so a **clean null is the modal outcome and is itself the lever-killing result for the 135M plan**. A clear WIN (Δval ≤ −0.005) confirms per-head Hessian similarity holds even at 0.94M / 4 heads — strong signal to port Adam-mini up the ladder. A clear LOSS (Δval ≥ +0.005) falsifies the head-sharing thesis at small scale and kills the Adam-mini family for the 135M recipe, freeing the slot from re-testing it on screen20m. Either reading retires the senior bet in the structured-2nd-moment cluster (036-lamb, 040-adafactor already taste-rejected with 031 named as the better bet); on one seed at tiny1m3m this is a single A/B that resolves a whole sub-family.
