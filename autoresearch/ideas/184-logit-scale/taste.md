# Taste log — 184 logit-scale

## r1 — 2026-06-15 — verdict: accept

- **Leverage / transfer are real, not vibes.** Three independent families (CLIP, TRL DPO/IPO/SimPO, T5 `lm_head` bias) all converge on a single learned scalar on the output logits as a temperature / confidence knob. CLIP validates the lever form at 400M image-text pairs (≥100M effective); TRL validates it at 1B–70B SFT models in the RLHF regime. The mechanism is scale-free — a single scalar with a positive parameterization — so if it binds at 0.94M, the recipe carries to 135M essentially for free. `transfer-risk: low` is honest.

- **Distinct from the in-repo null pile (placement matters).** The 5+ logit-shape nulls in this repo — 152-attn-logit-bias, 155-per-head-temp, 160-post-av-gain, 166-t5-rpe, 167-logit-zloss, plus the closed `logit-softcap` axis — all act on the *attention* softmax shape (per-head QK bias, per-head temperature, post-AV gain, additive RPE, additive z-loss, per-position soft-clip). 184 is the *first* lever in the repo to act on the *output cross-entropy* temperature, not the attention softmax. A global scalar on the LM head output cannot be absorbed by per-token/head-local Q/K updates (which is what closed 152/155/160/166), and the closed-axis reasoning does NOT transfer: those nulls closed the *attention-side* softmax-shape axis, not the *output-side* temperature axis.

- **Step-0 byte-identity is exact, not approximate.** `logit_scale_param = 0` ⇒ `exp(0) = 1.0` in IEEE-754 exactly ⇒ `logits * 1.0 = logits` exactly ⇒ loss / gradient / predictions bit-identical to baseline at step 0. No fp32 epsilon, no `gain ≈ 1` approximation (cf. 183's RMSNorm). The cleanest possible A/B boundary; any observed Δ at step ≥ 1 is the lever's effect, not initialization noise.

- **Tied-embedding argument is real, not hand-wavy.** With tied embeddings (`W_emb` shared input/output), a global scalar on the output is NOT equivalent to rescaling `W_emb` (rescaling `W_emb` would also shift the input distribution). The lever therefore cannot be trivially absorbed by the tied-embedding weight matrix — the symmetry that makes per-head attention-shape levers null at 0.94M does not apply here. The residual stream's magnitude going INTO the head is constrained by the (existing or absent) final norm, not by the head weights themselves, so the optimizer has a genuine axis to push on.

- **Information value is high in all three outcomes.** WIN unlocks a 1-parameter temperature knob for the entire pipeline — cheap, no architectural change, no new norm layer, no new FFN, just `logits *= exp(s)` after the LM head. NULL inside the 0.04 noise band cleanly closes the *global output temperature* axis at 0.94M as a separate line from the attention-softmax-shape nulls. DRIFT (`logit_scale` pushed to a value that hurts val) is itself informative — it would say the implicit temperature from the tied embedding is sub-optimal in the *opposite* direction of what cross-entropy wants, which is a real finding. No "meh inconclusive" branch.

- **Niche fit is clean.** Mechanism (yes — the lever form is architectural, not an HP), identity/zero-init-able (yes, `logit_scale_param = 0` ⇒ `logit_scale = 1` exactly, byte-identical at step 0), tiny1m3m runnable, +1 param (+0.0001% of 0.94M, the cheapest possible lever in the queue). No new norm layer, no new FFN, no architecture change. Easy to run, easy to revert.

- **Crisp bet (one sentence, in the idea):** "at 0.94M with 92 update steps, the LM head's logit magnitudes are tied to the tied-embedding matrix and the residual stream's accumulated magnitude — both of which grow during training — so a single learned scalar can decouple the temperature from those dynamics and let the optimizer find the right sharpness for cross-entropy at this tier." Yes, sharp.

- **Pass/fail bar is honest.** WIN bar `≤ ctrl − 0.005` matches the standard pipeline rule and the 175-alibi-slopes win margin (Δ = −0.1585). NULL band `|Δ| < 0.01` is the standard pipeline convention; DRIFT `> ctrl + 0.01` is the standard drift threshold. Sub-noise is logged NULL per one-seed-only rule.

- **Portfolio note.** Queue is busy (5+ per-head/attention-shape variants in flight: 152/155/160/166/167/177/180), but 184 is at the *output* (LM head logits), not attention. Different placement, different mechanism, different gradient dynamic. Accept holds on freshness grounds; the definition gate should not stack another *attention-softmax-shape* variant on top of this — diversify, but 184 itself is a fresh axis.

- **Worth a slot.** Sharp bet, cheapest possible run (1 param, 1 line in `forward`, 1 config flag), clean null-vs-DRIFT signal distinct from the attention-side null pile, scale-free mechanism that ports to 135M.

Routing: `needs-review` for the definition gate, `round=1`.
