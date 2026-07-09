# Taste log — 191 per-token-attn-gain

## r1 — 2026-06-15 — verdict: accept
- **Leverage / axis:** Per-token scalar gain on attention output before W_O is *granularity-distinct* from the three closed gain axes (142 per-channel residual, 160 per-head post-AV, 176 V-side pre-attention-product). Per-token = T scalars/block; per-head = H=4 scalars/block; per-channel = d_model=64 scalars/block. The 512× DOF increase over 160 is a real axis difference, not an HP tweak.
- **Bit-identity at step 0:** γ_t init 0 ⇒ `1+0 = 1` exactly. Clean.
- **Crisp bet:** "Per-token granularity finds a non-redundant axis with W_O downstream, where per-head granularity (160) did not." This is a falsifiable, mechanistic claim — 160 NULL'd because post-AV magnitude variation is plausibly absorbed by W_O; 191's bet is that T-token granularity escapes that absorption. Testable on one seed.
- **Information value both ways:** NULL confirms W_O-absorption extends to per-token granularity (closes the post-AV axis family). WIN gives a new lever for the 135M recipe (per-token conditioning primitive is well-validated at scale — NormFormer/Primer, CaiT CLS-token gain).
- **Portfolio concern noted (not blocking):** 4th gain-style lever in the family (142, 160, 176 closed null; 191 pending). The mechanism is distinct *enough* that this is not a duplicate — but if 191 NULLs, the entire post-AV gain family should be considered closed at 0.94M and the queue should pivot to a different family.
- **Niche fit:** mechanism-shaped (not HP), identity-able, fits tiny1m3m, transfer-risk: low.
- **Verdict: accept** → needs-review (round reset to 1 for definition gate).