# Taste log — 043 MLA

## r1 — 2026-06-11 — verdict: reject
- Already a closed lever: `autoresearch/closed.md` "Closed axes (seed)" lists `MHA vs GQA, MLA, Tied QK` as falsified architecture knobs — re-proposing MLA is a dup the miner should have caught.
- Already empirically run at this tier — `tiny1m arch` row 3 (`LEADERBOARD.md:48`): `vq-gain+rope250k+swa384+mla` (latent_dim=16) → 6.3253, behind tied QK (6.3041), MHA (6.3069), and LayerNorm (6.3109). Information value = 0; a second run would tell us nothing new.
- Already retested on the strong baseline — `LEADERBOARD.md:147,164` (`screen20m` #73): MLA on `vq-gain+swa+highrope` → 4.7269 vs 4.6364 = **+0.091 worse**, explicitly tagged "**MLA is CLOSED on the new best baseline**". Two independent nulls cover both tiers.
- Mechanism doesn't fire at our scale: MLA's leverage in DeepSeek-V2 is KV-cache compression for 128K-context · 236B-param inference, not pretraining loss. At tiny1m3m (6L · 4 heads · 16-dim head · 512 ctx, 0.94M params, 3M tokens) KV cache pressure is ~zero, so the bottleneck is pure capacity drain — exactly what the two prior runs measured. The `transfer-risk: low` tag conflates "paid off at scale" with "the loss-side mechanism transfers down"; it doesn't.
- "A null would say the latent bottleneck is not buying anything" — but we already have that null, twice. No bet left to make.
