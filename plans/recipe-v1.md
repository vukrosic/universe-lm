# Recipe v1 — current stack (auto-maintained by recipe-synthesizer)
updated: 2026-06-10T00:00:00Z

## In the stack (Δ cleared the two-ctrl bracket)

| lever | idea | Δ vs ctrl | bracket | transfer-risk | evidence |
|---|---|---|---|---|---|
| fire-pe | 009 | −0.064 / −0.082 | 0.0175 | low | autoresearch/ideas/009-fire-pe/evidence.md |
| qk-norm | 016 | −0.0138 / −0.0185 | 0.0047 | low | autoresearch/ideas/016-qk-norm/evidence.md |
| moonlight-muon-rms | 015 | −0.0138 / −0.0185 | 0.0047 | low | autoresearch/ideas/015-moonlight-muon-rms/evidence.md |
| cautious-lion | 011 | −0.0312 / −0.0321 | 0.0009 | low⚠️ | autoresearch/ideas/011-cautious-lion/evidence.md |

Notes:
- **009** is the largest single-lever win to date (−0.064 / −0.082, 3.7× the ctrl bracket). Robust.
- **015 and 016** tied at 6.3906 in the same session (ctrl 6.4044/6.4091). Independent mechanisms, same magnitude.
- **011** ⚠️: won by the two-ctrl rule within session (ctrl-gap 0.0009, Δ −0.031/−0.032 ≫ gap), but the session ctrl was drifted +0.19 from prior days (suspected wholesale file-sync baseline pollution — see 006 evidence). The treatment value (6.3941) sits at the prior-day ctrl level, meaning the effective improvement vs a clean session is ≈ 0. Carry it tentatively; treat as a candidate for re-run in a clean session before trusting the Δ.

## Excluded (and why)

- **cautious-muon (001)** — NULL inside bracket: Δ +0.006 (wrong sign); prior orphan-sweep −0.025 not reproducible with proper two-ctrl bracket.
- **cautious-adamw (002)** — NULL: emb-bucket Δ +0.0003, gain-bucket Δ −0.0066; both inside run-to-run variance (~0.04).
- **soap (003)** — NULL: vocab-size params (49 152) exceed MAX_PRECONDITIONER_DIM=2048, routing to AdamW fallback; SOAP preconditions only the tiny FFT/attn 2D matrices (minor fraction of total params at this tier) — mechanistic bypass, not a loss.
- **retnet-retention (004)** — NULL (probe): v1 ships kernel + stability probe; retention is not wired into MultiHeadAttention. The null characterises the probe, not the lever. v2 (full A/B) not yet filed.
- **decoupled-qkv-muon (005)** — NULL: Δ −0.003, sits between ctrls 6.3875/6.4050 (inside 0.0175 bracket).
- **schedule-free-adamw (006)** — NULL/negative: Δ +0.21 (wrong sign, large); ⚠️ session ctrl drifted +0.19 — A/B valid in-session but absolute numbers not trustworthy; re-test warranted on a clean session.
- **polyloss (010)** — NULL: Δ −0.0053 vs ctrl-to-ctrl gap 0.0059; margin inside session variance by the two-ctrl rule.
- **cope-stacked-fire (013)** — drift/destructive: treatment 6.4659 vs ctrls 6.3969/6.3891 (+0.069/+0.077 worse, ≫ gap 0.0078); +0.143 vs FIRE-alone (6.3234). CoPE + FIRE are destructive at tiny1m3m — do not compose.
- **sub-ln-sandwich (017)** — NULL: Δ +0.004/−0.001 inside gap 0.0047; lever per DeepNet only fires at 100+ layers; not useful at 6L.

## Conflicts — human call

- **015-moonlight-muon-rms × 011-cautious-lion**: both claim the 2D-param optimizer slot (Muon vs Lion); mutually exclusive optimizer families. Cannot stack. Pick one for the recipe; the other becomes the alternative route. (Moonlight won on a clean session; Cautious-Lion on a drifted one — Moonlight is the safer carry.)

## Untested interactions

- **009-fire-pe × 016-qk-norm** → composition idea filed as **026-fire-x-qknorm** — both in attention domain (positional bias on logits vs per-head Q,K normalization); interaction at logit level unknown.
- **015-moonlight-muon-rms × 016-qk-norm** → composition idea filed as **027-moonlight-x-qknorm** — different code paths (optimizer step vs forward pass); likely additive but untested.
- **009-fire-pe × 015-moonlight-muon-rms** — not yet filed (WIP gate budget used); different domains (attn vs optimizer); candidate for next pass.
