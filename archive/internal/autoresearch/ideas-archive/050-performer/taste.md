# Taste — 050-performer

## r1 — 2026-06-11 — verdict: reject

- **Linear-attention family is already falsified at tiny1m3m.** Three prior slots have decided the same bet: `004-retnet-retention` ran NULL (trt 6.4162 vs ctrls 6.3875/6.4050, loses to both), `008-gated-deltanet` taste-rejected as off-niche on tier, and `012-gated-deltanet` taste-rejected with the miner conceding "mechanism doesn't fire at this scale". Performer (2020) is the *parent* of that family (FAVOR+ is the random-feature ancestor of retention/delta-rule kernels); running it adds zero new information beyond what 004 already told us.
- **No leverage at this tier.** Performer's reason to exist is sub-quadratic cost at long sequences (proteins, pixels). At tiny1m3m — 0.94M params, ~512-token SWA window, 6 layers — exact softmax is already cheap, so the only path to a win is for kernelized features to be a *better inductive bias* than softmax. That is exactly the bet 004-retnet lost. A clean null here repeats public knowledge.
- **Crowded family in the active needs-taste queue.** 049-cosformer, 050-performer, 053-reluformer are three consecutive linear-attention mines. Even if each were individually defensible, taste says diversify.
- **Transfer to 135M is worse, not better.** Tagged `transfer-risk: med` but the *direction* of risk is wrong for this niche: the paper's headline wins are at long-context/cross-domain, not at the SWA-bounded short-seq regime the recipe screen targets. There is no mechanism-level argument in the idea for why FAVOR+ would fire below the seq-len/model-width threshold where it's normally measured.
- **Zero-init mix trick does not save the bet.** Starting the feature-map path at zero just guarantees the model can ignore Performer — it doesn't make kernelized attention a stronger inductive bias when the gate opens. Same critique already applied to other "mix in from zero" linear-attention pitches.
- Close: reject; move folder to `_closed/`; append to `closed.md`.
