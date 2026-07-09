# Taste — 045-transformerfam

## r1 — 2026-06-11 — verdict: revise (finalized)

- Confirmed the four findings from the in-flight review above: (a)
  `transfer-risk: low` is mis-tagged — the FAM lever is a *long-context*
  working-memory loop, and seq=2048 fits in a single attention window, so
  the recurrent feedback has nothing extra to attend to at our scale;
  (b) the bet must be sharpened to a single-sentence transfer hypothesis
  (FAM ≥ -0.02 vs FIRE-ctrl because explicit working memory > static
  V-shortcut, or null closes the feedback-loop subfamily); (c) portfolio
  is crowded — 021 V-residual, 023 Canon-conv, 024 Gated-attn, 044 AttnRes
  in the same two weeks — and the marginal-contribution question is what
  FAM adds over a static cross-layer shortcut (V-residual) and
  content-routed cross-layer attention (AttnRes); (d) zero-init gate is a
  strong niche fit (step 0 = unmodified baseline), so the A/B is cheap
  even if the expected outcome is a null.
- **No re-litigation** — these are settled. The miner re-pitches against
  these four points; the next round reads as `revise` only if any of the
  four is missed. If the re-pitch lands them all, this flips to `accept`
  and the definition loop runs its own 3-round budget.
- **Why not accept on the strength of the niche fit alone?** The lever
  doesn't fire at seq=2048 by the paper's own framing, so the A/B is
  informative only as a "close the family at our scale" log, not as a
  real bet on val loss. The re-pitch must own that framing — a tastey
  re-pitch is "clean null = closes feedback-loop family here" — so the
  miner isn't promising a win the data structure can't deliver.

- **transfer-risk is mis-tagged.** The miner writes `low` citing 1B/8B/24B
  evidence. The pipeline's `transfer-risk` is *our* tier (tiny1m3m, 3M tokens,
  seq=2048), not the paper's. FAM's published wins are on **long-context**
  tasks (the working-memory framing only pays off when the model's K/V
  receptive field is exhausted). Our seq=2048 fits comfortably in a single
  attention layer's window, so the recurrent feedback loop has nothing extra
  to read that attention alone doesn't already attend to. That's
  transfer-risk=`high` for our regime, regardless of how cleanly it scales
  upward. The tag is the first thing the definition gate reads.
- **the bet is vague.** "Does the tiny model need a recurrent memory trace"
  is a vibe, not a sharp prediction. The paper's actual claim is "long-context
  QA/retrieval" — not val loss on a 3M-token pre-training run. Frame the bet
  as a *transfer hypothesis*: FAM should match or beat 021-V-residual
  (Δ≥-0.02 vs FIRE-ctrl) because a feedback loop gives the model an explicit
  working memory rather than a static V shortcut, *or* the null is expected
  and the result logs "feedback attention family is dead at our scale
  regardless of recurrence" (which is also informative — closes the family).
- **portfolio crowding is real.** Recent queue is dominated by cross-layer /
  recurrent attention variants: 021 V-residual (done, WIN w/ caveat), 023
  Canon-conv (done, WIN w/ caveat — best of the cluster at -0.06 isolated),
  024 Gated-attention (done, WIN w/ caveat), 044 AttnRes (tasting, also
  cross-layer attention routing), plus 020 Forgetting-Attn (needs-run) and
  025 SSMax (done, WIN w/ caveat) live in the same edit-attention-priors
  family. Adding FAM as the 3rd distinct cross-layer mechanism in two weeks
  is fine *if* the bet is sharpened to the marginal contribution: **does a
  recurrent feedback loop beat a static cross-layer shortcut (V-residual)
  and a content-routed cross-layer attention (AttnRes) at our scale?**
- **zero-init safety is good.** The paper's feedback gate is zero-initialised,
  so step 0 reproduces the unmodified model — this is exactly the niche
  constraint we want. Implementation cost is small (~1 block add, gate on
  the feedback). That keeps the A/B cheap even if the expected outcome is a
  null.
- **what to fix for r2:** (1) correct the transfer-risk tag to `high` with a
  one-sentence mechanism argument (long-context lever into short-context
  regime); (2) replace the vibe with a one-sentence sharp bet against
  FIRE-equipped ctrl, e.g. "FAM ≥ -0.02 val loss vs FIRE-ctrl because the
  feedback loop gives the model an explicit working memory that the static
  V-residual shortcut cannot"; (3) one line of "vs the family" — name the
  two priors this run is competing against (021 V-residual and 044 AttnRes)
  and what FAM adds; (4) keep the zero-init gate and the lightweight
  implementation; (5) state explicitly that a clean null closes the
  feedback-loop subfamily at our scale, so the A/B is informative either way.
