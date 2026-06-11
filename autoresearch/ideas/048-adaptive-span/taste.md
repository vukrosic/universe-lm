# Taste log — 048 adaptive-span

## r2 — 2026-06-11 — verdict: revise
- **Repitch is built on a wrong window size — fix the baseline before accept.**
  The miner claims *"The current SWA-384 closed-win (uniform cutoff wins over
  256/512/768/1024/2048)"*. This is factually wrong on both halves:
  (a) `closed.md` records the SWA sweep winner as **512**, not 384 (the
  canonical "SWA window sweep (256/384/512/768/1024/2048) — 512 winner"
  line); (b) the fire-ctrl the pipeline actually launches against —
  `Tiny1M3MVQGainSWAHighRoPE250KConfig` in `configs/llm_config.py` — has
  `sliding_window_size: int = 512`, not 384. There is no SWA-384 winner
  in the codebase. The A/B as written is therefore unrunnable: the
  head-span identity-init at 384 starts as a *shrinking* gate over
  SWA(512), not a no-op. Fix: set the A/B to `Tiny1M3MVQGainSWAHighRoPE250KConfig
  ctrl` vs `same + per-head learnable span (init at 512)`, and re-state
  the composition with SWA(512) (the head-span mask caps where the head
  wants ≤512, lets SWA(512) decide the rest, and a head with `s_h > 512`
  gets the full SWA(512) window — i.e. span above 512 is *wasted
  capacity* unless the underlying mask is also widened, which it isn't
  here, so the upper bound on `s_h` is effectively 512 for this ctrl).
- **Rest of the repitch lands.** The composition spec is clean
  (multiplicative `SWA · head_span_mask(s_h)`, identity-relative-to-SWA
  at step 0), the bet is sharp (sink head ≤16, broad head toward seq_len,
  ≥0.005 val loss), both branches are informative (WIN → per-head
  structural prior carries to 135M; NULL → re-confirms SWA-uniform and
  kills #97 multi-scale heads as a follow-up), and the scale-transfer
  argument (head-level span heterogeneity is reported across scales, not
  a long-context-only artifact) is the right framing. These survive
  the window-size fix; the *only* thing standing between the repitch
  and `accept` is getting the baseline right.
- **One more sharpen pass — the upper-bound on `s_h`.** Even after
  the window-size fix, with the underlying mask still at SWA(512),
  letting `s_h` grow past 512 has no effect. Either (a) explicitly cap
  `s_h ≤ sliding_window_size` and state that a head "wants more" is
  not a signal the model can express, or (b) the broader-context
  treatment grows `sliding_window_size` *together* with `s_h` (i.e.
  the head's effective mask is `min(s_h, sliding_window_size)`, but a
  head with `s_h > sliding_window_size` triggers widening the base
  SWA). Option (a) is the cleanest test of "does heterogeneity help
  within the existing SWA budget?" Option (b) is a different lever
  ("heads can vote to widen the global cap") and a much bigger LoC
  commitment. Pick (a) for this A/B — it isolates the per-head
  *parameterisation* claim, which is the actual bet.
- **Portfolio fit is fine.** Per-head mask heterogeneity is a different
  family from the active needs-taste neighbours (talking-heads/041,
  knocking-heads/042, attnres/044 are all logit-mix at the attention
  matrix level; synthesizer/047 is attention-replacement). Adaptive
  span is a *mask-level* head lever — additive, distinct.

## r3 — 2026-06-11 — verdict: accept
- The SWA-512 baseline is now correct, and the cap at 512 makes the lever a clean per-head parameterisation test.
- The mechanism and distribution claim are sharp enough to evaluate at tiny1m3m.
- The slot is informative either way and no longer depends on a wrong baseline.

## r1 — 2026-06-11 — verdict: revise
- **Pitch is pre-empted by a closed lever.** `closed.md` records `SWA window
  sweep (256/384/512/768/1024/2048) — 512 winner`, and `LEADERBOARD.md`
  tiny1m3m has SWA(window=384) baked into every winning row (and into the
  shared fire-ctrl `Tiny1M3MVQGainSWAHighRoPE250KConfig`). The current "Why
  it's worth a slot" — *"simplest way to learn whether tiny1m3m is wasting
  compute on far-history tokens, and a null would say the model wants full-
  span attention everywhere"* — is structurally falsified before the run:
  the SWA-384 win already proves the model does **not** want full-span
  everywhere. So the framed null cannot fire.
- **Re-pitch as adaptive span *on top of* SWA-384, not vs full context.** The
  real bet is whether per-head learnable span beats the uniform SWA-384
  cutoff. The A/B is `Tiny1M3M…SWA384 (uniform) ctrl` vs `same + per-head
  learnable span (initialised at 384)`. Identity-init means the span gate
  starts as a no-op over the SWA mask and only shrinks (or, with a softplus,
  grows toward seq_len=2048) where the head wants it. That's the mechanism;
  state it explicitly.
- **Make the crisp bet a distribution claim, not a cutoff claim.** Sharpen
  to one sentence: *"heads will specialise — a sink-style head shrinks to
  ≤16, while ≥1 head pushes back toward seq_len=2048 — and that
  heterogeneity beats uniform-384 by ≥0.005 val loss."* Now both branches
  are informative: WIN → head heterogeneity > uniform window (mechanism for
  the 135M recipe). NULL → learned spans concentrate near 384 → re-confirms
  SWA-384 as the right uniform cap, kills "multi-scale heads" (#97 in
  `llm_config.py` and "Multiscale heads" in `closed.md`) as a follow-up.
  Either outcome is a slot worth spending.
- **Address scale transfer head-on.** `transfer-risk: high` is already
  tagged (good), but the Scale evidence section currently just lists the
  paper's text8/8k context provenance. Add one sentence on why head-level
  span heterogeneity is a *general* phenomenon (sink-vs-broad heads are
  reported across scales) and not a long-context-only artifact — otherwise
  the lever is hard to carry into the 135M recipe even on a WIN.
- **Spec the composition with SWA cleanly.** Say whether the learned span
  replaces SWA, masks within SWA, or runs without SWA. Without this the
  code gate cannot disambiguate; with it the bet is real and small. Param
  cost is tiny (1 scalar / head = 4 params) so this is *not* a budget
  concern — the budget concern is bet sharpness, not LoC.
