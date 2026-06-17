# LEARNINGS — what ~350 experiments taught us

Durable, scannable distillation of the autoresearch loop so future agents and
contributors don't re-walk dead ends. **Sources of truth:** `closed.md` (the
LOCAL greppable close-log, authoritative — not the Neon subset), `LEADERBOARD.md`
(frozen results), `champion.json` / `records.jsonl` (lineage), and
`DECISIONS.jsonl` (release policy). When this doc and those disagree, those win.

Almost everything below is at the **`tiny1m3m`** screen tier (0.94M params, ~3M
tokens, `d_model=64`, `n_layers=12`, `n_heads=4`, seed 42, ~92–733 steps). That
tier is a fast idea ranker, not a claim. **Nothing counts until it beats the
`10m` record (4.3011)** — read the acceptance rule in `LEADERBOARD.md`. The
**noise band is wide**: paired-3-seed confirm band is **0.02**; raw cross-box
single-seed drift can be **0.04–0.16**. Single-seed deltas inside ±0.02 are
noise. The lucky-seed guard (1-seed WIN → needs-confirm) exists because seed 42
repeatedly drew below the honest 3-seed mean.

---

## 1. Confirmed WINS — the champion lineage

The live champion is **`323` — val 6.1720** (`champion.json`). Lineage, each
step a paired/multi-seed-confirmed record:

| Step | Idea | Val | Δ vs prev | What actually moved loss |
|---|---|---|---|---|
| base | bare tiny1m3m | ~6.43 | — | control |
| 1 | **175-alibi-slopes** | 6.2539* | **−0.16** vs bare | ALiBi linear per-head distance bias. The single biggest lever. (*re-pinned from a lucky 6.2403 to the honest 3-seed mean.) **⚠ banned for release — see §4.** |
| 2 | **253-deepnet-alpha-alibi** | 6.2367 | −0.0172 | DeepNet-α fixed residual scaling (α=1/√24, **0 new params**, step-0 active). First real challenger to alibi. |
| 3 | **267-deepnet-poly-alibi** | 6.2209 | −0.0330 (paired vs alibi) | poly-ALiBi: per-head linear+quadratic distance decay (+48 params, c_h=0 at init). 3-axis stack: alibi + DeepNet-α + poly-alibi. **⚠ poly-alibi is also a distance penalty — banned for release.** |
| 4 | **296-champion-slope-curvature-combo** | 6.1998 | −0.0218 (paired) | step-0 warm-start of the alibi **slope + curvature** schedule (ENV-driven: geometric slope init ×3.0 + geometric poly c init ×3.0). 0 new params. First champion-era record. |
| 5 | **323-mom0p90-lr2x** (current) | **6.1720** | −0.0278 (paired) | muon_momentum 0.95→0.90 **+** peak-LR ×2.0 (muon_lr 0.048 / adamw_lr 0.012). 0 new params. |

Other historically-confirmed WINs (earlier tiers / superseded baselines):
- **009-fire-pe** WIN (Δ−0.064/−0.082) — FIRE positional encoding, the early
  positional lever before alibi.
- **015-moonlight-muon-rms** WIN (Δ−0.0138) and **016-qk-norm** WIN (Δ−0.0185)
  — but see §2: 162/165 later showed the QK-norm win is carried by the **joint
  QK symmetry**, neither Q-only nor K-only alone.
- **021-value-residual** WIN (Δ−0.034, V-side only; Q-side 164-q-carry is NULL).
- **023-canon-conv / 024-gated-attention / 025-scalable-softmax** WIN-with-caveat
  (measured against a buggy FIRE control; effect survives but isolation imperfect).
- On the **`10m`/`screen20m`** tiers the load-bearing recipe was
  **V-embed + per-head q_gain + SWA(window=512) + RoPE base=500k** (best screen
  4.6364). Note SWA there is hard sliding-window — also release-banned (§4).

**Key structural lessons from the wins**
- The biggest tiny-tier gains are **positional / attention-shape priors** (alibi,
  poly-alibi, FIRE) and **step-0 conditioning** (warm-starts, DeepNet-α residual
  scaling) — mechanisms that are *active at step 0*, not ones that need a long
  horizon to develop.
- The two **update-amount knobs (muon_momentum 0.90, ×2.0 LR) STACK
  super-additively** (−0.0085 + −0.0104 sub-band each → −0.0278 combined). This
  is because the model is **update-starved at 3M tokens** (311: batch=1/2×steps
  also helps; 312: grad_accum/½ steps badly hurts). But: momentum does **not**
  stack with batch=1 (322 anti-stack), and LR does not stack with batch=1 (313/314
  substitutes). Stacking is mechanism-specific, never assumed.

---

## 2. CLOSED / SATURATED axes

Each line: axis → one-line why it's closed. (IDs are `closed.md` entries.)

**Positional / attention-shape**
- T5-RPE additive logit bias (**166**, dup **212**) — one-shot per-head bias is
  swamped by accumulated QK dot-products (~10 vs ~0) at T=2048; closes the whole
  additive-logit-bias PE family (also 152 attn-logit-bias, 216 logit-scale).
- per-head temperature (155), per-head RMS gain (160), per-head RoPE base (172),
  QK clamp (195) — all per-head attention-shape levers absorbed by Q/K gradients
  at 0.94M/4 heads.
- QK-norm attribution (162 Q-only, 165 K-only both NULL) — 016's win is the
  **joint** symmetry, not either side; axis closed.
- cosine attention (255 on alibi, 189 no-alibi), talking-heads (177, overpowers
  softmax), xpos-decay (174), NoPE (load-bearing, removing RoPE = +0.44).
- poly-alibi on plain base (230 NULL), kerple-log (231/269), entmax (173/254
  reject), top-k attn (192), cosformer/linear attn (189), focal-mod (148).
- The whole **deepnet+poly-alibi refinement sweep 264–293** — qk-layernorm,
  sub-LN, mix-norm, value-residual, attn-sink, drop-path, rope-base, mup,
  beta-init, embed-sqrtd, pre-lm-head-rmsnorm — **all NULL** on the champion. The
  alibi/poly-alibi/deepnet base is locally saturated to attention-internal tweaks.

**Optimizer / HP-search drift (303–328) — the loop's biggest mistake; see §3**
- LR-value sweep (303–308): clean parabola, ×2.0 best, but confirm 306 = sub-band
  −0.0104. **LR axis CLOSED** (×2.0 is a real lever only when stacked, never alone).
- weight_decay (315/316/319): 0.0/0.1/0.3 all ~+0.017 worse; **0.2 default
  optimal**, axis fully closed.
- muon_momentum (317–321): convex parabola, optimum 0.90 = −0.0085 sub-band.
- embedding_scale (309/310): default sqrt(d_model)=8 optimal, both off-default
  worse — closed.
- batch/#steps (311–314): confirms update-starvation but caps at −0.0104 sub-band;
  optimization-amount axis closed at the sub-band level.
- schedule (305 no-warmup NULL, 324 cosine NULL).

**Optimizer zoo — tier-mismatch nulls (need a long horizon, dead at 92 steps)**
- 110-weight-ema, 132-born-again, 134-mega-ema — EMA windows (≫1000 steps) longer
  than the run.
- 112-lookahead, 113-galore, 114-mars, 119/138-SAM, 122-tiger, 124-radam, 125-psgd,
  126-adashift, 135-adan, 136-adapnm, 137-adamp, 139-lion, 141-adabelief, 003-soap,
  001-cautious-muon, 006-schedule-free — adaptive-LR / preconditioner / projection
  optimizers under-shoot or diverge at the 92-step horizon (several catastrophic:
  120-dadaptation, 121-prodigy, 123-came overflow). Mechanism not falsified, just
  invisible at this tier.
- 140-sophia, 115-rdrop, 150-xlayer, 254/185/186/173/180 — **reject (code/def-gate
  cap)**: 3 recode rounds exhausted, mechanism unrunnable in our harness, not a
  scientific null.

**Capacity / FFN / MoE (closed at 0.94M, FFN is not the binding constraint here)**
- 117-soft-moe, 118-mixture-of-depths, 145-expert-choice, 146-sparse-ffn, 156-moa,
  144-mos — capacity/routing overhead dominates at d_model=64; FFN binds only at
  ≥135M.
- FFN activation/gating: 153-relu2, 157-conv-ffn, 158-gau, 170-swiglu, 338-mish-glu
  — all NULL; activation curvature doesn't matter at this width.

**Depth-conditional levers (NULL at L=12, need L≥24)**
- 017-sub-ln-sandwich, 111-drop-path, 116-hyper-connections, 130-rezero,
  142-layerscale, 131-layer-drop, 056-branchnorm — residual-stability/depth levers
  with nothing to stabilize at 12 layers.

**Loss-shapers / regularizers (no val-loss axis to win on)**
- 010-polyloss, 066/067 label-smoothing/conf-penalty, 068-unlikelihood, 069-focal,
  070-mtp, 147-dropkey, 167-logit-zloss, 171-dropconnect, 133-seqmix — either
  generation-quality (not loss) or paper says they need ≥3B / classification.

**Cross-block / structural bolt-ons (208–352, late session)**
- 208-216 orthogonal bolt-ons on alibi (value-residual, qk-layernorm, swiglu,
  gated-attn, ssmax, cope, canon-conv) — **all NULL/inside variance**.
- 329–352 novel-arch shortlist (gmlp-sgu, q-feature-map, se-pre-wo, lowrank-w*,
  short-conv, unet-skips, tied-output, cross-block-ffn-share, qk-bilinear,
  layerscale) — **all NULL**. Several (339/342/343 conv-ffn / q-time-conv /
  v-mix-conv) were **LEAK rejects**: val ~0.43 means a broken causal mask, not a win.

**Tier saturation**
- The whole **tiny1m3m embed/gain family** (V/Q/K/O embeds + q/k gains, screen rows
  0–17) and the **tiny arch sweep** (tied-QK, MHA/GQA, MLA, post-norm) are mined
  out — only step-0 conditioning levers still win. The tier is **saturated**.

---

## 3. DO NOT RETRY

Crisp, with the reason. (RULE 0 in `EXPERIMENT-DESIGN.md` codifies the big one.)

1. **NO hyperparameter search.** Do not sweep LR, weight_decay, momentum, batch
   size, grad_accum, warmup, schedule, or emb_scale. 303–328 drifted into exactly
   this and burned ~25 runs to find sub-band gains that only mattered stacked. The
   operator directive (2026-06-17): *"try novel architectures, not basic
   hyperparam search."* All those axes are characterized and closed.
2. **No additive per-head logit-bias PE** (T5-RPE, attn-logit-bias, logit-scale) —
   166 closed the family; the bias is swamped by QK magnitudes at T=2048.
3. **No per-head attention-shape scalars** (temperature, RMS gain, per-head RoPE
   base, QK clamp/norm-half) — absorbed by Q/K gradients at 4 heads / 0.94M.
4. **No FFN capacity/activation/MoE levers at tiny1m3m** (soft-MoE, MoD, MoA, MoS,
   sparse-FFN, GAU, swiglu, relu2, conv-ffn) — FFN is not binding below ~135M.
   Defer to Phase-2.
5. **No long-horizon optimizers at the 92-step tier** (EMA/born-again/mega-ema,
   tiger/radam/psgd/adashift/adan/lion/sophia/galore/SAM, dadaptation/prodigy/came)
   — they need 3–4k steps; nulls here are tier-mismatch, not signal.
6. **No depth-conditional residual levers at L=12** (sub-LN, drop-path,
   hyper-connections, rezero, layerscale, layer-drop, branchnorm) — nothing to
   stabilize until L≥24.
7. **No loss-shaping/regularizers as val-loss bets** (label-smoothing, focal,
   unlikelihood, MTP-head, dropkey, dropconnect, z-loss, seqmix) — wins (if any)
   are on generation metrics we don't measure, or need ≥3B params.
8. **No deepnet+poly-alibi attention-internal refinements** — 264–293 closed ~25
   of them NULL. The current base is locally saturated; needs a *new independent
   lever*, not more tweaks.
9. **If a tiny1m3m run reports val < ~5 (e.g. 0.43/0.98), it's a LEAK** (broken
   causal mask), not a record. Real tiny1m3m floor is ~6.17. Auto-reject; build
   base and candidate the SAME way (most use_* flags are inert step-0 — a
   construction mismatch fabricates a 0.10+ artifact).

---

## 4. LONG-CONTEXT PRINCIPLE — the loss-game vs the release diverge

Recorded in `DECISIONS.jsonl` **D001** and **D002** (2026-06-17, status:
committed). Read those entries verbatim before proposing any positional or
attention-reach mechanism.

- **D001:** No ALiBi (or ALiBi-style linear recency bias) in the 135M release.
  ALiBi lowers perplexity by making each head a **soft recency window** — it
  *suppresses* distant tokens instead of using them. Baseline is RoPE
  (SmolLM2-class). Reopen only if a long-range eval shows ALiBi ≥ RoPE on
  *capability*, not loss.
- **D002:** Long-context capability is a **first-class, non-negotiable objective**
  of the 135M release. **REJECT any mechanism whose loss gain comes from
  suppressing/penalizing attention to distant tokens, even if it lowers val
  loss.** Generalizes D001 from "no ALiBi" to "no distance-punishing attention."
  - **Rejected for release:** `use_alibi_bias`, `use_poly_alibi`, KERPLE-style log
    bias, **hard sliding-window / local attention that caps reach**, any monotonic
    distance penalty on pre-softmax scores.
  - **Allowed:** RoPE + base/θ (NTK-aware) scaling, document/intra-doc attention
    masks, full O(n²) attention, sparse/selective attention that **preserves reach**.

**The critical divergence to internalize:** the tiny-tier champion's single
biggest lever — **alibi → poly-alibi (lineage 175 → 230/267, ~−0.16 then
−0.033)** — falls squarely in the **banned** category. It wins val loss precisely
*by* turning each head into a recency window. **Its loss win does NOT transfer to
the release objective**, so the poly-alibi ladder arm is **CUT**, leaving baseline
+ deepnet. Likewise the `10m`-tier SWA(window=512) win is a hard range cap — also
banned.

**Practical consequence for the loop:** val loss is a **misleading selection
metric** for the release. A tiny1m3m record built on distance-penalizing attention
is a dead end for the 135M goal even though it tops the leaderboard. Future levers
should win loss *without* punishing distant attention — i.e. genuinely use
long-range context (the thing long-file coding / long-doc QA / needle-retrieval
actually need). The DeepNet-α residual scaling and the step-0 / optimizer levers
(253, 296, 323) are release-safe; the alibi/poly-alibi positional core is not.
