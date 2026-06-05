# Query-side ablations — research plan

**For the implementing AI.** This folder is self-contained. Read this file top to
bottom, implement the batches in order, record numbers, then fill the
[tutorial scaffold](tutorial/). Do not claim a result until it clears the
protocol below.

Idea-bank source (do not edit, reference only):
[../../research-plans/query-tweaks/plan.md](../../research-plans/query-tweaks/plan.md)
+ [critique.md](../../research-plans/query-tweaks/critique.md). This plan promotes
the highest-signal subset into runnable experiments.

---

## Research question

The query path already has shipped levers: query-embed (`#30`), scalar per-head
`q_gain` (`#37`), tied-QK (`#72`), QK-norm position (`#49`), `rope_base` (`#63`),
`attn_sink` (`#99`). All are either a scalar or a tie. **What query-side mechanism
buys more than one scalar's worth of loss at a fixed parameter budget?**

Three axes, asked separately so wins are attributable:

- **Positional** — is there a better query/position interaction than a single global RoPE base?
- **Similarity** — is the plain scaled dot-product the bottleneck, or does a richer Q·K score help?
- **Capacity** — does a token benefit from issuing more than one query per head?

---

## Implementation contract

Where things live:
- Attention: `MultiHeadAttention` in [models/layers.py](../../../models/layers.py)
  (class at L203, `forward` at L509). Existing flag pattern: `use_q_gain`,
  `use_attn_sink`, `rope_base`, `qk_norm_type` — copy this style.
- Configs: [configs/llm_config.py](../../../configs/llm_config.py). Base tier is
  `Screen10M20MConfig` (L305). Each experiment gets a
  `class Screen10M20M<Name>Config(Screen10M20MConfig)` that flips exactly one flag.
- Run: `python train_llm.py --config <name> --seed 42` (see existing `--config` keys).

Hard rules (from the idea-bank critique):
1. **Identity / zero-init.** Every new lever must make step-0 logits *bit-identical*
   to the clean baseline (zero-init the new params, or gate them with a zero scalar).
   The A/B then isolates the mechanism, not a re-seed. Note any idea that can't.
2. **One flag per config.** No stacking inside an experiment. Stacking is a separate,
   later batch only after the single levers have a sign.
3. **Optimizer routing.** 2D mixing matrices / low-rank factors → Muon; per-head
   scalars / biases → AdamW. After wiring, confirm the new params actually receive
   gradient (a zero-init param under the wrong optimizer can silently stay zero).
4. **Wire a flag, never a fork.** Reuse the existing forward; guard new code with
   `if self.use_<x>:` so the baseline path is untouched.

---

## Protocol (what counts as a result)

| Tier | Config | Purpose | Claim? |
|---|---|---|---|
| tiny | `Tiny1M3M*` | rank ideas fast (~2 min/run) | no — kill bad ideas only |
| screen | `Screen10M20M*` | confirm sign survives 20M tokens | **3-seed mean (42/43/44)** before any claim |
| full | `Full10M200M` | the real target | a winner here only |

A screen winner re-runs on the full ladder before it's a claim. Screen tier
**does not transfer-promote**. Record val_loss vs the clean `Screen10M20MConfig`
control (currently **4.7984**, run dir `s_ctrl_full`). A lever is "live" only if
the 3-seed mean beats control by **≥0.01** and the seeds don't straddle zero.

---

## Batch 1 — run these first (high signal, low lift)

Run order is top to bottom. Stop a lever if tiny-tier shows it clearly washing.

| # | Name | Mechanism | Spec (step-0 == base) | Params/block | Opt | Lift | Conf |
|---|---|---|---|---|---|---|---|
| Q1 | `AlibiBias` | non-rotary positional prior | scores `+= -m_h·(i-j)`, per-head slope `m_h`, **init m=0** | n_heads | AdamW | S | med |
| Q2 | `QTempToken` | token-conditioned query temperature | `Q *= (1 + tanh(x·w_h))` per head, **init w=0** | n_heads·d_model | Muon | S | med |
| Q3 | `CosineAttn` | bounded logits | L2-normalize Q,K then per-head learnable temperature τ (init τ→ matches `1/sqrt(d_k)`) | n_heads | AdamW | S | med |
| Q4 | `QKBilinear` | per-channel relevance | score `Qᵀ diag(d_h) K`, **init d_h = 1** (exact dot at step 0) | n_heads·d_k | AdamW | S | med |

Head-to-head A/B to also record: **Q1 (`AlibiBias`) vs the shipped `attn_sink`** —
both are token-independent score priors; the screen budget only justifies one.

## Batch 2 — flagship + positional structure (more lift, gated on Batch 1)

| # | Name | Mechanism | Spec | Params/block | Opt | Lift | Conf |
|---|---|---|---|---|---|---|---|
| Q5 | `TalkingHeadsQ` | cross-head logit mixing (Shazeer) | mix attention **logits** across heads via learned `n_heads×n_heads` M, **M=I init** | n_heads² | Muon | M | med |
| Q6 | `PerHeadRopeBase` | multiscale position | each head learns its own rotary base (Q&K), **init = global `rope_base`** | n_heads | AdamW | S | med |
| Q7 | `PartialRotary` | content/position split | rotate only fraction `p` of Q/K dims, rest position-free, **p=1 == base** | 0 | — | S | med |

## Batch 3 — exotic, only if Batch 1–2 leave loss on the table

`QExpansion` (2 queries/head, 2nd zero-init), `DecoupledContentPos` (DeBERTa-style
two score streams), `AntisymQK` (skew term `Qᵀ S K`, S=0 init). Specs in the idea bank.

---

## Batch 4 — query-norm zoo (cheap, mostly free)

The repo's `make_norm` already accepts: `rmsnorm`, `layernorm`, `pnorm<p>`,
`clipnorm<k>`, `center`, `centeredl1`, `channelscale`, `manhattan`, `manifold`,
`median`, `peak`, `squash`. Today `qk_norm_type` ties the Q and K norm together.

**Prereq (one small wire):** add a `q_norm_type` flag so the query norm can be set
independently of K (default falls back to `qk_norm_type` → step-0 unchanged). Then
sweep the query norm only. None of these are param-heavy; most are identity-ish.

| # | Name | Query norm | Note |
|---|---|---|---|
| Q11 | `QNormPnorm15` | `pnorm1.5` | sub-quadratic norm — softer outlier handling on Q |
| Q12 | `QNormClip` | `clipnorm3` | clips query magnitude — caps logit blow-up via Q side |
| Q13 | `QNormChannelScale` | `channelscale` | per-channel learnable scale folded into the norm |
| Q14 | `QNormManhattan` | `manhattan` | L1-geometry query — tests if dot-product wants L1 normalization |
| Q15 | `QNormCenter` | `center` | mean-subtract query before scoring |
| Q16 | `QNormNone` | identity (skip q_norm) | ablation floor — is normalizing Q load-bearing at all? |

Run the 2–3 most distinct (`channelscale`, `manhattan`, `none`) at tiny first; only
promote a clear tiny mover to the 3-seed screen.

## Batch 5 — learnable-parameter zoo on Q (one knob each)

Each adds the smallest possible learnable thing to the query and asks if it pays.
All identity/zero-init unless flagged.

| # | Name | Mechanism | Spec (step-0 == base) | Params/block | Opt |
|---|---|---|---|---|---|
| Q17 | `QBiasHead` | per-head query bias | `Q += b_h` after q_norm **and** after RoPE (a non-rotated constant prior) | n_heads·d_k | AdamW |
| Q18 | `QGainChannel` | per-channel query gain | `Q *= (1 + g_c)`, `g_c=0` init — richer than the per-head scalar `q_gain` | d_k | AdamW |
| Q19 | `QGainHeadChannel` | per-head × per-channel gain | `Q *= (1 + G_{h,c})`, `G=0` init | n_heads·d_k | AdamW |
| Q20 | `QGateNorm` | norm-conditioned gate | per-head `σ(a_h·‖x‖ + b_h)` scaling Q, init gate ≈ 1 | 2·n_heads | AdamW |
| Q21 | `QResidualLowRank` | low-rank query refinement | `Q += U(φ(V·Q))`, U zero-init so step-0 == base | 2·d_k·r | Muon |
| Q22 | `QLayerScale` | per-layer query LayerScale | scalar `s` on the query's contribution, `s` init small/zero-gated | 1 | AdamW |
| Q23 | `QSoftplusGain` | positive-only gain reparam | `q_gain = softplus(w)`, `w` set so gain ≈ 1 | n_heads | AdamW |

## Batch 6 — query architecture / mixing (more lift, breadth screen)

Higher-variance "does this whole shape help" probes. Expect more nulls; keep the
identity-init rule so a null is informative.

| # | Name | Mechanism | Spec | Lift |
|---|---|---|---|---|
| Q24 | `QHeadMix` | vector-level cross-head query mix | `Q' = M·Q` across heads, `M=I` init (sibling of Q5's logit-level mix) | M |
| Q25 | `QTimeConv` | depthwise causal conv on Q | short (k=3) per-channel causal conv over time on Q, init = identity tap | M |
| Q26 | `QEMASmooth` | temporal query smoothing | `Q_t ← (1-α)Q_t + α Q_{t-1}`, learnable `α`, `α=0` init | S |
| Q27 | `QFeatureMap` | positive query feature map | apply `elu+1` / `relu` to Q (linear-attn style positivity) — **not** identity-init, note baseline shift | S |
| Q28 | `QPerTokenRope` | query-conditioned rotation | learned per-token scalar scales the RoPE angle, `0` init == base | M |
| Q29 | `QNoiseReg` | stochastic query (train-only) | add Gaussian noise to Q during training, σ schedule → 0; eval clean | S |

**Selection rule for Batches 4–6:** these are a *breadth screen*. Run tiny first,
keep only the top ~2 per batch for the 3-seed screen. The point is to map which
query axis is alive, not to run all 19 at full cost.

---

## When a batch finishes

1. Append numbers to [tutorial/results.md](tutorial/results.md) (control + each lever,
   3-seed mean + std, run dir, commit).
2. Update status in [tutorial/experiments.md](tutorial/experiments.md).
3. Once a batch has a clear story (win or instructive null), draft
   [tutorial/README.md](tutorial/README.md) following the house style of
   [../../tutorials/qk_gain/README.md](../../tutorials/qk_gain/README.md):
   problem → one mechanism → fair baseline → number → what it means.
4. Add the run's `metrics.json` to `runs/` and re-run `runs/make_evidence_index.py`.
