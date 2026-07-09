# Long-context architecture levers for the release ladder — ranked shortlist

> Companion to [`LADDER.md`](LADDER.md) (the scaling-law search) and gated by
> [`DECISIONS.jsonl`](DECISIONS.jsonl) **D001/D002**: the 135M release is a
> **long-context** model (long-file coding, long-doc QA / needle retrieval), so
> we **REJECT any mechanism that lowers loss by suppressing distant attention**
> (ALiBi / poly-ALiBi / KERPLE / hard sliding-window / any monotonic
> pre-softmax distance penalty). Levers below either keep FULL O(n²) reach or
> *improve* it. All obey `EXPERIMENT-DESIGN.md` **RULE 0** (structural
> mechanisms only — no LR/wd/momentum/batch/schedule/init sweeps).
>
> Each lever is screened on the tiny ladder, then carried up all 5 rungs as its
> own `_arq` arm and judged by the **fitted L(N) curve at the 135M target N**
> (steeper α preferred over a flat intercept shift) PLUS a long-range capability
> eval — never loss alone (D002 reopen clause).

Baseline arch: RoPE + RMSNorm + squared-ReLU FFN (d_ff = 4·d_model) + Muon +
FULL attention + GQA. Most of these levers are **already wired** in
`models/layers.py` / `train_llm.py` (flag names noted) — the work is the ladder
arm + the long-range eval, not new modeling code.

---

## 1. RoPE base / θ scaling (NTK-aware) — `rope_base` (already wired)

**(a) Mechanism.** Raise the RoPE rotary base from 10 000 to ~100k–500k so the
lowest-frequency rotary dimensions complete fewer turns across the context. One
scalar; no new parameters; full attention untouched.

**(b) Long-range capability.** The lowest RoPE frequencies are the channels that
encode *absolute / coarse* position over hundreds of tokens. With base=10 000
those channels already alias (wrap) well within a few-k context, so distant
positions become indistinguishable — the model literally cannot tell a token
1 000 back from one 3 000 back. A larger base de-aliases them, preserving
distance information at range. This *adds* long-range resolution; it does not
penalize distant tokens, so it is explicitly allowed by D002 (`allows: RoPE
base/theta NTK-aware scaling`).

**(c) Steeper-α plausibility.** Bigger models have more low-frequency channels
and longer effective contexts, so position-aliasing at base=10 000 bites
*harder* the larger the model and the longer the eval sequence — the deficit
the lever removes grows with N and with context length. A constant intercept
shift would not interact with depth/width; this one should widen as rungs grow.

**(d) Ladder arm.** `_arq_ladder*_ropebase.py`: `class C(LadderXConfig):
rope_base = 100000` (sweep {100k, 250k, 500k} once on tiny, carry the winner).
This is a structural positional-scheme change (RoPE geometry), NOT an HP knob —
RULE 0 lists init-magnitude/LR/schedule as banned scalars; rotary base is the
positional mechanism, the same axis ALiBi lived on. Existing configs already use
`rope_base: 500000`, so wiring is proven.

**(e) Eval.** Needle-in-haystack retrieval at 2×–4× the train length (length
generalization is exactly what base scaling buys) + long-file code completion.

---

## 2. Differential attention (Ye et al. 2024) — `use_diff_attn` (already wired)

**(a) Mechanism.** Split each head's d_k in half, compute two softmax maps, and
output `softmax₁ − λ·softmax₂` (λ learnable per-head, paper init 0.5). The
common-mode subtraction cancels attention noise. Implemented via SDPA linearity
in V (two flash calls, no [T,T] matrix) in `models/layers.py:3021`.

**(b) Long-range capability.** Diff-Transformer's headline result is precisely
**better needle-in-haystack retrieval and reduced "lost-in-the-middle"** — the
λ-subtraction removes the diffuse background attention that, in long contexts,
dilutes the signal token among thousands of irrelevant ones. It sharpens
retrieval *without* any distance prior, so it keeps full reach (D002-clean).

**(c) Steeper-α plausibility.** The attention-noise floor that diff cancels
scales with sequence length and head count; the fraction of probability mass
lost to background grows with context, so the correction's value grows with both
N (more heads/layers) and eval length. This is a genuinely different operator
(not identity-init), so it can bend the curve rather than just shift it — at the
cost of convergence risk on the tiny rung (watch the screen).

**(d) Ladder arm.** `_arq_ladder*_diffattn.py`: `class C(LadderXConfig):
use_diff_attn = True`. Needs even d_k (head_dim 64 → fine). One flag.

**(e) Eval.** Needle-in-haystack (the paper's own benchmark) + long-doc QA where
a single fact must be retrieved from distractors.

---

## 3. QK-normalization post-RoPE — `use_qk_norm_post_rope` (already wired)

**(a) Mechanism.** RMSNorm Q and K (after RoPE) before the dot product, so
attention logits depend on the *angle* between Q and K, not their growing norms.
A few d_k-sized norm params per head.

**(b) Long-range capability.** Long sequences are where logit-magnitude blow-up
hurts most: a handful of high-norm tokens saturate the softmax and starve every
distant token of probability mass (entropy collapse). QK-norm caps logit scale
so the softmax stays able to attend far away — it *protects* long-range reach
rather than restricting it, and adds no distance term (D002-clean; it's on the
"allowed" side as a norm mechanism).

**(c) Steeper-α plausibility.** Logit-norm growth is worse in deeper/wider models
and over longer contexts (more tokens, more chances for a norm spike), and
attention-entropy-collapse is a documented *scale* failure mode (it's why QK-norm
is now standard in large training runs, e.g. Chameleon/Gemma-2). The instability
it prevents intensifies with N → plausibly a steepening lever, not a fixed
offset. Also a known training-stability win, which de-risks the expensive rungs.

**(d) Ladder arm.** `_arq_ladder*_qknorm.py`: `class C(LadderXConfig):
use_qk_norm_post_rope = True`. One flag; existing configs at line 6060/6583 use
it, so wiring is proven. (Cheap, stable → good early arm; see note.)

**(e) Eval.** Needle-in-haystack at long range (where entropy collapse bites) +
long-file code completion stability.

---

## 4. Per-head RoPE base (multi-resolution heads) — `use_per_head_rope_base` (already wired)

**(a) Mechanism.** A learnable per-head log-multiplier on the RoPE base
(`per_head_rope_log`, init 0 → baseline at step 0), letting each head pick its
own rotary frequency band — some heads short-range, some genuinely long-range.

**(b) Long-range capability.** A single global base forces every head onto one
position resolution. Letting some heads adopt a *large* base gives the model
dedicated long-range heads that can resolve far-apart tokens, while others stay
fine-grained for local syntax — strictly *more* long-range capacity, never less
(no head is penalized for attending far). D002-clean: it's per-head positional
geometry, the allowed axis. (This is the "head-wise" / multi-scale idea the
brief asks for, expressed on the positional axis.)

**(c) Steeper-α plausibility.** More heads (bigger rungs) = more room to
specialize a subset into long-range heads, so the benefit of head-wise frequency
diversity should grow with head count and with eval length. Risk: init-0
log-multiplier must *grow* to take effect — Lesson 1 (zero-init levers wash out
in 92 steps) means the tiny screen may under-credit it; judge on the **upper
rungs**, where there are real steps to learn the per-head bases.

**(d) Ladder arm.** `_arq_ladder*_phrope.py`: `class C(LadderXConfig):
use_per_head_rope_base = True`. One flag (impl at `models/layers.py:3553`).

**(e) Eval.** Needle-in-haystack across multiple distances (tests whether
long-range heads actually formed) + long-doc QA.

---

## 5. Document-masked / intra-doc attention (no cross-document leakage)

**(a) Mechanism.** Block attention from crossing document boundaries inside a
packed training sequence (an intra-doc causal mask), so a token never attends
into the *previous* unrelated document. No new parameters — a mask change.
(Needs a small forward/data-collate change; not a single existing flag.)

**(b) Long-range capability.** Standard packing lets attention bleed across
concatenated docs, which (i) teaches the model that far-back tokens are usually
*irrelevant noise* — the exact wrong prior for long-context retrieval — and (ii)
wastes capacity modeling spurious cross-doc correlations. Masking to true
document spans makes every long-range dependency the model sees a *real* one, so
it learns to use distant context instead of ignoring it. Critically this is the
opposite of D002's banned mechanisms: it does not down-weight distant tokens, it
makes the *valid* distant tokens the only ones in scope. Strong literature
support (Llama-3, OLMo, "in-context pretraining" / "structured packing").

**(c) Steeper-α plausibility.** The cross-doc-noise tax scales with sequence
length and with model capacity (a bigger model wastes more capacity fitting
spurious cross-doc signal). On a long-range eval the gap should widen with N —
the small model can't model the noise anyway, the large one can and is hurt by
it. This is a curve-bender, not an intercept shift, *specifically on the
long-context eval* (it may barely move packed-LM val loss — judge on capability,
per D002).

**(d) Ladder arm.** Two-part: (1) build doc-boundary positions in the collate
step; (2) add an intra-doc band to the attention mask in the manual-attention
path. New small flag `use_intra_doc_mask`. Heavier wiring than 1–4 → schedule
*after* the one-flag arms confirm the harness.

**(e) Eval.** Needle-in-haystack (does it retrieve the planted fact vs. get
distracted by packed neighbors) + long-doc QA + long-file code completion.

---

## 6. Talking-heads output mixing — `use_talking_heads_out` (already wired)

**(a) Mechanism.** A learnable H×H linear mix across heads applied to the
attention *outputs* (post-AV), letting heads share what they retrieved (Shazeer
et al. 2020). Identity-init → baseline at step 0. No change to attention range.

**(b) Long-range capability.** Long-range retrieval is often a *multi-hop*
operation — one head locates a definition far back, another carries the local
query. Output mixing is the cheapest structural way to let a "finder" head route
its long-range hit into a head that uses it, approximating two-hop reasoning
without any distance bias. It enriches how distant information is *combined*,
never restricting reach (D002-clean). This is the brief's "head-wise output
mixing / two-hop" lever in minimal form.

**(c) Steeper-α plausibility.** Cross-head routing has more to exploit when there
are more heads (upper rungs) and more long-range dependencies to chain (longer
evals), so its value should grow with N and context — not a fixed offset.
Caveat: identity-init H×H must grow off identity (Lesson 1 risk on the tiny
rung); judge on upper rungs.

**(d) Ladder arm.** `_arq_ladder*_talkheads.py`: `class C(LadderXConfig):
use_talking_heads_out = True`. One flag (`models/layers.py:3679`).

**(e) Eval.** Long-doc QA requiring fact composition + needle-in-haystack with
the "needle" needing two cues to locate.

---

## 7. Content-gated residual / attention-output gate — `use_gated_attn` (already wired)

**(a) Mechanism.** A content-dependent (input-conditioned) sigmoid gate on the
attention branch's contribution to the residual stream (`use_gated_attn`,
identity-init), so each token decides per-position how much of the attention
read-out to write back.

**(b) Long-range capability.** In long contexts most positions are local-syntax
positions and a few are genuine long-range retrieval positions; a static
residual treats them identically. A content gate lets the rare
"I-need-to-look-far-back" token open the attention branch fully while local
tokens damp it — increasing the *effective signal-to-noise of long-range reads*
without ever capping attention range (the gate is on the output write, not the
attention scores → D002-clean).

**(c) Steeper-α plausibility.** Conditional computation has more to gain when the
mix of local-vs-long-range positions is richer, which it is at longer contexts
and (arguably) in deeper models with more specialized layers. Weaker steepening
argument than 1–3, and identity-init gate carries the Lesson-1 wash-out risk on
tiny — lowest-confidence of the structural set, kept for breadth.

**(d) Ladder arm.** `_arq_ladder*_gatedattn.py`: `class C(LadderXConfig):
use_gated_attn = True`. One flag (existing config at line 5059).

**(e) Eval.** Long-file code completion (long-range positions are sparse and
identifiable) + needle-in-haystack.

---

## Ranking rationale & wiring order (next arms after `baseline` + `deepnet`)

Best-first ordering balances **(i) long-range capability evidence**,
**(ii) steeper-α plausibility**, **(iii) implementation surface / convergence
risk on the tiny screen**, and **(iv) D002 cleanliness** (all 7 are clean by
construction).

| # | lever | flag | α-bend case | tiny-screen risk | wiring |
|---|---|---|---|---|---|
| 1 | RoPE base scaling | `rope_base=100k+` | strong (de-aliasing grows w/ len) | none (no new params) | trivial |
| 2 | differential attn | `use_diff_attn` | strong (noise floor ∝ len) | medium (new operator) | one flag |
| 3 | QK-norm post-RoPE | `use_qk_norm_post_rope` | strong (entropy collapse ∝ scale) | none (stabilizing) | one flag |
| 4 | per-head RoPE base | `use_per_head_rope_base` | medium (head specialization) | zero-init wash | one flag |
| 5 | intra-doc mask | `use_intra_doc_mask` (new) | strong on eval only | none | collate+mask |
| 6 | talking-heads out | `use_talking_heads_out` | medium (multi-hop) | zero-init wash | one flag |
| 7 | content-gated resid | `use_gated_attn` | weaker | zero-init wash | one flag |

**Easiest to wire as the immediate next ladder arms after `baseline`+`deepnet`:**
**#1 RoPE-base** and **#3 QK-norm** — both are one-line config subclasses on
already-proven flags, add ~zero parameters, are step-0 active (no Lesson-1
wash-out), and are *stabilizing* (low risk on the expensive rungs). Wire those
two first, then **#2 diff-attn** (one flag, the strongest pure long-context
literature result but a genuine new operator → watch the tiny screen for
convergence). **#5 intra-doc mask** is the highest-upside *capability* lever but
needs a collate + mask change, so schedule it once the one-flag arms have
exercised the ladder+eval harness end-to-end.

**Selection discipline (all arms):** a lever earns a 135M slot only if its
fitted L(N) sits below `baseline` at the target N — *and* it does not regress the
long-range eval vs. RoPE full-attention (D002's reopen clause: loss alone never
qualifies a long-context lever). Run the same single lever at every rung, one
lever per arm, so the verdict is interpretable.
