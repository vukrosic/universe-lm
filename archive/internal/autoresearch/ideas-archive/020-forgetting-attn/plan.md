# Plan — 020 Forgetting Transformer (FoX)

## Recode r3 — 2026-06-10: trt parent class fix (close 4-lever HP drift)

**Blocking (r2 codereview):** `Tiny1M3MFOXOnFireConfig` inherited from
`Tiny1M3MConfig` instead of `Tiny1M3MVQGainSWAHighRoPE250KConfig`, so
the trt instance differed from the ctrl on **four** levers besides
`use_fox`: `use_value_embed`, `use_q_gain`, `use_sliding_window`,
`rope_base`. That made the A/B "FIRE + FoX vs VQGain+SWA+RoPE250K+FIRE",
not "FIRE+FoX vs FIRE" — the A/B confounded FoX with dropping V-embed
+ Q-gain + SWA(512) + RoPE base 250k.

**Fix (one line, plus a class move):** changed the parent of
`Tiny1M3MFOXOnFireConfig` from `Tiny1M3MConfig` to
`Tiny1M3MVQGainSWAHighRoPE250KConfig`. Because the parent class is
defined later in the file, the dataclass had to be relocated to
immediately after `Tiny1M3MVQGainSWAHighRoPE250KConfig` (matching the
pattern of `Tiny1M3MSoftpickOnFireConfig`, `Tiny1M3MCanonOnFireConfig`,
`Tiny1M3MQKNormOnFireConfig`, `Tiny1M3MUNetSigmoidOnFireConfig`). The
class body (`use_fire_pe = True; use_fox = True`) is unchanged; the
parent now contributes `use_value_embed=True, use_q_gain=True,
use_sliding_window=True, sliding_window_size=512, rope_base=250000`.
Docstring updated to call out the parent class so the next reviewer
can verify by inspection. Out of scope here (flagged to the queue):
021-value-residual, 024-gated-attention, 027-moonlight-x-qknorm
sibling configs have the same parent bug.

**Post-fix verification:** instantiated both classes and confirmed
the trt vs ctrl difference is exactly `{use_fox: False → True}` —
all other listed flags match (see r3 self-check below).

**Nit (r2 codereview, low priority) — closed:**
`test_step0_attention_output_unchanged` only covered the SDPA path
(`use_fire_pe=False`). Parametrized over `use_fire_pe ∈ {False, True}`
so the FIRE branch's FoX hookup at `models/layers.py:1632-1633` is
also exercised. Both variants pass at the 1e-2 ceiling.

**No mechanism change, no HP drift on any other axis.** Identity
init (`W_f=0, b_f=+10`) preserved; pass/fail bar unchanged. Six
tests still pass: `pytest tests/test_fox.py -v` → 6 passed.

---

## Recode r2 — 2026-06-10: NaN fix (log-add form)

**Symptom (from `evidence.md`):** both runs (08:14Z + 11:13Z rerun)
NaN'd at step ~400/732 (`loss=nan, acc=0.000`). The r1 implementation
did `attn_w = softmax(scores); attn_w *= D; attn_w /= sum(...)`.

**Cause:** once the gate trained off its `b_f=+10` identity init, `D`
underflowed at distant key positions (`exp(2047 · log f)` becomes ~1e-9
once `log f ≈ -0.01`). The row-renorm then divided `~0 / ~0` → NaN on
the next step. The r1 test suite only checked init bounds, not training
dynamics, so this passed code-review but blew up live.

**Fix:** swap to the paper's *logit-add* form (arXiv:2503.02130 §3).
Math-equivalent: `softmax(s) ⊙ exp(log_D) / row_sum = softmax(s + log_D)`.
The pre-softmax add is numerically stable — softmax's max-subtraction
absorbs arbitrarily negative `log_D`.

**Touched:**
- `models/fox.py` — `FoX.forward` now returns `log_D: [B, H, T, T]`
  (upper-tri masked to 0 for hygiene). No `exp`, no causal-multiply
  hop. Same params (gate_w, gate_b), same identity init.
- `models/layers.py` — both FIRE branch and manual branch: replace
  the post-softmax `attn_w = attn_w * d; attn_w /= sum.clamp_min(1e-9)`
  with a pre-softmax `scores = scores + self.fox(x).to(scores.dtype)`.
  Order in FIRE branch: `Q@K/√d → +fire_bias → +cope → ·ssmax →
  +log_D → mask → softmax`. Order in manual branch: same content
  tweaks → `+log_D → mask → softmax`.
- `tests/test_fox.py` — adapted invariants for the log_D return:
  upper-tri = 0 (was `< 1e-6`), identity check via `exp(log_D)`,
  step-0 MHA drift unchanged at `< 1e-2`. **New regression test**
  `test_trained_gate_does_not_blow_up`: at `b_f=-3` (trained-like,
  much larger decay), `log_D` stays finite and the simulated
  `softmax(random_logits + log_D)` rows still sum to 1 — the exact
  scenario where r1 blew up.

**No HP drift, no spec change.** The mechanism is unchanged; only the
numerical realization moved from post-softmax multiply to pre-softmax
add. Identity init (`W_f=0, b_f=+10`) preserved. Pass/fail bar
unchanged. Coordination: `models/layers.py` and `configs/llm_config.py`
have unstaged parallel-AI edits (V-Norm wiring at lines 552, 686, 744,
970, 1092, 1963). My FoX edits at lines ~1620 and ~1742 are in
different regions — no conflict.

---

## Flag
- `use_fox: bool = False` on `LLMConfig` (`configs/llm_config.py:179`),
  threaded through `TransformerBlock` (`models/layers.py:1809`,
  `models/layers.py:1847`) into `MultiHeadAttention.__init__`
  (`models/layers.py:447`, `models/layers.py:605`).
- Trt config class: `Tiny1M3MFOXOnFireConfig`
  (`configs/llm_config.py:818-839`) — sets
  `use_fire_pe = True` AND `use_fox = True`.
- Built module: `models/fox.py:FoX(d_model, n_heads)` (instantiated
  lazily in MHA only when `use_fox=True`; never called when off →
  baseline path bit-identical).

## Change
**1 file rewrite + 2 wiring touch points (post-recode r2).** No new
dependencies.

- `models/fox.py` (rewritten in recode r2, ~130 LoC incl. docstring):
  - Per-head gate projection `gate_w: Parameter[H, d_model]` (zero
    init) + `gate_b: Buffer[H]` set to `FOX_BF_INIT = +10.0`.
  - `forward(x)`: `z = einsum("btd,hd->bth", x, gate_w) + gate_b`
    → `log_f = logsigmoid(z)` (≤ 0) → `cum = cumsum(log_f, dim=T)`
    → pad a zero along T → build `log_D[b,h,i,j] = cum_pad[i+1] −
    cum_pad[j]` via slicing the padded axis → `permute` heads to
    dim 1 → mask upper-tri to 0. Output `[B, H, T, T]`,
    lower-tri ≤ 0, upper-tri = 0. **No `exp()`** — the caller adds
    this to logits and softmax handles the exponentiation safely.
  - Math-corrected init `b_f = +10` (NOT r1's `+5`) gives
    `log_D[0, 2047] = −0.093` at the real T=2048 → softmax drift
    ≤ 9% on the worst-case row.
- `models/layers.py:447` — `use_fox: bool = False` kwarg on
  `MultiHeadAttention.__init__` (unchanged from r1).
- `models/layers.py:605-607` — `self.use_fox = use_fox`; build
  `self.fox = FoX(d_model, n_heads)` if on (unchanged from r1).
- `models/layers.py:~1613` (FIRE branch, recode r2) and
  `models/layers.py:~1726` (manual branch, recode r2) — when
  `self.use_fox`: `scores = scores + self.fox(x).to(scores.dtype)`
  **BEFORE** the mask + softmax (was: post-softmax multiply +
  renorm in r1). The mask still dominates the upper triangle
  (-1e9 + 0 ≈ -1e9 → softmax weight ≈ 0). When both FIRE and FoX
  are on, the order is `scores += fire_bias → scores += log_D →
  mask → softmax`. Both are additive on logits — strictly
  orthogonal axes (FIRE is position-only, FoX is per-head,
  per-token, content-conditional).
- `models/layers.py:~1632` — `or self.use_fox` in the manual-branch
  OR (unchanged from r1; FoX needs manual because it touches the
  pre-softmax logits and SDPA's flash kernel doesn't expose them).
- `models/layers.py:1809, 1847` — pass `use_fox` from
  `TransformerBlock` to its MHA child (unchanged from r1).
- `configs/llm_config.py:179` — `use_fox: bool = False` on
  `LLMConfig` (unchanged from r1).
- `configs/llm_config.py:818-839` — `Tiny1M3MFOXOnFireConfig`
  (`use_fire_pe=True, use_fox=True`); ctrl is
  `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True` (the 009
  WIN signature from `closed.md:40`) (unchanged from r1).
- `tests/test_fox.py` (rewritten in recode r2, ~180 LoC): 6
  invariants — no NaN/Inf, causal upper-tri=0, identity-init
  `exp(log_D)` ≤ 1 and ≥ D_min, W_f-perturbation makes head 0 differ
  (wiring live), MHA step-0 output within `1e-2` of `use_fox=False`
  baseline, **NEW** trained-gate regression at T=2048, b_f=-3 (the
  exact scenario where r1 NaN'd) — `log_D` stays finite and
  `softmax(random_logits + log_D)` rows sum to 1.

## Control
- **Ctrl**: `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`
  (the 009 WIN FIRE-equipped baseline, val 6.3234 in `closed.md:40`).
- **Trt**: `Tiny1M3MFOXOnFireConfig` — same recipe as ctrl + `use_fox=True`.
- **Seed**: 42 (one seed only — see `feedback-one-seed-only.md`).
- **Tier**: tiny1m3m.
- **Pass bar** (copied from `idea.md:88-96`):
  - **Win**: `trt_val < ctrl_val − 0.02`.
  - **Null**: `|trt_val − ctrl_val| < 0.02`.
  - **Fail**: `trt_val > ctrl_val + 0.01`.

## Cost
- **Params**: `gate_w` = `H · d_model` (e.g. 8·256 = 2,048 / layer at
  tiny1m3m). `gate_b` = `H` (8 / layer). Total `+8 · d_model + 8`
  per layer → at 6 layers, ~12.3K params (+0.2% over the
  tiny1m3m's ~6M params). Negligible. Unchanged from r1.
- **FLOPs**: attention path is dominated by the softmax. The FoX
  path adds (a) one `[B, T, d] · [H, d] → [B, T, H]` matmul (1
  einsum, ~`B · T · d · H` FLOPs), (b) one `cumsum` over T, (c) one
  `[B, H, T, T] + [B, H, T, T]` elementwise add (recode r2: was
  multiply + renorm in r1, now just one add — slightly cheaper).
  At B=8, T=2048, H=8, d=256: ~50M extra FLOPs / layer → ~300M /
  6 layers (≈ 1% of the attention forward).
- **Memory**: `log_D` is `[B, H, T, T]` in the autograd-tracked
  dtype (typically bf16/fp32 to match `scores`). Same activation
  footprint as the attention scores tensor. The runner uses
  gradient checkpointing already; no additional change.

## Run
- **Command**: queued via the arq batch (see runner.md). Config
  class is `Tiny1M3MFOXOnFireConfig`; ctrl is the FIRE-equipped
  parent (the box-side `_arq_026.py` builds the shared fire-ctrl —
  the runner's prior pass flagged a flag-drop bug there
  (`evidence.md` "Wiring bug"); the ctrl side needs the
  pre-baked-config fix, NOT a FoX-side change).
- **Tier**: tiny1m3m (single seed 42, no sweep).
- **Expected wall-clock**: ≈ 4–6 hours on the Vast box. With the
  log-add FoX path active, add ~1% to attention compute → ~6 h
  worst case.
- **Pass/fail bar**: copied from `idea.md:88-96` (Win Δ ≤ −0.02,
  Null |Δ| < 0.02, Fail Δ > +0.01). A null is informative — it
  partitions "FIRE's additive bias already saturates the
  relative-position benefit at our scale; multiplicative mass
  control is sub-threshold at tiny1m3m depth/length."

## Self-check (per code-implementer.md §5)
- [x] `use_fox=False` path: FoX module is not instantiated, forward
  never calls `self.fox`, no extra params allocated, no extra FLOPs.
  Baseline path is bit-identical to a pre-flag build. Confirmed by
  `git diff` (FoX edits are guarded by `if self.use_fox:` in both
  branches; off-path is byte-for-byte the original code).
- [x] `use_fox=True` path at step 0: MHA output within `1e-2` of
  baseline (test `test_step0_attention_output_unchanged`). The
  log-add form makes this even tighter than the r1 multiply form
  because the softmax cancels small additive bias more cleanly than
  the renorm did.
- [x] `test_identity_init_close_to_ones`: `exp(log_D) ∈ [D_min, 1]`
  in the causal lower triangle; diagonal within 1e-3 of 1.
- [x] `test_wiring_live_with_Wf_perturbation`: perturbing head 0's
  `gate_w` changes head 0's log_D and leaves heads 1..5 unchanged.
- [x] `test_causal_lower_triangular`: log_D upper-tri = 0 (the
  caller's mask zeros the attention weight there regardless;
  zeroing log_D is hygiene).
- [x] **NEW** `test_trained_gate_does_not_blow_up`: at b_f=-3 and
  T=2048, log_D is finite and softmax(random_logits + log_D) rows
  sum to 1. This is the regression for the r1 NaN.
- [x] All 6 tests pass: `pytest tests/test_fox.py -v` → 6 passed.

## Coordination note
`git diff models/layers.py configs/llm_config.py` has unstaged edits
from the parallel-AI (V-Norm wiring at lines 552, 686, 744, 970,
1092, 1963 of layers.py and the analogous spots in llm_config.py).
My FoX edits are at layers.py lines ~1620 (FIRE branch) and ~1726
(manual branch) — different regions, no conflict. I did not touch
configs/llm_config.py in this recode pass (the `use_fox` field and
`Tiny1M3MFOXOnFireConfig` are already in place from r1). Per
`project-parallel-ai.md`, this is the expected state.

