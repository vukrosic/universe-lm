---
id: 175-alibi-slopes
status: running
round: 1
updated: 2026-06-15T05:07:19Z
transfer-risk: low
plain: Add learnable per-head linear-distance bias to attention scores (each head learns its own "how local is local" slope), starting at zero slope so step-0 is byte-identical.
---

# 175 — ALiBi with Learnable Per-Head Slopes (Additive Linear-Distance Bias on Attention Scores)

## Source
- Press, Smith, Lewis, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (ICLR 2022, arXiv:2108.12409). ALiBi adds a non-learned, fixed per-head slope to attention scores: `score[i,j] += −s_h · (i − j)` for j ≤ i (causal). The slopes `s_h` are *not* trainable in the original paper — they are set to a fixed geometric sequence `1/2^(8k/H)` for k=0..H-1. ALiBi was validated on BLOOM-176B (per the BigScience 2022 paper), GPT-3-style training, and many follow-ups.
- 175 makes the slopes **learnable** per head (each head learns its own slope via backprop). Init slope=0 ⇒ bias=0 ⇒ **bit-identical to baseline at step 0**.
- In-repo context: 152-attn-logit-bias closed null (per-head additive bias, init 0). 166-t5-rpe closed null (per-head additive bucketed bias, init 0). 175 is *structurally different* from both: the bias is a **function of position distance** (`-s_h · (i-j)`), not a free per-head offset. The structured inductive bias gives the lever an axis the optimizer can exploit that 152/166 do not have — namely, *how local is local* per head. At 0.94M/12L/4H, this is a fresh lever in the same locality-prior family as 009-FIRE WIN, 154-rebased-attn WIN, 023-canon-conv WIN, 143-shortconv borderline.
- Code is already in place: `models/layers.py:1776-1777` declares `self.use_alibi_bias` and `self.alibi_slope = nn.Parameter(torch.zeros(self.n_heads))` (init 0). `models/layers.py:2957-2960` applies the bias: `scores = scores - m * diff.view(...)`. **The mechanism is built but has never been tested at tiny1m3m** — no `175-alibi-slopes` idea file, no entry in `closed.md`.

## Mechanism
Standard attention: `score[i,j] = Q_i · K_j / √d_k`, then `p = softmax(score)`.
ALiBi: `score[i,j] -= s_h · (i − j)` for j ≤ i (causal, no bias on j > i).

With learnable slopes:
- `s_h = softplus(s_raw_h)` with `s_raw_h` init 0 ⇒ `s_h = ln(2) ≈ 0.693` ⇒ step-0 slope is 0.693, **NOT bit-identical**.
- `s_h = s_raw_h` with `s_raw_h` init 0 ⇒ `s_h = 0` ⇒ bias is 0 ⇒ bit-identical at step 0. ✓
- `s_h = abs(s_raw_h)` with `s_raw_h` init 0 ⇒ `s_h = 0` ⇒ bit-identical. (More robust against negative slopes.)
- `s_h = softplus(s_raw_h - 4)` (shift the softplus) ⇒ at init `s_h = softplus(-4) ≈ 0.018`, near-zero but not exactly 0.

We use the **direct linear parameterization** `s_h = s_raw_h` init 0 for cleanest step-0 identity. Negative slopes are allowed (they would amplify distant attention, the opposite of ALiBi's intent) — the optimizer will find the right sign.

A **second lever axis**: a global scale `γ` on the bias, init γ=0 ⇒ bias is 0 ⇒ bit-identical. Lets the model scale all per-head slopes uniformly. (Orthogonal to per-head slopes.)

A **third lever axis**: instead of linear distance `|i-j|`, use logarithmic bucket index `bucket(|i-j|)` (T5-style). This converges to 166-t5-rpe. Skip — already filed and closed.

## Design sketch
- **Files**:
  - `models/layers.py` — the mechanism is **already implemented** at
    lines 1776-1777 (init) and 2957-2960 (use). The implementation work
    is purely the **config wiring**:
    - `configs/llm_config.py` — add `use_alibi_bias: bool = False` on
      `LLMConfig` (default off) and a `Tiny1M3MAlibiConfig` subclass
      with `use_alibi_bias: bool = True`.
    - Verify that `use_alibi_bias` is read from config and threaded
      into the four `TransformerBlock(...)` / `MultiHeadAttention(...)`
      construction sites in `models/llm.py` (the sites that already
      thread `use_alibi_bias` per `models/layers.py:3973`).
  - `models/layers.py:1778` already does
    `self.alibi_slope = nn.Parameter(torch.zeros(self.n_heads))` —
    confirm this is the init we want (yes: init 0 ⇒ bit-identical).
- **Config flag**: `use_alibi_bias: bool = False` (default off).
- **Step-0 byte-identical**: with `alibi_slope = 0` for all heads
  (the default init for the existing parameter at line 1778), the
  bias is `-0 * (i-j) = 0` for all positions ⇒ scores unchanged ⇒
  softmax unchanged ⇒ AV unchanged ⇒ output unchanged ⇒
  **byte-identical to baseline at step 0 (max-abs-diff = 0.0)**.
- **Intuition (why it might lower val loss)**: ALiBi is a *locality
  prior*: it tells the model "nearby tokens matter more than distant
  tokens" via a linear decay. The 009-FIRE WIN and 154-rebased-attn WIN
  both win on similar locality-prior logic, so a different locality
  prior (linear distance bias instead of continuous integrable PE or
  rebased K/V) should plausibly fire too. Learnable per-head slopes
  let the model decide *how* local each head wants to be — early
  layers may want gentle decay (large slope of decay... wait, slope
  *is* the decay rate), late layers may want sharp decay. Per-head
  slope is a 4-scalar-per-block × 12 blocks = 48-scalar lever with
  rich per-head specialization room.
- **LoC**: ~10 lines of config wiring (mechanism is already in
  `models/layers.py`). Smallest implementation cost after 172.

## Scale evidence
- ALiBi validated at BLOOM-176B (BigScience, 2022). **Direct
  validation at ≥100M, the highest scale in any of our filed ideas.**
- 152-attn-logit-bias closed null at 0.94M (per-head additive bias,
  init 0). 166-t5-rpe closed null (per-head additive bucketed bias,
  init 0). Both are *content-free* per-head biases; 175 has
  *position-distance structure* that gives the lever a non-trivial
  axis the optimizer can exploit.
- Closest in-repo WIN: 009-FIRE PE (Δ -0.064/-0.082). 175 is a
  different locality-prior lever (linear-distance decay vs
  continuous integrable PE) and is well-validated at scale.
- **Transfer risk: low** (validated at 176B; mechanism is scale-free;
  per-head slopes are a tiny lever that any tier can absorb).

## Why it's worth a slot
The bet: ALiBi is a *proven* locality-prior mechanism at 176B, the
closed-axis nulls (152, 166) were both content-free per-head additive
biases, and 175's structured position-distance bias provides a fresh
axis the optimizer can exploit. We expect Δval ∈ [-0.005, -0.025]
(modest, similar to the locality-prior family). A null tells us the
locality-prior family is exhausted at this tier (alongside 152, 166)
and per-head-attention-shape axes are closed. A win unlocks the lever
at Phase-2 ≥135M where each head has more gradient signal to develop
a useful per-head slope. Smallest implementation cost of the three
filed levers (~10 LoC of config wiring only — the mechanism is built
in `models/layers.py` already).
