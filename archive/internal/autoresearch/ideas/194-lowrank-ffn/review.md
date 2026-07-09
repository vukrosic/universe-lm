# Review log — 194 lowrank-ffn

## r1 — 2026-06-15 — verdict: approve

**Definition gap (one sentence):** the taste gate's re-pitch moves the rank-r residual correction off the closed FFN axis onto W_V — a within-attention sub-block whose rank structure is untested at 0.94M, on a projection (V) that is the only single-side attention projection that *positively* binds at this tier (021-vres WIN) — and the spec now carries a clean pre-registered null that closes the entire low-rank-residual sub-block family at this tier.

**Findings:**

- **Source is real and current.** LoRA (Hu et al. arXiv:2106.09685, 2021), LLM.int8() (Frantar et al. 2022), Arora et al. on linear-algebraic word-sense structure — all real, all post-2020, citations check out.
- **Mechanism is a mechanism.** Rank-r residual correction `W_V_eff = W_V + α·W_V_A·W_V_B` with `α = sigmoid(α_raw)`, `α_raw` init = −10 ⇒ α ≈ 4.5e−5 at step 0 ⇒ W_V_eff ≈ W_V to fp32 precision. Structural change with identity/zero-init, not an LR/schedule/init-constant lever.
- **🔴 tiny1m3m only.** Plan explicitly names tiny1m3m (0.94M · 3M tok, seed 42); no screen20m / ladder / multi-tier refs anywhere. Conforms.
- **Not a duplicate of a closed lever.** Cross-checked closed.md:
  - FFN-side low-rank axis (146, 153, 157, 158, 170, 117/118/145) is the *exhausted* family this re-pitch explicitly moves *off* — W_V ≠ W_down.
  - 021-value-residual (V *cross-block* residual, WIN Δ=−0.034) is a *graph* lever (residual stream), not a *parameter* lever (W_V rank) — different axis. The V-bind at 0.94M is the *motivation* for the W_V placement, not a duplicate.
  - 184-v-carry-block (V *cross-block* re-attempt, rejected 3x) is the same graph axis at 0.94M and already failed — the new axis (parameter rank, not graph) is orthogonal.
  - 164-q-carry (Q-side cross-block, null) confirms V-bind is *not* symmetric to Q — a Q-rank pitch would be dead-on-arrival; V-rank is the live one.
  - 151-rov-gated (intra-V rotary, null) — rotary on V, not rank on W_V.
  - 176-v-pre-av-norm (pre-AV V normalization, null) — magnitude axis on V, not rank.
  - 160-rms-gain-per-head (post-AV gain, null) — magnitude axis on attention output.
  - 016-qk_norm (WIN) — pre-softmax QK magnitude; *motivates* W_V placement by ruling out Q/K.
  - 207-wo-lowrank-bottleneck (in-repo, needs-taste r1) — *same* mechanism on W_O. 194-r2 on W_V is the *complementary* axis on the other d_model×d_model attention sub-block. Not a duplicate.
  - 197-tied-wo-across-blocks, 199-spectral-attn-output — orthogonal axes (W_O sharing / W_O Lipschitz), not rank.
  - No W_V *rank* correction has been tested at 0.94M. The slot is unprobed.
- **Implementable in < 200 LoC.** Spec names `models/layers.py` (attention module) + three config flags (`use_lowrank_wv`, `wv_rank=8`, `wv_alpha_raw_init=-10.0`). Per-block W_V_A ∈ R^{d_model×r}, W_V_B ∈ R^{r×d_model}, α scalar = 2×(64·8+8·64)·12 + 12 = 12,288 params (+1.3%). Comfortably under 200 LoC.
- **Falsifiable pass/fail bar with real numbers.** Pre-registered test against cached baseline 6.4394±0.04 (noise band ±0.04 from closed.md:139 170-swiglu-ffn row, ctrl cluster):
  - If `effective_rank(W_V) < 32` (L1/∞-SV ratio or sum-of-SV²) → expect Δ < −0.005 (optimizer activates rank-r path).
  - If `effective_rank(W_V) ≥ 56` → expect null.
  - Win criterion: Δ < −0.01.
  - Tight enough for tiny1m3m noise band; not a "wide expected-Δ range".
- **Transfer-risk: med, justified.** Scale evidence section cites LoRA at 7B–65B (residual low-rank) and LLM.int8() at 7B (effective rank 30–60% of nominal on Q/K/V/O). Explicitly notes no <100M from-scratch win. Lever is well-defined (LoRA analog, init α=0), novel placement (W_V, not W_O), identity-init-able. The med tag is right — not a low (because at-scale-validated) and not a high (because no small-scale validation, but the mechanism is conservative).

**Verdict routing:** `needs-plan`. Round stays at 1 (it was reset to 1 by the taste gate's accept; definition gate opens fresh at r1).
