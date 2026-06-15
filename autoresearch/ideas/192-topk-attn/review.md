# Review log — 192 topk-attn

## r1 — 2026-06-15 — verdict: approve
- **Source is real.** Touvron et al. 2021 "Going Deeper with Image
  Transformers" (DeiT III, arXiv:2103.17239) — top-k attention is a real
  mechanism the paper validates at 100M-300M on ImageNet. Sparsemax
  (Martins & Astudillo 2016, arXiv:1602.02068) is the related fixed-budget
  cousin. No fabricated citation.
- **Mechanism is a mechanism, not an HP lever.** Hard-fixed top-k is a
  *structural* attention sparsification (pre-softmax per-row hard mask
  with softmax renormalization). 0 new params. Step-0 is non-identical
  (correctly framed as the 173/022/154 *structural lever* category, no
  byte-identity pretense after the r1 correction). Pass the
  mechanism-vs-HP check.
- **🔴 tiny1m3m only.** Pitch is explicit. No screen20m, no ladder, no
  multi-tier. ✓
- **Not already closed.** Cross-checked closed.md and the in-pitch
  engagement list: 173 (learned support, recode-cap), 022 (soft sparse),
  182 (windowed contiguous, null), 154 (soft locality prior, WIN), 177
  (H×H output mixing, DRIFT), 148 (gated additive). 192 is *hard
  score-sorted non-contiguous pre-softmax per-row* — none of those are
  this. 192 also explicitly engages the *competing-not-stacking*
  relationship to 154 (axis-slot) and the d_k=16 hostility of 177
  (doesn't transfer because 192 doesn't cross heads). Cleanly distinct.
- **Implementable in <200 LoC.** Plumb `use_topk_attn: bool` and
  `topk_k: int` through:
  - `configs/llm_config.py` — 2 fields on `LLMConfig` (~2 LoC) +
    `Tiny1M3MTopKAttnConfig(Tiny1M3MConfig)` subclass (~10 LoC).
  - `models/layers.py` — 2 fields on `MultiHeadAttention.__init__`
    (~2 LoC), conditional in the manual-branch softmax site at the
    `if self.use_entmax: ... else: torch.softmax(...)` swap (~6 LoC),
    and append `or self.use_topk_attn` to the manual-path-forcing
    `elif` chain at lines 4070-4090 (~1 LoC).
  - `models/llm.py` — 1-line `getattr` and 1-line plumb to the MHA
    constructor (~2 LoC).
  - Total ~25 LoC. Well under the 200-LoC budget.
- **Has a falsifiable pass/fail bar tied to a real control.** `WIN:
  trt_val ≤ ctrl_val − 0.005 AND clears the two-ctrl rule` against the
  175-alibi champion (val 6.2403±0.04 from baseline-cache). `NULL:
  |Δ| < 0.01`. `DRIFT: trt_val > ctrl_val + 0.01`. The bar is
  resolvable at the box's ±0.04 noise band. ✓
- **Transfer-risk: med — justified.** Frontmatter tag and the `## Scale
  evidence` section cite Touvron 2021 at 100M-300M vision and Sparsemax
  at 30M-100M classification. The pitch correctly flags that the
  *forced-sparsity ratio* (k/T) matters at scale — k=512/T=2048 is 75%,
  k=512/T=8192 is 94%, sweet spot may shift at 135M where T grows. The
  `k ∈ {256, 512, 1024}` 135M-stage re-test is flagged but not locked.
  The "med" tag is the right call: not "low" because the prior is
  not-validated-at-LM-scale directly, not "high" because the mechanism
  *is* scale-stable (Touvron 100-300M). ✓
- **Magnitude prediction is banded with mechanism.** Primary = NULL
  (|Δval| < 0.01), long-shot WIN in [−0.005, −0.015], DRIFT in [+0.01,
  +0.05]. Each band has a named mechanism. The DRIFT cap is correctly
  bounded at ~+0.05 because 192 doesn't cross heads (177's +0.95
  blow-up mode doesn't transfer). ✓
- **Causal-mask interaction is correctly specified.** Topk on the
  *already-masked* scores: future positions are -inf after the mask, so
  `scores.topk` never selects them. `k = min(self.topk_k, scores.size(-1))`
  guard handles shorter evaluation contexts. ✓

### Non-blocking notes for the reviser / code-implementer (not revise findings)

- The topk insertion point must be **after** the mask (so -inf positions
  are below the topk budget) and **before** the softmax — the natural
  site is the existing `if self.use_entmax: ... else: torch.softmax(...)`
  swap at line 4316, with `use_topk_attn` added as a third branch.
- Force the manual path: append `or self.use_topk_attn` to the
  manual-path-forcing `elif` chain at lines 4070-4090 (topk can't go
  through SDPA's flash kernel, same rationale as 173/022/166/etc.).
- `topk_k` is a fixed int (no parameter). No RNG consumption, no
  parameter registration. The default `k=512` is a config constant, not
  a learnable scalar.
- For the plan: at seq_len=2048 with causal mask, the first query
  (t=0) has only 1 causal key, so `min(topk_k, t+1)` would be a more
  defensive bound than `min(topk_k, scores.size(-1))`. The latter is
  safe in practice (topk of 1 element = that element) but
  explicit-causal-bound is cleaner. Note, not a revise finding.

### Verdict: approve
The lever is sound, falsifiable, distinct from all closed axes, and
implementable in ~25 LoC against the real files. Step-0 framing is
correct (structural lever category, not byte-identity). The 1-D-vs-2-D
search bet against 173 is a real, testable claim with bounded DRIFT.
The 154-axis-slot framing is the right move. Null is the primary
prediction, but a null is strictly more informative than re-running 173
— it confirms the 173 null on a *different* parameterization. Slot
earned.

→ `needs-plan`. Round reset to 1 so the code gate gets a fresh budget.
