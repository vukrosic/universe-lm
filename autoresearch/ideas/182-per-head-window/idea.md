---
id: 182-per-head-window
status: rejected
round: 1
updated: 2026-06-15T07:05:54Z
transfer-risk: med
plain: Give each attention head its own learnable "how far back should I look" window, starting with every head looking at the full sequence so step-0 is byte-identical.
---

# 182 — Per-Head Learnable Attention Window (Soft Local Window Size per Head)

## Source
- Zaheer et al., "Big Bird: Transformers for Longer Sequences" (NeurIPS 2020, arXiv:2007.14062). BigBird uses windowed local + global + random attention patterns. Per-head pattern was explored in the paper as an ablation.
- Beltagy, Peters, Cohan, "Longformer: The Long-Document Transformer" (arXiv:2004.05150, 2020). Per-layer attention window with sliding-window of size 512.
- In-repo context: closed.md line "SWA window sweep (256/384/512/768/1024/2048) — 512 winner" closed the global SWA window as a hyperparameter sweep. The closed axis is "fixed global window per model." **Per-head learnable window** is a different lever — each head picks its own window size as a function of its gradient signal, not a fixed model-wide HP. The closest in-repo analog is 143-shortconv (pre-attention depthwise conv — a soft locality prior, different mechanism). 174-xpos-decay null tested learnable decay (different shape).

## Mechanism (committed sub-lever: hard window only)

Standard attention: position t attends to all positions s ≤ t.

With per-head learnable window (hard-window variant — the only sub-lever for this run):
- For each head h, learn a scalar `w_h ∈ ℝ` that parameterizes a window size.
- Map to a window size: **`W_h = 2T · sigmoid(w_h)`** (note the `2T`, not `T` — see step-0 init math below). Equivalently the mask checks `|t − s| ≤ W_h/2 = T · sigmoid(w_h)`.
- Apply a hard-style mask: for query position t, the score to position s is reduced by **`1e9 · relu(|t − s| − W_h/2)`** when s ≤ t (causal), where `M_h(t, s) = 1` if `|t − s| ≤ W_h/2`, else `0`. Outside the window the `1e9` penalty effectively zeroes the softmax probability for that key (fp32-clean; no `−∞`); inside the window, the `relu` is 0 and the score is unchanged.

**Step-0 byte-identical (FIXED — r1 reviewer caught a math error):** the r1 draft used `W_h = T · sigmoid(w_h)` and `|t − s| ≤ W_h/2`. At init with `w_h_init = 10`, `sigmoid(10) ≈ 0.99995`, so `W_h/2 ≈ T/2 = 1024` at T=2048 — but `max|t − s| = T − 1 = 2047 > 1024`, so the mask zeroed the first ~1024 positions of every query and step-0 was NOT byte-identical. The corrected parameterization:

- `W_h = 2T · sigmoid(w_h)` (the `2T` is the one-line fix), so `W_h/2 = T · sigmoid(w_h)`.
- At init, `w_h_init = 10` ⇒ `sigmoid(10) ≈ 0.99995` ⇒ `W_h/2 ≈ T · 0.99995 ≈ T − 0.00005·T`.
- For T=2048, `W_h/2 ≈ 2047.9` and `max|t − s| = T − 1 = 2047 < 2047.9`.
- Therefore every valid (t, s) pair with s ≤ t satisfies `|t − s| ≤ W_h/2`, the mask is all-ones, `relu(|t − s| − W_h/2) = 0` everywhere, the score is unchanged, softmax is unchanged ⇒ **byte-identical (max-abs-diff = 0.0) at fp32**.

The `β_h = sigmoid(w_h)` alternative (with `W_h = T · β_h`) is dropped — it has the same off-by-one (`W_h ≈ T < T`, so `max|t−s| = T−1 > T − 0.00005·T ≈ W_h`, mask is NOT all-ones). Specifying only one formulation removes any implementer choice that could pick the broken version.

> **Other sub-lever (deferred, NOT shipped in this run):** a soft Gaussian decay `score[b,h,t,s] -= λ_h·(t−s)^2` is a different mechanism (different gradient dynamics, different prior — closer to ALiBi but quadratic, not linear) and is its own future idea if hard-window is interesting. Per the r1 taste verdict, hard-window and soft-decay are NOT combined into one config; the implementer ships hard-window only.

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_per_head_window: bool = False` to `MultiHeadAttention.__init__`. Allocate `self.head_window_logit = nn.Parameter(torch.full((n_heads,), 10.0))` (init 10 ⇒ sigmoid ≈ 0.99995 ⇒ `W_h/2 = T · sigmoid(w_h) ≈ T − 0.00005·T > T − 1` ⇒ mask all-ones ⇒ byte-identical). After computing scores and applying the causal mask, compute the per-head half-window: `half_w = T * sigmoid(self.head_window_logit)` (shape `[H]`, broadcast to `[B, H, 1, 1]`). Apply the per-head window penalty: **`score -= 1e9 * relu(rel_dist - half_w)`** where `rel_dist = |t − s|`. Use `1e9` (matches 154-rebased-attn's rebased-softmax style — fp32-clean, no `−∞`, no NaN risk; softmax of `score − 1e9 ≈ 0` is well-behaved).
  - `configs/llm_config.py` — add `use_per_head_window: bool = False`. Add `Tiny1M3MPerHeadWindowConfig` subclass with `use_per_head_window: bool = True`.
  - `models/llm.py` — thread into both `TransformerBlock` sites.
- **Config flag**: `use_per_head_window: bool = False`.
- **Step-0 byte-identical (re-verify in implementer self-check):** at init, `T · sigmoid(10) ≈ T − 0.00005·T`, and `max|t − s| = T − 1 < T − 0.00005·T` for T ≥ 2, so the mask is all-ones ⇒ softmax is unchanged ⇒ `max_abs_diff(logits, baseline_logits) < 1e-6` at fp32 (must be confirmed by the implementer's self-check, mirroring 154-rebased-attn's step-0 identity test).
- **Param count**: H=4, n_layers=12. Per block: 4 window params. Total: 48 params (+0.005% of 0.94M).
- **Intuition (why it might lower val loss)**: the closed SWA sweep showed 512 wins over 1024 and 2048 at 0.94M. This means SOME local attention is helpful. The natural extension: let each head learn its own window size — some heads want a small window (local features), some want a large window (long-range dependencies). The closed SWA test fixed the window globally; 182 lets the per-head gradient find the right mix. Different from per-head scalar levers (152, 155, 160, 166) because it changes the **attention pattern shape** (which positions are attended to), not just the score distribution. Like 154-rebased-attn (WIN, Δ-3.48) and 143-shortconv (borderline, 4/4 same-day ctrls beaten), 182 acts on the **spatial pattern** of attention, not on score magnitudes — same mechanism-shape as those wins.

## Scale evidence
- BigBird at 100M-300M (encoder models).
- Longformer at 100M+ (encoder models, sliding window).
- Per-head window (this filing) is novel at 100M+ — most window-attention papers use a global window. Transfer-risk is **med** (windowed attention is well-validated at 100M+; per-head learnable window is novel but the underlying locality-prior is well-established).

## Why it's worth a slot
The bet, in one sharp sentence: **the closed SWA window sweep found 512 wins globally but didn't test per-head window specialization, and 182 lets the optimizer distribute local-vs-global attention across heads — a different lever from the closed per-head scalars (152, 155, 160, 166) which only affect score magnitudes, not the spatial pattern of which positions are attended to.** A null at 0.94M would close the per-head-window axis and confirm that 512 is the right window globally (no per-head benefit); a win would unlock the per-head window family for Phase-2 ≥135M where the per-head gradient signal is larger and the optimizer can find a richer mix of local/global heads.

## Pass / fail bar (per r1 review — must be in `plan.md`)
- **NULL band (per-head scalar null cluster):** `|trt − cached_baseline| < 0.01`. Mirrors the four closed per-head scalars (152/155/160/166) which all nulled inside this band at tiny1m3m.
- **WIN pass bar (plan-side):** `trt ≤ cached_baseline − 0.01`. Matches the magnitude of 016-qk_norm (Δ=−0.0138 at the same tier) — a real per-head attention-shape win at tiny1m3m is plausibly of that order.
- **WIN cache rule (cache-authoritative):** `trt < cached_val_mean − noise_band`. Per `BASELINE-CACHE-DESIGN.md`, this is the only verdict-bearing test. **noise_band = max(0.04, 2·val_std)**. As of the r1 self-pull on 2026-06-15: `cached_val_mean = 6.3988`, `val_std = 0.0088`, `noise_band = 0.04` ⇒ WIN iff `trt < 6.3588`. **Re-pull from `autoresearch/baseline-cache.json` on run day** — the cache has moved multiple times (6.4394 → 6.4504 → 6.4447 → 6.4346 → 6.4455 → 6.3988 across the last week); do NOT lock to the r1-draft number 6.4320.
- **Two-ctrl rule (when running live, not cached):** the WIN must also be strictly less than BOTH same-session ctrls (per the §2 two-ctrl rule used by 143-shortconv / 131-layer-drop). If `trt` beats the cached mean but lands inside the ctrl pair, that is **DRIFT**, not WIN.
- **Plan-mirror:** the implementer must put these four numbers (NULL band, plan pass bar, cached `val_mean`, `noise_band`) verbatim into `plan.md`'s pass/fail section, with the run-day re-pull instruction. If on run day the cache has been refreshed, the numbers in `plan.md` are the source of truth — `evidence.md` cites whichever version of the cache was current.

## Distinct from closed per-head axes (defensive, per r1 review)
The closed levers 152/155/160/166/172 are per-head *scalars* acting on attention *magnitudes* (additive bias, temperature, post-AV gain, additive RPE, RoPE base) — all nulled at 0.94M/12L/4H because per-head gradient signal is too weak to specialize. 154-rebased-attn (WIN, Δ-3.48) and 143-shortconv (borderline, 4/4 same-day ctrls beaten) both show that *spatial-pattern* changes (which positions are attended to) DO bind at this tier, with sharper gradient signal than scalar levers. 182 acts on the spatial pattern (which positions are inside the window), not the magnitudes — closer in mechanism-shape to 154/143 than to 152/155/160. The closed SWA-axis line ("SWA window sweep (256/384/512/768/1024/2048) — 512 winner") swept a fixed global window HP, not a per-head learnable window; 182 is not a duplicate. The 174-xpos-decay null tested a *learnable scalar decay* (not a window), so it's a different mechanism from 182's hard window. **Distinct, salvageable.**
