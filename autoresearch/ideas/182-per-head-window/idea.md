---
id: 182-per-head-window
status: reviewing
round: 1
updated: 2026-06-15T05:50:58Z
transfer-risk: med
plain: Give each attention head its own learnable "how far back should I look" window, starting with every head looking at the full sequence so step-0 is byte-identical.
---

# 182 — Per-Head Learnable Attention Window (Soft Local Window Size per Head)

## Source
- Zaheer et al., "Big Bird: Transformers for Longer Sequences" (NeurIPS 2020, arXiv:2007.14062). BigBird uses windowed local + global + random attention patterns. Per-head pattern was explored in the paper as an ablation.
- Beltagy, Peters, Cohan, "Longformer: The Long-Document Transformer" (arXiv:2004.05150, 2020). Per-layer attention window with sliding-window of size 512.
- In-repo context: closed.md line "SWA window sweep (256/384/512/768/1024/2048) — 512 winner" closed the global SWA window as a hyperparameter sweep. The closed axis is "fixed global window per model." **Per-head learnable window** is a different lever — each head picks its own window size as a function of its gradient signal, not a fixed model-wide HP. The closest in-repo analog is 143-shortconv (pre-attention depthwise conv — a soft locality prior, different mechanism). 174-xpos-decay null tested learnable decay (different shape).

## Mechanism
Standard attention: position t attends to all positions s ≤ t.

With per-head learnable window:
- For each head h, learn a scalar `w_h ∈ ℝ` that parameterizes a window size.
- Map to a window size: `W_h = T · sigmoid(w_h)` where `T` is the sequence length.
- Apply a soft mask: for query position t, attention score to position s is reduced by `−∞ · (1 − M_h(t, s))` where `M_h(t, s) = 1` if `|t − s| ≤ W_h/2`, else `M_h(t, s) = 0`. (Causal mask still applies.)
- Equivalently, in soft form: add `−∞ · (1 − sigmoid((W_h/2 − |t − s|) / τ))` to scores outside the window, where τ is a temperature (small = hard mask, large = soft mask).

Bit-identity at step 0: parameterize `w_h` such that `W_h = T` at init. Use `w_h_init = large_positive` (e.g., 10) so `sigmoid(10) ≈ 1` and `W_h ≈ T`. Then for any query position t and any s ≤ t, `|t − s| ≤ T/2` is always true (since |t − s| ≤ T − 1 < T/2 for T ≥ 2), so the window mask is all-ones ⇒ **no mask applied ⇒ byte-identical to baseline at step 0**.

Alternatively, parameterize via `β_h = sigmoid(w_h)` directly and use `W_h = T · β_h` with `β_h_init ≈ 1` ⇒ `W_h ≈ T`.

A second lever axis (orthogonal): **per-head learnable decay rate** (different from per-head bias): instead of a hard window, decay the attention score as a Gaussian: `score[b, h, t, s] -= λ_h · (t − s)^2` where `λ_h` is a per-head decay scalar. At `λ_h = 0` (init), the decay is zero ⇒ byte-identical. This is a **soft locality prior**, related to ALiBi but with quadratic (not linear) distance decay.

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_per_head_window: bool = False` to `MultiHeadAttention.__init__`. Allocate `self.head_window_logit = nn.Parameter(torch.full((n_heads,), 10.0))` (init 10 ⇒ sigmoid ≈ 0.99995 ⇒ W ≈ T). After computing scores, compute per-head window: `W_h = T * sigmoid(self.head_window_logit)`. Apply soft mask: `score[b, h, t, s] -= 1e9 * relu((|t − s| − W_h/2))` for s ≤ t (causal); positions outside the window get a very negative bias ⇒ softmax puts ~0 probability there. With `W_h ≈ T` at init, the relu is 0 for all valid (t, s) pairs ⇒ **byte-identical**.
  - `configs/llm_config.py` — add `use_per_head_window: bool = False`. Add `Tiny1M3MPerHeadWindowConfig` subclass with `use_per_head_window: bool = True`.
  - `models/llm.py` — thread into both `TransformerBlock` sites.
- **Config flag**: `use_per_head_window: bool = False`.
- **Step-0 byte-identical**: at init, `W_h ≈ T` for all h ⇒ no positions are masked out ⇒ softmax is unchanged ⇒ **byte-identical (max-abs-diff = 0.0)** at fp32 (since `relu(|t − s| − T/2) = 0` for all s ≤ t with s ∈ [0, T)).
- **Param count**: H=4, n_layers=12. Per block: 4 window params. Total: 48 params (+0.005% of 0.94M).
- **Intuition (why it might lower val loss)**: the closed SWA sweep showed 512 wins over 1024 and 2048 at 0.94M. This means SOME local attention is helpful. The natural extension: let each head learn its own window size — some heads want a small window (local features), some want a large window (long-range dependencies). The closed SWA test fixed the window globally; 182 lets the per-head gradient find the right mix. Different from per-head scalar levers (152, 155, 160, 166) because it changes the **attention pattern shape** (which positions are attended to), not just the score distribution.

## Scale evidence
- BigBird at 100M-300M (encoder models).
- Longformer at 100M+ (encoder models, sliding window).
- Per-head window (this filing) is novel at 100M+ — most window-attention papers use a global window. Transfer-risk is **med** (windowed attention is well-validated at 100M+; per-head learnable window is novel but the underlying locality-prior is well-established).

## Why it's worth a slot
The bet, in one sharp sentence: **the closed SWA window sweep found 512 wins globally but didn't test per-head window specialization, and 182 lets the optimizer distribute local-vs-global attention across heads — a different lever from the closed per-head scalars (152, 155, 160, 166) which only affect score magnitudes, not the spatial pattern of which positions are attended to.** A null at 0.94M would close the per-head-window axis and confirm that 512 is the right window globally (no per-head benefit); a win would unlock the per-head window family for Phase-2 ≥135M where the per-head gradient signal is larger and the optimizer can find a richer mix of local/global heads.
