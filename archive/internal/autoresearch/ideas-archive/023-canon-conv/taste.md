## r1 — 2026-06-10 — verdict: accept

- **Leverage (medium-high).** A single causal depthwise Conv1d per sublayer is the local-mixing half of the Griffin/Mamba hybrid playbook. At 0.94M params, the model has very few attention heads and limited capacity to model local n-gram structure; explicit cheap local mixing plausibly offloads short-range work and frees heads for longer-range. Distinct from the closed SWA/dilated-attn levers (those reshape the attention window) and from the closed NSA/diff-attn/hybrid axes (those are still inside attention). This adds an *orthogonal* mixer on the residual stream.
- **Information value (clean A/B).** Zero-init gate guarantees step-0 ≡ baseline, so a null is informative ("local mixing is redundant with RoPE+attention at 6L"); a win is a transferable structural lever. Both outcomes are worth logging — not a vibes test.
- **Niche fit.** Mechanism (not HP), zero-init-able, ~45 LoC, identity at init, fits the tiny1m3m budget. Clean drop-in. Kernel size 3-4 is the sweet spot — not a hyperparameter sweep.
- **Portfolio fit.** Active queue has 020 (per-head forgetting decay), 021 (V residual shortcut), 022 (rectified softmax). 023 is in the same "structural add to the residual path" neighborhood but not a duplicate of any of them — it's the local-mixing half of a hybrid, the others are attention-side or normalization tweaks. Not a 5th-of-its-family situation.
- **Crisp bet.** "Local conv offloads short-range n-gram work from few-head attention → frees heads for longer-range → val drop." One sentence, falsifiable.
- **Taste concerns (for the reviser, not a gate).**
  - Placement is slightly under-specified: "just before each attention/FFN sublayer" means 2 convs per block. Worth pinning to a single location per block (e.g. one conv on the residual stream before the attention sublayer only) to keep the A/B tight; otherwise conv-on-residual vs conv-before-attn vs conv-before-FFN becomes a hidden second axis.
  - Kernel size 3 vs 4 should be a single plan-time choice, not swept.
  - This is not a substitute for the reviser's job — just flagging that the spec needs a placement call before the code loop.

This earns a slot. The bet is real, the A/B is clean, the null is informative, and the lever is distinct from anything in the closed list or active queue.
