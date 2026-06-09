## r1 — 2026-06-09 — verdict: accept
- **Leverage**: small but real — expected Δval ≈ 0.005–0.02 at zero compute/arch cost. Sigmoid loss + z-loss is a known drop-in. The bet is the *shape of the loss*, not a model change; the risk is the gain is below noise at tiny1m3m (0.94M params).
- **Info value**: a clean null still teaches us softmax-CE is already in its basin on this data — strictly better than another optimizer ablation. A win is a win; a null closes a lever.
- **Non-obviousness**: not novel in the literature (Apple 2023), but **untried at this scale in this project** — that's the bar, not "no one has ever tried it".
- **Portfolio fit**: ✅ diversifies. Active queue is optimizer-heavy (4 optimizer ideas in `needs-run`); this is the *first* loss-shape ablation. Accept on portfolio grounds alone.
- **Niche fit**: loss-head-only, identity/zero-init safe (no weight init change), runs at tiny1m3m, transferable across scale. Clean.
- **Crisp bet**: "per-token sigmoid + z-loss beats softmax-CE on val loss at tiny1m3m because bounded per-position gradients stop logit-magnitude drift". Sharp enough.
- **Minor flag (not a gate-block)**: arXiv ID is fuzzy in the source; the definition gate can lock the canonical cite. Not a taste issue.
- Verdict: **accept** → `needs-review`, round reset to 1 for the definition gate.
