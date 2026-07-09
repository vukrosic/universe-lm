# Taste log — 009 fire-pe

## r1 — 2026-06-09 — verdict: accept
- **Leverage**: small but real — content-aware positional bias is a known win in the 2023-2024 literature (length-extrapolation in particular). Bet is sharp: pure-rotation RoPE can't express content-dependent distance; FIRE's `φ(x)` does.
- **Info value**: a clean A/B at tiny1m3m (positional bias only) teaches us something *whether it wins or loses*. Null closes a lever.
- **Non-obviousness**: not novel in the literature, but **untried at this scale in this project** — RoPE-500k was our only PE result (closed by sweep). FIRE is the strongest non-RoPE relative PE; worth a slot.
- **Portfolio fit**: ✅ diversifies. Active queue is optimizer-heavy + 1 loss-shape (007). 009 is the **first positional ablation** to clear taste. Accept on portfolio grounds.
- **Niche fit**: drop-in for RoPE, ~30-50 LoC, identity-safe (`γ` is a fixed kernel, no learnable weight init change; `φ` MLP can be zero-init). Tier `tiny1m3m` ✓.
- **Crisp bet**: "FIRE's content-aware bias beats RoPE-500k on val loss at tiny1m3m because token content can modulate effective distance". Sharp enough.
- Verdict: **accept** → `needs-review`, round reset to 1.
