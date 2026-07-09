# Taste — 030-unet-skip-sigmoid

## r1 — 2026-06-10 — verdict: accept

- **Leverage is small but real at 6L.** modded-nanogpt's +1.25% speedup is at ≥100M
  params where the long-range skip (layer 0 → layer N-1) carries information that
  would otherwise pass through ~12 residual blocks. At tiny1m3m (6L) the mirrors
  are 0↔5 / 1↔4 / 2↔3 — only 3 short pairs, so the predicted effect on val loss
  is "small but non-zero", not big-if-true. Still worth a slot: a small win here
  plausibly amplifies at 135M where depth grows.
- **Prior null is confounded, not informative.** The previous attempt (`docs/youtube-architecture-ablation-log.md` §5, val +0.0003 worse) used a **raw multiplicative gate init to `torch.zeros`** — `memory/unet-skips-gate-fix.md`
  documents that a gate at exactly 0 receives almost no gradient and never turns
  on. The mechanism never actually ran; the test is a bug-experiment, not a
  mechanism A/B. The modded-nanogpt sigmoid(-1.5) fix is a ~5 LoC patch on the
  existing `unet_skip_gates` code — cheapest possible "bug-fix-becomes-lever" test.
- **High information either way.** A win adds an architectural lever
  orthogonal to the active attention/optimizer queue (FIRE, QK-Norm, V-Norm,
  Moonlight RMS, and their compositions). A clean null — gate learns, still no
  effect — definitively closes the U-Net direction at tiny1m3m. Both outcomes
  log. The 5 LoC cost makes the experiment easy to absorb even if it's null.
- **Portfolio fit is good — diversifies, not crowds.** Current queue is
  attention (009/016/020/029) and optimizer (015) plus 2 stacks (026/027). U-Net
  skips is a *residual-stream* architectural change, a different family.
- **Crisp bet.** "We expect U-Net skip with sigmoid(-1.5) init to win at
  tiny1m3m because the previous failure was a dead-gate bug, not a mechanism
  failure; sigmoid(-1.5) ≈ 0.18 gives a non-zero, bounded starting point so
  the gate can actually learn the skip magnitude."
- **Transfer-risk low and mechanism scale-invariant.** Skip connections are
  useful at any depth; the magnitude may grow with N (more layers to bridge),
  so even a small tiny1m3m win likely carries to 135M.

### Routed
→ `needs-review` (round reset to 1 for definition gate's own budget).
