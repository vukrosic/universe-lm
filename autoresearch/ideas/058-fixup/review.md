## r1 — 2026-06-11 — verdict: approve

- **Source is real, current-relative, and the cited file:line hold up.**
  arXiv:1901.09321 (Zhang, Dauphin, Ma) is the canonical Fixup paper. The
  flag is at `configs/llm_config.py:44-47` and the post-init zero pass is at
  `models/llm.py:531-538` — I read both. The O-slice is correct
  (`qkvo_proj[qkv_size:]` matches `MHA.qkv_size = q + 2*kv`, and the
  qkvo_proj layout is verified at `models/layers.py:681, 1930-1931`).
  `block.feed_forward.down_proj` is the right FFN output. No fabrication.
- **Mechanism is structural, not HP.** Zero-init of two *named* weights per
  block, with the explicit property "step-0 ≡ identity" — this is the
  canonical in-bounds example of the protocol's "step-0 ≈ baseline
  (identity/zero-init)" clause. Not a std tweak, not a coefficient sweep.
- **Tiny1m3m scope correct, no multi-tier creep.** Plan will run at
  tiny1m3m only. No mention of screen20m / 135M / 1.3B — the plan stage
  inherits that constraint.
- **Not a closed lever.** Distinct from `norm-zoo` (swaps the normalizer;
  closed at `closed.md:24`), 017-sub-ln-sandwich (adds sub-norms, null at
  6L, closed at `closed.md:40`), and 019-dyt (operational duplicate of
  `squash`, closed at `closed.md:41`). 058 keeps RMSNorm untouched and
  changes only the *init state* of branch outputs — different family. The
  taste r2 paragraph on init-stability vs normalization-operation is the
  right framing; the loop-crowding concern is real but mitigated.
- **< 200 LoC trivially.** The lever is already implemented (4be65bb
  "residual-stream levers"). Zero new LoC. Plan just flips the flag.
- **Falsifiable, but plan must set the number.** The bet is binary
  falsifiable (does step-0 identity help at 6L, yes/no, against a real
  ctrl) but no numeric bar is in the idea. The plan stage must set it.
- **Transfer-risk `med` is honest.** Mechanism's *property* (init-time
  identity) is depth-invariant; its *value* at 135M is unproven. `med`
  is the right tag — not too generous, not too tight.

### Findings for the planner (the four things the plan must nail)

1. **Numeric pass/fail bar.** The plan must set a concrete Δ threshold
   against an *identical-config* control with `zero_init_resid=False`
   and seed 42. Suggest anchoring to the 016-qk-norm WIN bar
   (`-0.005`) and a `null-band` of `±0.01` (box noise). A small positive
   win is the expected outcome; ship if `trt ≤ ctrl - 0.005`, otherwise
   null.
2. **Two-ctrl variance bracket.** Per repo protocol (one seed only),
   variance is bounded by the gap between *two* baseline ctrls. The
   plan must run two ctrls (both `zero_init_resid=False`, otherwise
   identical) and the trt, and the WIN bar must be evaluated relative
   to `min(ctrl1, ctrl2)` AND must exceed the ctrl1↔ctrl2 gap.
   (Otherwise a "win" can be inside baseline variance and the slot
   burns compute for noise.)
3. **Verify both projections are zero-initialized, not just one.** The
   lever is *O-proj* (`qkvo_proj[qkv_size:]`) **and** *FFN down-proj*
   (`feed_forward.down_proj.weight`). Plan must add a one-line
   post-build probe — `assert (qkv[qkv_size:] == 0).all() and
   (ffn.down_proj.weight == 0).all()` — to catch any future refactor
   that drops the `getattr(config, "zero_init_resid", False)` branch.
   This is the same hygiene pattern as the gated-attn assertion
   adjacent to the O-zero pass at `models/llm.py:544-548`.
4. **Step-0 ≡ baseline check (optional, but cheap).** Log the
   trt-vs-ctrl Δ at step 0 (or first 50 steps) — it should be ~0 (the
   whole point of the lever is identical start). If Δ at step 0 is
   non-trivial, something is wrong with the wiring and the run should
   be killed before the 92-step loop burns.

### Verdict

Approve. Sharp bet, zero LoC, retires dead code, distinct from the
closed norm-axis family, falsifiable at the binary level. Resetting
`round` to 1 so the plan stage gets a fresh budget.

---
