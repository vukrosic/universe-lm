# Review log — 022 softpick-attention

## r2 — 2026-06-10 — verdict: approve

- **All 8 r1 findings applied.** Definition block now present with
  Ctrl/trt (`Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True` vs same
  + `use_softpick=True`), numerical pass bar (WIN −0.005 / NULL ±0.01 / FAIL
  +0.01), seed 42, ε=1e-6 pinned, fp32 for `exp−1` pinned, swap site named
  (`models/layers.py:1421`), step-0 smoke + mask-interaction test specced.
- **Swap site verified.** `models/layers.py:1421` is indeed the
  `torch.softmax(scores, dim=-1)` call inside the `if self.use_fire_pe:`
  branch (the manual attention path); the OR-list at `layers.py:1435-1445`
  is the elif that fires only when FIRE is off. Since ctrl + trt both have
  `use_fire_pe=True`, the swap-line is the only softmax call reached in the
  A/B. The defensive OR-list entry (item d in the LoC budget) is
  belt-and-braces — fine.
- **Mask interaction spec is correct in substance and the test is the right
  shape** (zero contribution to both numerator and denominator on masked
  positions). The middle prose paragraph is a little muddled (the
  `masked_fill(−1e9) … after subtracting 1` sentence reads as describing the
  bug, not the fix), but both valid forms are explicitly stated — (i) set
  masked scores to 0 before `exp − 1`, or (ii) multiply numerator and
  denominator by a 0/1 mask after `exp − 1` — and the assertion test
  unambiguously pins the required behavior. Implementer can work from this.
- **Not a duplicate.** `attn-sink` in closed.md is the *additive* lever
  (added a learnable sink token to absorb wasted probability mass); softpick
  is the *normalization-function swap* (rectifies the softmax so a head can
  legally emit zero total mass, with no sink token). Different mechanisms,
  different LoC sites, different failure modes. 020-forgetting-attn keeps
  softmax and multiplies post-softmax by a causal decay; orthogonal.
  `sigmoid-loss` (closed) is an output-layer loss, not an attention
  normalizer. New axis confirmed.
- **Scope-clean.** tiny1m3m only, seed 42, no multi-seed sweep, no tier
  promotion, no HP grid. LoC ~45 ≤ 50 cap (well under the 200 ceiling).
- **Step-0 caveat properly handled.** Spec explicitly states the A/B
  asymmetry (trt step-0 attention output = 0, not pass-through-V) and gates
  the run on a smoke test that fails fast if Q/K/V grads vanish. This is
  the right place to catch a dead-on-arrival lever before burning the box.
- **Reset round to 1 for the code gate.**

## r1 — 2026-06-10 — verdict: revise

- **Source verified.** arXiv:2504.20966 resolves: "Softpick: No Attention Sink,
  No Massive Activations with Rectified Softmax", Zuhri/Fuadi/Aji, 29 Apr 2025.
  Mechanism (`relu(exp(x)−1) / (Σ|exp(x)−1| + ε)`) matches the paper. Not a
  fabrication. Distinct from `closed.md` `attn-sink` (which *added* a sink
  token) and 020-forgetting-attn (multiplicative post-softmax decay, softmax
  stays). Categorically a softmax-function swap — new axis.
- **Definition section missing — blocks the code gate.** No
  `## Definition (gate 2)` block with ctrl/trt/pass-bar/seed/LoC budget — see
  020-forgetting-attn/idea.md:76-115 for the required shape. Add a full
  Definition with ### Ctrl vs trt, ### Pass bar, ### Seed, ### LoC budget, ###
  Step-0 smoke check (called out below), ### Mask interaction (called out
  below).
- **Pin ctrl/trt to the FIRE-equipped baseline** (mirrors 020's choice; FIRE is
  the 009 WIN from `closed.md:43`). Ctrl =
  `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`
  (`configs/llm_config.py:773`). Trt = ctrl + `use_softpick=True`. Stacking on
  FIRE tests whether sink-removal is orthogonal to the best-known additive
  positional bias.
- **Pin a numerical pass bar.** "Val-loss improvement or at least equal" is not
  falsifiable. Use the same ±0.01 box-noise band as 020 (closed.md:33-40 ctrl
  spread 6.3875–6.4050): **WIN** `trt < ctrl − 0.005`, **NULL**
  `|trt − ctrl| < 0.01`, **FAIL** `trt > ctrl + 0.01`. Pick the −0.005 WIN bar
  (not −0.02) because the bet is "small but real" and we don't want a real
  effect lost in the noise floor — but state it explicitly so the runner
  doesn't have to guess.
- **🔴 Mask interaction is unspecified and is a likely correctness bug.**
  Standard softmax handles masking by setting masked scores to `−∞` →
  `exp(−∞)=0` → zero attention. Softpick under the same regime:
  `exp(−∞)−1 = −1`, then `|−1| = 1` *adds to the denominator*, so masked
  positions silently pollute the normalizer; and `relu(−1) = 0` for the
  numerator, which is fine — but the denominator contamination is wrong. Spec
  must say: **mask before softpick by setting masked scores to a value such
  that `exp(score)−1 ≈ 0`** (i.e. `score = −1e9` is still −1 after `exp−1`;
  the cleanest fix is `score = 0` for masked positions then exclude them from
  both numerator AND denominator — i.e. multiply both by a 0/1 mask after the
  `exp−1` op). The repo uses both causal masks and SWA (window 512); both
  interact with this. Pin a single canonical mask-handling line and add an
  assertion test that masked positions contribute zero to both numerator and
  denominator.
- **Step-0 is NOT identity — must be acknowledged explicitly, with a smoke
  test.** At init, `Q,K ≈ 𝒩(0, small)` → scores ≈ 0 → `exp(0)−1 = 0` →
  numerator = 0 → output = 0 / ε = 0. This means at step-0 the attention path
  returns **zero**, not pass-through-V. The residual stream survives (the `+
  attn(x)` adds 0), but the model starts with effectively no attention. Taste
  r1 flagged this as "carry, don't block" but it must be **stated in the spec**
  as a known A/B asymmetry (treatment doesn't have the same step-0 distribution
  as control) and tested: add `(e) step-0 smoke: build the trt model, run one
  fwd+bwd, assert loss is finite and grads on Q/K/V projections are non-zero`.
  If grads vanish (because attention output is exactly zero, `∂L/∂Q = 0`), the
  lever is dead on arrival and the A/B is malformed — caught before burning
  GPU.
- **Pin ε.** Idea says "ε" generically. The paper uses `ε = 1e-6`; pin that in
  the spec to remove a hidden HP. Also pin `dtype` for the `exp−1` op
  (compute in fp32 then cast back to model dtype, otherwise overflow at
  large positive scores).
- **Pin the swap site.** `models/layers.py` has multiple attention paths
  (manual-path for FIRE at `layers.py:1394`, SDPA fast-path, FoX path for
  020). Spec must say exactly which path gets softpick: since ctrl =
  FIRE-equipped (manual-path), the swap site is the `softmax` call inside the
  manual path *only* — not all paths. Otherwise the SDPA path silently keeps
  softmax and the A/B becomes config-dependent. Name the line.
- **LoC budget breakdown.** Mirror 020's decomposition. Estimate: (a) `softpick`
  helper function (rectified exp−1, denominator sum, ε guard, mask
  multiply) ~12 LoC; (b) swap site in manual MHA path ~3 LoC; (c) flag wiring
  `use_softpick: bool = False` through MHA + TransformerBlock + LLMConfig +
  new config class `Tiny1M3MSoftpickOnFireConfig` ~10 LoC; (d) mask-handling
  assertion test ~6 LoC; (e) step-0 smoke test (loss finite, non-zero grads on
  QKV) ~10 LoC. Total ~41 LoC ≤ 50 cap.
