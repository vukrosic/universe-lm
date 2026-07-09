# Taste — 176-v-pre-av-norm

## r1 — 2026-06-15 — verdict: accept

- **Niche-fit (clean).** Mechanism, not HP. Zero-init-able via the
  per-head scalar `α_h` gate (relu(0)=0), so step-0 is bit-identical
  to the baseline (verified algebraically in the idea). Lever is
  parameter-light (~12 α + 192 γ = 204 params, +0.022% of 0.94M) and
  uses the existing `nn.RMSNorm` from `models/layers.py:340,879`. Trivially
  testable at tiny1m3m.
- **Leverage — V is genuinely the missing tensor.** The "normalize the
  attention-input tensor before it interacts" family is well-supported
  by **016-qk-norm WIN** (Δ -0.0138/-0.0185, ≫ gap 0.0047). 162-q-only
  and 165-k-only nulls together tell us the QK-norm WIN was carried by
  the *joint* symmetry. V has no such symmetry partner (only V → AV),
  so V-norm is a structurally independent axis and the 162/165 nulls
  are not a counter-argument. The only adjacent V-magnitude lever is
  **160-rms-gain-per-head NULL** (post-AV per-head gain) — 176 is
  pre-AV, a different tensor location. **151-rov-gated NULL** is
  V-position-coupling, not V-magnitude. The pre-AV V-norm axis is open.
- **Information value (high either way).** A WIN at tiny1m3m extends
  the 016 family to the third attention-input tensor and unlocks
  per-head gating at Phase-2 (135M) where the gradient signal is
  stronger. A NULL is informative: it tells us V-norm alone cannot
  fire the way joint QK-norm did, which bounds the pre-interaction-
  normalization family at 0.94M. The miner's expected Δ ∈ [-0.005,
  -0.020] sits in the testable band; a null is not a wasted slot.
- **Crisp bet (yes).** "We expect Δval ∈ [-0.005, -0.020] because 016's
  pre-softmax normalization WIN extends to V, the only attention-input
  tensor not yet tested for normalization. V is structurally different
  from Q/K (no dot-product symmetry partner), so it can act alone
  where 162 and 165 could not." One sentence. Mechanism-named, scale-
  bound, null-informative.
- **Transfer (med, defensible).** The bet does not rely on a
  tiny1m3m-only artefact (vocab/embedding dominance, 3M-token
  regime). The 016 family is the closest validated analog and it
  transfers by extending the same normalization-on-attention-tensors
  thesis. If the gate-α stays pinned near 0 at 0.94M, the per-head
  gain γ is still a structural lever at 135M where the gradient is
  stronger — same null-but-recoverable pattern as 160.
- **Portfolio fit (good).** Active queue has 168-av-output-carry and
  163-v-mix-conv (V-side conv mix), but neither is V-magnitude-norm.
  169-qk-norm-depth is in the same family but the *depth* axis. The
  V-pre-AV-magnitude axis has no near-duplicate in flight.
- **One taste-side concern flagged for definition gate, not a
  block.** The 162/165 nulls show single-side pre-softmax
  normalization is hard to fire alone at 0.94M. The miner's
  counter-argument (V has no symmetry partner, so single-side is
  fine) is plausible but unverified. The definition gate should
  ensure the run controls the gate trajectory (log `α_h` at the
  end of training) — if `α` stays near 0 across all heads, that's
  the lever failing to engage, not a real null. Cheap to add to
  the runner.
- **Verdict: accept.** Sharp, high-leverage, fits the niche. Send to
  definition gate with `round=1` reset.

  - Reset round: yes (1).
  - Reason: missing-tensor extension of a WIN family, clean
    bit-identity, informative-null value, parameter-light.
