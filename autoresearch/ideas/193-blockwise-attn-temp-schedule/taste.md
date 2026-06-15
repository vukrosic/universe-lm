## r1 — 2026-06-15 — verdict: revise

The lever is well-anchored on 175's depth-uniform WIN and the mechanism
(no-param cosine schedule, bit-identical at α=0) is niche-fit. But the
bet as pitched has two taste gaps that have to close before this is
worth a GPU slot.

- **Sign convention is internally inconsistent.** The pitch says
  "α > 0, early blocks have τ_0 = 1 + α (cooler/sharper softmax) and
  late blocks have τ_{L-1} = 1 − α (warmer/softer softmax)." Plug in
  the math: `τ_b = 1 + α·cos(π·b/L)`. For b=0, `τ_0 = 1+α` (larger),
  so `scores/τ` shrinks, so softmax is **smoother** (cooler) — not
  sharper. For b=L-1 (L=12), `τ_{L-1} ≈ 1 − 0.966·α` (smaller, so
  **sharper**). The narrative ("cooler/sharper softmax" for early
  blocks) flips the standard τ-vs-sharpness convention. A null run
  with the current pitch is uninterpretable because the "expected
  winner" sign of α is undefined. **Fix:** pick one sign convention
  (recommend: τ < 1 → sharper, τ > 1 → softer, the standard usage),
  restate the bet as "we expect α < 0 to win because early
  sharpness (small τ_0) captures local patterns and late softness
  (large τ_{L-1}) integrates context," and commit to a single α
  value (e.g. α = −0.3) for the A/B.

- **"Explore both signs" is two ideas in a trench coat.** The design
  sketch says "for the lever we explore both signs" — but the A/B is
  one scalar value. Either (a) commit to one sign with a
  mechanistic argument and run it (preferred — fits the one-slot
  budget), or (b) split into 193a (α > 0) and 193b (α < 0) as two
  separate ideas. Option (a) is the right move; the
  sharpness-of-attention literature is mature enough that a
  direction can be picked.

- **Portfolio crowding with 188 (per-block learnable scalar).** The
  active queue has 188-qk-rms-scaling in needs-taste, which is the
  learnable counterpart on the same axis (depth-varying multiplicative
  scale on QK, init=1, bit-identical baseline). 188 subsumes 193 in
  the limit (the optimizer can in principle learn a cosine-shaped
  schedule if it has enough steps). The only information 193 adds
  over 188 is "the *fixed shape* is the right prior, optimizer can't
  find it in 92 steps" — which is informative only **conditional on
  188 nulling**. Tighten the pitch to make this explicit: "193 is
  the follow-up bet to 188 — if 188 nulls, 193 tests whether the
  schedule shape is binding and the optimizer is starved; if 188
  wins, 193 is subsumed and can be skipped."

- **Other points that pass taste cleanly:** leverage (real lever if
  it wins, depth-varying multiplicative prior is novel and not in
  the closed axes list), niche fit (mechanism-shaped, no params,
  bit-identical at α=0, runs at tiny1m3m), transfer (low-risk
  correctly tagged, fixed-function lever), info value on the
  conditional path (null closes the axis decisively), crisp one-line
  thesis (anchored on 175's WIN).

**Action for miner (round 2):** (1) resolve the sign convention to
α < 0 = "sharpen early, soften late" with the standard
τ-sharpness mapping; (2) commit to one α value and a predicted Δval
range (e.g. "we expect α = −0.3 to give Δval ≈ −0.01 to −0.05 at
tiny1m3m, building on 175's −0.1585 depth-uniform WIN"); (3)
explicitly frame 193 as the **fixed-shape follow-up to 188** so the
A/B is conditional, not redundant. After that, the lever is
accept-ready.
