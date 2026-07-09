## r2 — 2026-06-15 — verdict: accept
- **All three r1 findings are cleanly closed.** (1) Sign convention is resolved to the standard `τ < 1 ⇒ sharper softmax, τ > 1 ⇒ softer softmax`; the math walkthrough for α = −0.3 at b = 0 / b = L-1 is correct (τ_0 = 0.7, τ_11 ≈ 1.29) and the narrative ("sharpen early, soften late") matches the math. (2) Single value committed: α = −0.3, with a mechanistic argument (consistent with 175's locality-rewarding WIN, just on the multiplicative depth-varying side). (3) The 188-conditional framing is explicit and honest — 193 is the *fixed-shape* follow-up to 188's *learned* counterpart, with a clear "if 188 wins, 193 is subsumed" branch.
- **The bet sits in a genuinely unoccupied slot of the design plane.** Active-and-closed neighbors: 175-alibi (depth-uniform *additive* WIN), 188-qk-rms (depth-varying *multiplicative* *learned*), 155-per-head-temp (head-varying *learned* NULL), 161-dyt-temp (depth-varying *multiplicative* *learned* DRIFT), 169-qk-norm-depth (placement NULL). 193 is **depth-varying *multiplicative* *fixed-shape*** — a clean cell on the {additive × multiplicative} × {uniform × varying} × {fixed × learned} grid that no closed or in-flight idea occupies. 197-deepnet-α is init-time-only on a different axis (residual-stream scale, not attention score scale).
- **Crisp bet sentence (single-line check passes):** "α = −0.3 (sharpen early, soften late) gives a depth-varying multiplicative prior on attention scores; conditional on 188 NULLing at tiny1m3m, this A/B tests whether the *fixed cosine shape* is the right depth-conditional prior the optimizer couldn't find in 92 steps." That's the bar.
- **Predicted Δval range is realistic and well-anchored.** 175's −0.1585 is the depth-uniform *additive* reference; 193 is depth-varying *multiplicative* with a smaller amplitude (τ ∈ [0.7, 1.29]). The expected band [−0.01, −0.05] sits well below the 175 reference (multiplicative scale is a smaller lever than additive bias) but is still meaningful relative to the 0.04 noise floor. Conditional on 188 winning, 193 is subsumed; conditional on 188 nulling, 193 has a clean pass/fail/DNULL/DDRIFT taxonomy.
- **Bit-identity at α = 0 holds.** `τ_b = 1 + 0·cos(…) = 1` for all b, and `scores / (1 · √d_k) = scores / √d_k` exactly. SDPA flash exclusion is correctly noted. The ctrl is the same `Tiny1M3MConfig` (val 6.4216 or 6.4044/6.4091 for the bracket). No leak risk.
- **Niche fit is solid.** Mechanism (not HP), identity-init-able, no learnable params (the deliberate contrast to 188), runs at tiny1m3m, transferable to 135M (fixed-function lever — `τ_b` is a depth function, not a model-size function). transfer-risk: low is correctly tagged. ALiBi is validated 0.4B-6.7B; the depth-varying *multiplicative* analog is novel in-repo but literature-supported (cascaded / curriculum attention).
- **Portfolio fit holds.** 188 is the only direct neighbor, and 193's *fixed* schedule is the natural control to 188's *learned* schedule — they tile the fixed-vs-learned axis cleanly. The conditional framing ("run 188 first; if it nulls, 193 isolates the shape axis") is exactly the right architecture. 197 is a different axis (residual init, not attention score) — no crowding.
- **Minor concern (not blocking):** the pass/fail bar uses ctrl val 6.4216 *or* the 6.4044/6.4091 bracket — the definition gate should pin which ctrl is the reference (recommended: the active champion 6.3988 ± 0.04 from the 175-alibi WIN, since the 175 reference is the entire reason the sharpen-early sign is motivated). 016's WIN was 6.3906 (on a per-head-norm ctrl), so any ctrl pinned to the bracket must be the 175-armed champion to be a fair reference. Flag for the definition gate, not a taste issue.
- **Verdict: accept** → `needs-review`, round reset to 1 for the definition gate's own budget. The definition gate should verify: (a) the `use_block_temp_schedule: bool` and `block_temp_alpha: float` plumbing is off by default (α=0 baseline bit-identical), (b) `τ_b` is computed once at forward time or cached as a `[L]` `Buffer` (not per-step), (c) the manual-path forcing list includes `use_block_temp_schedule` so SDPA flash doesn't perturb step-0 numerics at α=0, (d) the ctrl is the 175-armed champion (val 6.3988 ± 0.04) since the 175 reference anchors the sign choice, (e) the conditional logic — "if 188 reports WIN ≥ −0.005 before 193 runs, redirect 193 to a different axis" — is preserved in the plan, (f) predicted band is Δval ∈ [−0.01, −0.05] with a 0.01 pass/fail bar (consistent with the active two-ctrl rule).

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
