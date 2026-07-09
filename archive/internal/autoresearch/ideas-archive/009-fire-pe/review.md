# Review log — 009 fire-pe

## r2 — 2026-06-09 — verdict: approve

**All 5 r1 findings cleanly addressed. Ready for plan.md.**

- **Pass/fail bar with named control (r1 finding) — fixed.** Now: V+q+SWA+HighRoPE tiny1m3m control **6.4287** (cited from `queue.md` Remote run log row 1). Thresholds: pass ≤ 6.4237 (Δ ≤ −0.005), fail > 6.4287, noise |Δ| ≤ 0.005. Box-noise ±0.01 cited per the new prompt. ✓
- **Expected-Δ range (r1 finding) — fixed.** "−0.005 to −0.02; |Δ| ≤ 0.005 is noise / inconclusive." Resolvable at tiny1m3m. ✓
- **Seed protocol (r1 finding) — fixed.** "tiny1m3m, seed 42 only" explicit. No multi-seed, no seed sweeps. Sub-noise = inconclusive, not real. ✓
- **Source staleness acknowledged (r1 finding) — fixed.** One-line note: "FIRE has been the leading non-RoPE relative PE since 2023; no clear successor in 2024-2025 has dethroned it." Preempts the code-gate critic. ✓
- **Length-extrapolation moved (r1 finding) — fixed.** New "Transfer argument / future-work" section explicitly says tiny1m3m cannot test length-extrap and leaves it as future-work for a longer-context tier. Bar is now train-distribution loss only. ✓

**5-check sweep (r2):** Source ✓ (real, NeurIPS 2023, widely-cited, staleness noted), Mechanism ✓ (drop-in PE, identity/zero-init safe), tiny1m3m-only ✓, Not closed ✓, < 200 LoC ✓, Falsifiable bar ✓.

**Hand-off to code-implementer:** promote to `plan.md`. Wiring is concrete: drop-in for RoPE, ~30-50 LoC for the kernel + MLP + bias-add. Φ MLP zero-init means step-0 ≈ baseline (RoPE-equivalent) — that satisfies the "step-0 ≈ baseline" rule. The `use_fire_pe: bool = False` flag should land next to the existing RoPE/position flags in `LLMConfig`.

## r1 — 2026-06-09 — verdict: revise

**Mechanism is clean, dedup is clean, scope is correct (tiny1m3m, no multi-seed) — but the doc has no falsifiable bar, no seed commit, and no expected-Δ range. All fillable in <15 min.**

**5-check sweep (new rules):**
- **Source real & current:** Li et al., "Functional Interpolation for Relative Positional Encoding" (NeurIPS 2023, arXiv:2306.02613). Plausible authors, NeurIPS venue, widely-cited. ⚠️ 2023 paper, not 2025–2026 — see finding below.
- **Mechanism is structural, not HP:** drop-in PE replacement (additive logit-bias with content-aware φ + fixed γ kernel). Identity/zero-init safe (γ is a fixed kernel; φ MLP can be zero-init to start as RoPE-equivalent). Not an LR/schedule/init-constant lever. ✓
- **🔴 tiny1m3m only:** doc says "Tier: tiny1m3m" — explicit. ✓
- **Not already closed:** `closed.md` says "RoPE base sweep — 500k winner" (closes RoPE variants, not FIRE); NoPE is also closed. FIRE is a relative PE, not NoPE, no conflict. ✓
- **< 200 LoC:** "~30-50 LoC" for the kernel + MLP + bias-add. ✓
- **Falsifiable pass/fail bar with real control at tiny1m3m:** ⚠️ doc says "we get a small val-loss win" without a control number, pass threshold, fail threshold, or noise band. Per the new prompt §2, "a wide expected-Δ range that can't be resolved at tiny1m3m (box noise ~±0.01 val loss) is a finding."

**Findings (must be addressed before `needs-plan`):**

- **No pass/fail bar with named control.** Doc says "small val-loss win" but never cites a baseline. **Fix:** name the V+q+SWA+HighRoPE tiny1m3m control (**6.4287**, per `queue.md` Remote run log row 1), and state `pass / fail / noise` thresholds against it. Example: `pass: tiny1m3m val ≤ 6.4237 (Δ ≤ −0.005)`, `fail: val > 6.4287`, `noise: |Δ| ≤ 0.005 (single-seed, tiny1m3m)`. Cite the box-noise ±0.01 (per the new prompt) so the bar makes sense.
- **No expected-Δ range.** "Small val-loss win" is not a range. **Fix:** commit to a range resolvable at tiny1m3m noise, e.g. "expected Δ −0.005 to −0.02; |Δ| ≤ 0.005 is noise / inconclusive." The taste-reviewer's "win or lose" framing is fine; the plan needs a number.
- **Seed protocol not committed.** Doc doesn't say "seed 42 only" — and the prior 003/004/007 plans slipped into "3 seeds (42/43/44)" partly because the reviser / code gate didn't see an explicit pin. **Fix:** add a one-line "Seed: 42 only" (no multi-seed, no seed sweeps). This is a hard rule of the pipeline and must be stated in the plan.
- **Source is 2023, not 2025–2026.** Doc says "strongest *non-RoPE* relative PE in the 2023-2024 literature" but cites the 2023 paper. Not a reject (the new prompt §2 says "prefer 2025–2026," not "require"), but worth one sentence acknowledging the staleness: "FIRE has been the leading non-RoPE relative PE since 2023; no clear successor in 2024-2025 has dethroned it." That preempts the code-gate critic asking the same question.
- **Length-extrapolation claim is out of scope.** Doc says "genuine length-extrapolation" as part of the upside. tiny1m3m is a 3M-token, fixed-length run — we will not see length-extrapolation at this tier. The expected-Δ in the bar is val-loss-on-the-train-distribution, not extrapolation. **Fix:** move "length-extrapolation" from the *expected outcome* to the *transfer argument / future-work* section, so the bar isn't asked to test something it can't.

**Hand-off to reviser:** 5 findings, all doc-completeness. Mechanism paragraph and "Why it's worth a slot" are good — keep them. The Bar and Tier sections are the only rewrites.

## r0 — (none)
