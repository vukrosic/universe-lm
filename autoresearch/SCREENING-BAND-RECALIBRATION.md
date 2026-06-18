# Screening Band Recalibration: 0.02 → 0.01

**Issue:** Current screen band (0.02) blinds the lab to real improvements in the 0.005–0.02 range. These improvements read NULL and never reach the paired 3-seed confirm step, wasting GPU on other candidates instead of validating real wins.

**Evidence:**
- Your leaderboard shows real improvements clustering at 0.001–0.010 (p < 0.05, 3-seed confirmed)
- Within-session 1σ is ~0.017, 2σ ~0.033 (from the 2026-06-16 noise audit)
- Your paired confirm band is ~0.018 (drift-free, very tight)
- A 0.02 screen vs 0.018 confirm band conflates two different jobs (sensitivity vs. specificity)

**Current trap:**
```
Real +0.008 improvement → screens NULL (< 0.02) → never confirmed → GPU wasted
Lucky +0.025 artifact → clears screen → goes to 3-seed confirm → confirm kills it → GPU wasted anyway
```

**The fix:**
Lower the screen band from **0.02 to 0.01**. This:
1. Catches real 0.005–0.02 improvements → they reach paired confirm
2. Paired confirm (band ~0.018, drift-free) filters out flukes
3. Good wins promote, bad ones die at confirm
4. GPU is spent validating real candidates, not chasing false positives

**Safety:** The confirm step is TIGHT (0.018 band, p < 0.05, drift-free). The screen's only job is to pass candidates upstream; confirm's job is to kill flukes. A loose screen (0.02) with a tight confirm (0.018) is backwards. Flipping it to loose confirm (0.02) with tight screen (0.01) is the right design.

**Implementation:**
- File: `queue-daemon.sh` (daemon finalize_one function)
- Line: where SCREEN_BAND is set or overridden
- Change: `SCREEN_BAND=${SCREEN_BAND:-0.01}` (was 0.02)
- Also update: `DECISIONS.jsonl` or `daemon-state.json` to document the change
- Test: run a few iterations and check that 0.008 improvements now clear the screen and go to confirm

**Expected impact:**
- More candidates reach the confirm step (noise ratio improves from "few" to "most")
- Confirm step sees real 0.005–0.02 wins, validates them (3-seed pair confirms or rejects)
- GPU allocation shifts from wide-net screening to focused confirmation
- Research velocity improves: fewer false-NULL gaps, more decisive verdicts

**Rationale for 0.01 specifically (not 0.005):**
- 0.01 = ~0.6σ within-session (safe margin, 2–3 confirmations in the null band expected)
- 0.005 = ~0.3σ (too tight for a screen; every candidate needs confirm, diminishes the gate value)
- 0.01 balances: catches real 0.005–0.02 range, still filters obvious nulls (> 0.02), pair-confirms the borderline
- Your leaderboard's submission bar is 0.005 + p < 0.01; screen at 0.01 sits right between (catches submit candidates early, doesn't over-commit)

**Next step:**
After deepnet/ablations close, implement the 0.02 → 0.01 fix and document in a commit. This is a research-quality improvement that unlocks wins currently hidden by the wide screen band.
