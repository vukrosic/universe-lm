# Review log — 001 cautious-muon

## r1 — 2026-06-08 — verdict: approve

**Verdict: keep. Idea is sound, but the doc needs tightening before the in-progress run finishes.** (Doc fixes applied; run shipped — promoted past review to `running`.)

## What's good
- Mechanism is one line, gated by a single flag (`use_cautious_muon: bool` in `configs/llm_config.py:360`); code path is bit-identical to baseline when off. Lowest possible implementation risk in the queue.
- The "+4% lr bump to compensate" is documented in *both* the config flag's inline comment and the optimizer docstring (`optimizers/muon.py:67-74`). The mechanism and the offset aren't separated by reading distance — good.
- Run is already in-progress on the cheapest harness (tiny1m3m, T4, ~2 min). Right scope for a sanity check.

## What's missing / weak
1. **No pass/fail bar written in the idea doc itself.** The gate is buried in `002`'s text ("tiny1m3m val ≤ 6.4206") and in the queue's run log. For someone landing on `001/idea.md` cold, the success criterion is invisible. Put the bar in the idea doc: `pass = tiny1m3m val ≤ 6.4206 (ctrl 6.4287, −0.0081)`, `fail = val > 6.4287`.
2. **The "−0.01 to −0.05" expected-Δ range is too wide for a sanity check.** The actual Liang et al. result is a few-percent loss reduction; on a 2-minute tiny1m3m run you can only resolve Δ ≳ 0.005. Tighten to "expected Δ ≈ −0.005 to −0.02; anything inside ±0.005 is noise" so the verdict isn't ambiguous.
3. **The "+4% lr bump" claim is unsourced.** The paper's recommended procedure is *not* a fixed bump — it's "scale the masked step norm back to its pre-mask norm" (i.e. divide by the mask-fraction, per-step). A constant 4% bump is a project-specific choice; either cite where the 10–20% step-norm-shrink estimate comes from, or drop the bump and let the caller tune.
4. **No mention of the seed-sensitivity caveat.** Liang et al. report the largest gains at small batch sizes / short runs. tiny1m3m *is* that regime, so this is actually the right place to test — but the doc should say so explicitly, because the screen20m follow-up is *not* in that regime and may show a much smaller or null effect.
5. **No link from the idea doc to the in-progress run evidence.** Once the Kaggle run finishes, `evidence.md` lands in this folder and currently nothing in `idea.md` points to it. Add one line: `see evidence.md` after status.

## Suggestions (cheap)
- Add a 3-line "What we'd close this for" section: e.g. *no improvement at tiny1m3m* OR *unstable at screen20m despite tiny1m3m pass* OR *ortho+sign-mask interaction is bf16-noisy on the polar-express path*.
- The "mechanism" sentence is good but packs 3 facts (mask, norm-shrink, lr-bump) into one line. Split into one-line mechanism + one-line lr-rationale.

## Verdict
Ship the run. Fix the missing pass-bar and the unsourced "+4%" in the next doc pass — they cost nothing and make the evidence.md verdict unambiguous.
