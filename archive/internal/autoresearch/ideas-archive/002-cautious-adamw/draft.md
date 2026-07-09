# 002 — Cautious AdamW
_Auto-drafted 2026-06-10 from `autoresearch/ideas/002-cautious-adamw/`._

## Abstract
Same sign-mask as Cautious Muon, applied to the AdamW update for 1D parameters. The mechanistic claim from the paper is that the mask helps when the *preconditioned* update direction disagrees with the current gradient sign — that disagreement is the stale-momentum / 2nd-moment-scaling artifact. On Muon this is common (orthogonalized update is sign-agnostic by construction); on AdamW it is rarer in steady state because 2nd-moment normalization already pulls the update toward the sign of the gradient, so the mask is mostly a no-op. The gain on AdamW is therefore expected to be smaller than on Muon (Liang et al. Table 1), but the *failure mode* it targets is different — and complementary, not redundant. A null on this idea does NOT imply Muon-cautious was useless; the two paths are independent and the paper reports both as additive in their small-scale ablations. We test on tiny1m3m (seed 42). We report a NULL: treatment lies within the ctrl-to-ctrl noise band (Δ = None).

## 1 Introduction
This work re-implements and stress-tests the mechanism from Liang et al. 2024, "Cautious Optimizers" (arXiv 2411.16085). Extension of [[001-cautious-muon]] to the AdamW path (1D params: gains, scalars, embeddings, head)..
We integrate the change into our standard tiny-scale training harness (MinimalLLM, seed 42) and evaluate against a two-control bracket to separate signal from kernel-level nondeterminism.

## 2 Method
Same sign-mask as Cautious Muon, applied to the AdamW update for 1D parameters. The mechanistic claim from the paper is that the mask helps when the *preconditioned* update direction disagrees with the current gradient sign — that disagreement is the stale-momentum / 2nd-moment-scaling artifact. On Muon this is common (orthogonalized update is sign-agnostic by construction); on AdamW it is rarer in steady state because 2nd-moment normalization already pulls the update toward the sign of the gradient, so the mask is mostly a no-op. The gain on AdamW is therefore expected to be smaller than on Muon (Liang et al. Table 1), but the *failure mode* it targets is different — and complementary, not redundant. A null on this idea does NOT imply Muon-cautious was useless; the two paths are independent and the paper reports both as additive in their small-scale ablations.

## 3 Experimental setup
- Run only after [[001-cautious-muon]] passes Phase 1 (tiny1m3m val ≤ 6.4206). If 001 fails, close this idea too — same mechanism, different path, gated on first.
- **Tier:** screen20m is the only tier where this is resolvable. tiny1m3m at ~8M training tokens has noise ±0.06-0.16 (`LEADERBOARD.md` line 96-99); the expected Δ is below the noise floor there.
- **Conditions:** 2 (A first; B only if A is null/in noise). C dropped.
- **Seeds:** 1 seed (42) per condition — single seed, pipeline rule. Worst case 2 screen20m runs ≈ 40 min on the RTX 3050; happy case 1 run ≈ 20 min if A hits cleanly. A sub-noise Δ is **inconclusive**, not a confirmed effect — do not add seeds to chase it.
- **Control:** V+q+SWA+HighRoPE 4.6364 (`LEADERBOARD.md` row 18d) — same control as [[001-cautious-muon]]'s screen20m follow-up so the two A/Bs are directly comparable.
- **Expected Δ:** `−0.005 to −0.01` on screen20m (per-parameter, single seed), with `−0.02` as a stretch outcome. A null is informative, not a failure.
- **Fallback:** if 001's screen20m follow-up lands first, the cheaper move is to add `use_cautious_adamw=True` to the same config and run a 2-flag combo (`use_cautious_muon=True` + `use_cautious_adamw=True`) on the V+q+SWA+HighRoPE baseline — one run, additive answer, no fresh A/B. (The previous "001's run exercises the AdamW path with the same config" claim was wrong: 001 only flips the Muon mask; the AdamW path is untouched in 001's run.)

(Pipeline status lives in the frontmatter above.)

## 4 Results
| Arm | Val loss (per run) | Mean |
|---|---|---|
| Control (ctrl + ctrl2) | 6.4403 | 6.4403 |
| Treatment | 6.4337, -0.0066 | 3.2136 |

<details><summary>raw evidence.md</summary>

# Evidence — 002 cautious-adamw

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: vast-34386 (220.82.52.202:34386, RTX 3060, sm_86)
- control val: 6.4403
- treatment A val: 6.4406  (embedding bucket: `token_embedding` + `emb_proj`)  Δ: +0.0003
- treatment B val: 6.4337  (gain bucket: `*.norm.weight` + 1D scalars)            Δ: -0.0066
- pass/fail bar (tiny1m3m): both A (Δ +0.0003) and B (Δ −0.0066) are well inside
  run-to-run variance (~0.04 on this box) → no effect
- → NULL. tiny1m3m is the only tier; there is no larger-tier re-test.
- box check: ctrl 6.4403 vs leaderboard 6.4287 (+0.0116)
- raw: remote-results/2026-06-09-vast-tiny1m3m/results.json
- date: 2026-06-09

The 002 wiring (`use_cautious_adamw` flag + `CautiousAdamW` subclass) is in place
and bit-identical when `"none"` (default); `boxval` smoke (max diff 2.98e-08)
confirms the gate. Closed as a tiny1m3m null.

</details>

## 5 Discussion
Treatment lands inside the ctrl-to-ctrl noise band; the two-ctrl bracket is not cleared. Δ = n/a. Reporting as NULL and closing the idea — no further runs on additional seeds (single-seed rule).

## References
1. Liang et al. 2024, "Cautious Optimizers" (arXiv 2411.16085). Extension of [[001-cautious-muon]] to the AdamW path (1D params: gains, scalars, embeddings, head).

---
_Status_: **done** · _Verdict_: **NULL** · _Closed_: 2026-06-09T01:02:18Z
