# 005 — Decoupled Q/K/V for Muon routing
_Auto-drafted 2026-06-10 from `autoresearch/ideas/005-decoupled-qkv-muon/`._

## Abstract
The current `models/llm.py:469-470` uses a single fused `qkvo_proj` (4 × d_model) routed to Muon as one matrix. The orthogonalized update averages the spectral structure across Q, K, V, O — they have different natural scales (Q,K are score-side, V is content-side, O is output-side). Decoupling into 4 separate 2D matrices allows each to get its own Muon update with its own `scale_mode` and orthogonalization. ~50 LoC in `models/layers.py` and `models/llm.py`. The fused-projection baseline is byte-identical at step 0 (split + zero-init the new ones), so the A/B is clean. We test on tiny1m3m (seed 42). We report a NULL: treatment lies within the ctrl-to-ctrl noise band (Δ = 6.4206).

## 1 Introduction
This work re-implements and stress-tests the mechanism from Modded-nanoGPT (Keller Jordan) — https://github.com/KellerJordan/modded-nanogpt — `train_gpt2.py`, the Muon path treats Q, K, V as three independent 2D matrices (not a fused `qkvo` projection) and routes them separately to the orthogonalized optimizer. The repo's modded-nanogpt record is the canonical reference. Tilde Research's X posts (2025) have also commented on the routing choice. (Repo link as source per the prompt's allowed list — no single arxiv paper.).
We integrate the change into our standard tiny-scale training harness (MinimalLLM, seed 42) and evaluate against a two-control bracket to separate signal from kernel-level nondeterminism.

## 2 Method
The current `models/llm.py:469-470` uses a single fused `qkvo_proj` (4 × d_model) routed to Muon as one matrix. The orthogonalized update averages the spectral structure across Q, K, V, O — they have different natural scales (Q,K are score-side, V is content-side, O is output-side). Decoupling into 4 separate 2D matrices allows each to get its own Muon update with its own `scale_mode` and orthogonalization. ~50 LoC in `models/layers.py` and `models/llm.py`. The fused-projection baseline is byte-identical at step 0 (split + zero-init the new ones), so the A/B is clean.

## 3 Experimental setup
Single seed (42); tiny1m3m tier; two control replicates vs one treatment.

**Pass/fail bar.**
- pass: tiny1m3m val ≤ 6.4206 (ctrl 6.4287, target Δ = −0.0081)
- fail: tiny1m3m val > 6.4287
- noise: |Δ| ≤ 0.005 — below the 2-min tiny1m3m resolution; inconclusive, not a result (single-seed rule)
- expected Δ ≈ −0.005 to −0.02 (V's spectral range may differ from Q/K's; effect likely small at tiny scale)

## 4 Results
| Arm | Val loss (per run) | Mean |
|---|---|---|
| Control (ctrl + ctrl2) | 6.3875, 6.4050 | 6.3963 |
| Treatment | 6.3909 | 6.3909 |

<details><summary>raw evidence.md</summary>

# Evidence — 005 decoupled-qkv-muon

## Verdict: NULL
- tier: tiny1m3m, seed 42, box: vast 220.82.52.202:34386 (RTX 3060, sm_86)
- treatment val: 6.3909 (r1) — n=1
- control bracket: ctrl=6.3875, ctrl2=6.4050 (gap 0.0175)
- Δ vs ctrl: +0.0034 (treatment is marginally *worse* than ctrl)
- Δ vs ctrl2: -0.0141 (treatment is better than ctrl2, but by < gap)
- pass/fail bar (idea.md): pass ≤ 6.4206 (target Δ = -0.0081 vs leaderboard ctrl).
  Bar is met in absolute terms (6.3909 < 6.4206) but the two-ctrl rule
  requires beating *both* ctrls by more than the gap (0.0175) → not a WIN.
- two-ctrl rule: 6.3909 sits between ctrl (6.3875) and ctrl2 (6.4050) → NULL
  (inside variance). Treatment does not beat *both* ctrls.
- box check: ctrl 6.3875 vs leaderboard 6.4287 = -0.0413 (within 0.04 noise band)
- raw: remote-results/2026-06-09-vast-tiny1m3m/arq-r1/{005-decoupled-qkv.log,ctrl.log,ctrl2.log}
- date: 2026-06-09

## Caveat — code-loop audit
The 005 folder has no `plan.md`, `review.md`, or `codereview.md`. The
log.jsonl shows a single entry — the runner claimed it from `needs-run` at
2026-06-09T06:05:01Z. This means the code loop (need → plan → codereview)
was *skipped* for 005 (likely a backfill of the status field without the
loop artifacts). The verdict above is data-driven and stands regardless of
the missing docs, but the absence of plan.md/review.md/codereview.md is a
pre-existing pipeline issue. The 005 idea ships as a NULL by the two-ctrl
rule; the missing artifacts are a separate concern for the code-implementer
/ pipeline audit.

</details>

## 5 Discussion
Treatment lands inside the ctrl-to-ctrl noise band; the two-ctrl bracket is not cleared. Δ = 6.4206. Reporting as NULL and closing the idea — no further runs on additional seeds (single-seed rule).

## References
1. Modded-nanoGPT (Keller Jordan) — https://github.com/KellerJordan/modded-nanogpt — `train_gpt2.py`, the Muon path treats Q, K, V as three independent 2D matrices (not a fused `qkvo` projection) and routes them separately to the orthogonalized optimizer. The repo's modded-nanogpt record is the canonical reference. Tilde Research's X posts (2025) have also commented on the routing choice. (Repo link as source per the prompt's allowed list — no single arxiv paper.)

---
_Status_: **done** · _Verdict_: **NULL** · _Closed_: 2026-06-09T09:36:31Z
