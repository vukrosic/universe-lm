---
id: 005-decoupled-qkv-muon
status: done
round: 1
updated: 2026-06-09T09:36:31Z
---

# 005 — Decoupled Q/K/V for Muon routing

## Source
Modded-nanoGPT (Keller Jordan) — https://github.com/KellerJordan/modded-nanogpt — `train_gpt2.py`, the Muon path treats Q, K, V as three independent 2D matrices (not a fused `qkvo` projection) and routes them separately to the orthogonalized optimizer. The repo's modded-nanogpt record is the canonical reference. Tilde Research's X posts (2025) have also commented on the routing choice. (Repo link as source per the prompt's allowed list — no single arxiv paper.)

## Mechanism
The current `models/llm.py:469-470` uses a single fused `qkvo_proj` (4 × d_model) routed to Muon as one matrix. The orthogonalized update averages the spectral structure across Q, K, V, O — they have different natural scales (Q,K are score-side, V is content-side, O is output-side). Decoupling into 4 separate 2D matrices allows each to get its own Muon update with its own `scale_mode` and orthogonalization. ~50 LoC in `models/layers.py` and `models/llm.py`. The fused-projection baseline is byte-identical at step 0 (split + zero-init the new ones), so the A/B is clean.

## Pass / fail bar
- pass: tiny1m3m val ≤ 6.4206 (ctrl 6.4287, target Δ = −0.0081)
- fail: tiny1m3m val > 6.4287
- noise: |Δ| ≤ 0.005 — below the 2-min tiny1m3m resolution; inconclusive, not a result (single-seed rule)
- expected Δ ≈ −0.005 to −0.02 (V's spectral range may differ from Q/K's; effect likely small at tiny scale)
