# Research Ideas (backlog)

Durable backlog of experiment ideas. **Notes live here; scoped, actionable work becomes a GitHub issue.** Don't bury ideas in issue comments — they're hard to find later.

**Rules:**
- Only **real research** — a new mechanism or structural change. **No hyperparameter tuning** (LR, momentum, weight-decay schedules, Newton-Schulz coefficients, etc. are out).
- Must be **transferable** across scale — test at 25M against baseline #0, confirm winners once at 135M.
- Skip anything that only wins Parameter-Golf's *different* game: test-time training, quantization (GPTQ/AWQ/QAT), novel tokenizers, hash/int embeddings. Those exploit a byte-size constraint + test-time adaptation we don't have. (Quantization may return later for edge deployment, not the speedrun.)

## Transferable ideas (mined from the Parameter-Golf record)

Each has a GitHub issue (AI-generated, idea-only until a human signs off). Confidence = likelihood it helps val loss *and* survives 25M→135M.

| Idea | What | Conf | Issue |
|---|---|---|---|
| QK-Gain | learnable per-head attention temperature | — | [#19](https://github.com/vukrosic/universe-lm/issues/19) (implemented, A/B pending) |
| Embedding residual | re-inject token embedding `x0` each block | high | [#20](https://github.com/vukrosic/universe-lm/issues/20) |
| LayerScale | learnable per-dim scale on each residual branch | high | [#21](https://github.com/vukrosic/universe-lm/issues/21) |
| Zero-init resid projections | start each block as identity | high | [#22](https://github.com/vukrosic/universe-lm/issues/22) |
| U-Net skips | encoder/decoder symmetric layer skips | med-high | [#23](https://github.com/vukrosic/universe-lm/issues/23) |
| Logit softcap | Gemma-style `tanh` cap on logits | med | [#24](https://github.com/vukrosic/universe-lm/issues/24) |
| Weight EMA | eval/ship an EMA of the weights | med-high | [#25](https://github.com/vukrosic/universe-lm/issues/25) |
| Dual-lane parallel residuals | separate attn/MLP residual lanes (deep layers) | med (complex) | [#26](https://github.com/vukrosic/universe-lm/issues/26) |
| SmearGate | blend previous token's embedding (free bigram) | med-high (shrinks at scale) | [#27](https://github.com/vukrosic/universe-lm/issues/27) |
| Attention output gate | per-head zero-init gate on attn output | med-high | [#28](https://github.com/vukrosic/universe-lm/issues/28) |

## Tested in-house

| Idea | What | Status | Note |
|---|---|---|---|
| Embedding factorization + depth | low-rank embedding (vocab×r @ r×d_model, r=48) frees ~4.7M params at the 10M budget, spent on depth 3→24 layers | **wins at 10m** (in validation, 200M run) | Crosses baseline by step 2000; −0.176 nats by step 20k; already under the 4.5486 record at ~40% of the run. **Transfer caveat:** the win comes from embedding being 91% of params at 10M; at 135M (d_model 576) embedding is only ~21%, so the gain should shrink sharply with scale. Strong 10m record, weak 135M bet. Branch `exp/emb-factor-depth`. |

## External idea inbox

- 2026-06-06: Tilde Research X post to inspect and convert into a scoped mechanism if it fits the rules: https://x.com/tilderesearch/status/2062928775613206777

## Planned (own folders, drafted, not yet issues)
- **RMSNorm micro-tweaks** → [research-plans/rmsnorm-tweaks/plan.md](research-plans/rmsnorm-tweaks/plan.md)
- **Query / W_Q micro-tweaks** → [research-plans/query-tweaks/plan.md](research-plans/query-tweaks/plan.md)
- **Tied output MLP vs. embedding residual** → [research-plans/tied-output-mlp/plan.md](research-plans/tied-output-mlp/plan.md) — structural boundary coupling (tied autoencoder MLP) vs. #20's layer-by-layer `x0` injection; head-to-head + stack test

## Explicitly excluded (objective-specific or tuning)
- Test-time training (TTT, Legal/Phased/Score-First) — adapts on the eval stream; we score frozen weights
- Quantization: GPTQ/AWQ/QAT, int6/int7/FP8, ternary/binary, GPTQ-embeddings — wins a byte-size budget we don't have
- Tokenizer / vocab changes (CaseOps, novel BPE, bigram hash embeddings) — we fix SmolLM2's tokenizer for comparability
- Long-context / efficiency attention (sliding-window, XSA, VarLen, YaRN) — we're dense at seq 2048
- N-gram token tilt — inference-time decoding hack, not a better trained model
- MLP-width / aspect-ratio changes — doesn't transfer 25M→135M (param-count/HP, not a mechanism)
- Depth recurrence — trades params for compute; changes the fixed-budget frame (revisit separately)
- Pure hyperparameter tuning (LR, Muon momentum 0.97, WD schedule, NS coefficients, init std, EMA decay value)
