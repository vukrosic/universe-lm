# Research Ideas (backlog)

Durable backlog of experiment ideas. **Notes live here; scoped, actionable work becomes a GitHub issue.** Don't bury ideas in issue comments — they're hard to find later.

**Rules:**
- Only **real research** — a new mechanism or structural change. **No hyperparameter tuning** (LR, momentum, weight-decay schedules, Newton-Schulz coefficients, etc. are out).
- Must be **transferable** across scale — test at 25M against baseline #0, confirm winners once at 135M.
- Skip anything that only wins Parameter-Golf's *different* game: test-time training, quantization (GPTQ/AWQ/QAT), novel tokenizers, hash/int embeddings. Those exploit a byte-size constraint + test-time adaptation we don't have. (Quantization may return later for edge deployment, not the speedrun.)

## Transferable ideas (from the Parameter-Golf leaderboard)

| Idea | What | Status |
|---|---|---|
| QK-Gain | learnable scalar gain on attention logits | **scoping** (#TBD) |
| Value residual | mix layer-0 value into later layers' V | not started |
| Parallel residuals | run attention + MLP on the same input (GPT-J style) | not started |
| Attention output gate | learnable gate on attention output | not started |
| Logit softcapping | Gemma-style softcap on output logits | not started |
| Depth recurrence | loop / weight-tie layers for more depth per param | not started |

## Explicitly excluded (objective-specific or tuning)
- Test-time training (TTT)
- Quantization: GPTQ, AWQ, QAT, int6/int7 embeddings
- Tokenizer changes (CaseOps, novel tokenizers, hash embeddings) — we fix SmolLM2's tokenizer for comparability
- Pure hyperparameter tuning (LR, momentum, WD schedule, NS coefficients)
