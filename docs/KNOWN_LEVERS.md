# Known Levers — the finite backlog of proven wins

The space of things you *could* try is infinite. The space of things that **reliably
help** in nanoGPT-speedrun / Chinchilla-style training is small and finite — about
thirty. This file is that list.

**Work it top-down.** Exhaust the proven levers (high prior, cheap to validate) before
chasing speculative architecture ideas (the infinite tail). Half of these are already
in the model; the `open` rows are the near-term research queue.

```text
FINITE  (this file)  = ~37 proven levers → screen, then full, then leaderboard
INFINITE (later)     = novel architectures (JEPA, SSMs, …) → only after the finite set is banked
```

Every `open` row is a contributor on-ramp: **claim it, screen it, and if it beats the
`10m` record (5.015) open a PR.** See [EXPERIMENT_WORKFLOW.md](EXPERIMENT_WORKFLOW.md)
for the branch/tag/leaderboard mechanics and [LEADERBOARD.md](../LEADERBOARD.md) for the
standing record.

---

## How to use this list

1. Pick an `open` lever (start at the top — highest prior).
2. **Screen it** — `screen3m` (9s) for sign, `screen10m` (~4m) to confirm the sign survives.
   Screen tells you the *direction*, not the exact value (the optimum drifts with tokens).
3. **Full-validate survivors** — `--config 10m --seed 42`. This is the only number that counts.
4. Beat 5.015 by ≥0.01 → PR → the lever moves to `done` and merges to `main`.

Change **one lever per run** so you can attribute the result — unless two are known-independent
(e.g. LR schedule + QK-gain init), which you may stack.

---

## The list

Status: ✅ `have` (already in the model) · 🔲 `open` (unexploited — claim it) · ⚠️ `partial/untuned`

### Architecture & components

| # | Lever | What it is | Status | Notes / where |
|---|---|---|---|---|
| 1 | Muon optimizer | Momentum orthogonalized by Newton–Schulz / Polar-Express | ✅ have | `optimizers/muon.py` |
| 2 | RoPE | Rotary positional embeddings | ✅ have | `models/layers.py` |
| 3 | QK-norm | RMSNorm on Q and K before attention | ✅ have | `models/layers.py` |
| 4 | Squared-ReLU MLP | Primer-style relu² feed-forward | ✅ have | `models/components.py` |
| 5 | Pre-norm RMSNorm | RMSNorm, pre-norm placement | ✅ have | `models/layers.py` |
| 6 | Weight tying | Share embedding ↔ lm_head weights | ✅ have | `models/llm.py` |
| 7 | Chinchilla ~20 tok/param | Compute-optimal token budget | ✅ have | `Full10M200M` config |
| 8 | **Logit softcap** | `tanh` cap on output logits for stability (Gemma-style) | 🔲 open | small change in `models/llm.py` |
| 9 | **Value residual / value embeddings** | Learnable mix of value across depth (lambda) | 🔲 open | modded-nanoGPT trick |
| 10 | **Zero-init output projections** | Init attn-out & MLP-out to 0 (identity start) | 🔲 open | currently std=0.02 init |
| 11 | **SwiGLU MLP** | Gated-linear-unit FFN as an alternative to relu² | ⚠️ alt | bigger change; only if relu² plateaus |
| 12 | **Attention output gate** | Per-head zero-init gate on attn output | 🔲 open | similar to zero-init output projections |
| 13 | **Embedding residual** | Re-inject token embedding x0 each block | 🔲 open | pass-through skip every block |
| 14 | **U-Net skips** | Encoder/decoder symmetric layer skips | 🔲 open | complex — lower prior |
| 15 | **Dual-lane parallel residuals** | Separate attn/MLP residual lanes in deep layers | 🔲 open | med complexity |
| 16 | **LayerScale** | Learnable per-dim scale on each residual branch | 🔲 open | small init γ → 1 |
| 17 | **MLP depth vs width ratio** | Tune d_ff / d_model ratio (currently 4×) | ⚠️ untuned | architectural tuning |

### Optimization & training schedule

| # | Lever | What it is | Status | Notes / where |
|---|---|---|---|---|
| 18 | **LR warmup + decay-to-zero** | Trapezoidal/warmdown schedule instead of constant | 🔲 open | config is `constant`, `warmup_ratio=0` — highest-prior open win |
| 19 | **Per-group LR** | Higher LR for embeddings / scalar params | 🔲 open | needs param-group split in trainer |
| 20 | **Tuned LR for this shape** | `muon_lr`/`adamw_lr` re-tuned for 10M (not inherited from 88M) | 🔲 open | sweep after #18 lands |
| 21 | **Lion optimizer** | Like AdamW but only 2 momentum terms (memory efficient) | 🔲 open | ~33% less memory than Muon for optimizer state |
| 22 | **Sophia optimizer** | Second-order clipping on gradient statistics | 🔲 open | not well-tested at small scale |
| 23 | **Weight EMA** | Exponential moving average of weights for eval/shipping | 🔲 open | modest compute cost; stable win |
| 24 | **Gradient clipping** | Clip grads by global norm (1.0 is standard) | 🔲 open | usually already 1.0 but worth sweeping |
| 25 | **Cosine vs linear decay** | Switch LR schedule from linear to cosine with restarts | 🔲 open | small change in trainer |
| 26 | **Warmup ratio tuning** | Adjust `warmup_ratio` from 0 to 0.01–0.05 | 🔲 open | cheap sweep |
| 27 | **AdamW betas tuning** | Default (0.9, 0.95) vs (0.95, 0.99) or similar | ⚠️ untuned | trivial to try |
| 28 | **Gradient accumulation steps** | Tune effective batch size via accumulation | ⚠️ untuned | changes effective batch; interacts with LR |
| 29 | **Weight decay tuning** | Re-tune from the 0.2 default | ⚠️ untuned | low effort |

### Data & training dynamics

| # | Lever | What it is | Status | Notes / where |
|---|---|---|---|---|
| 30 | **SmearGate** | Blend previous token embedding into current (cheap bigram injection) | 🔲 open | shrinks at scale — screen at 10M first |
| 31 | **Curriculum learning** | Start on short sequences, grow to full length | 🔲 open | seq_len as curriculum variable |
| 32 | **Bigram injection** | Add shallow bigram signal into early layers | 🔲 open | cheap to screen |
| 33 | **Dropout** | Residual dropout rate (currently 0 = off) | 🔲 open | small models benefit from 0.01–0.1 dropout |
| 34 | **Attention dropout** | Drop attention weights during training | 🔲 open | different from normal dropout |
| 35 | **Mixed precision (BF16)** | Train in BF16 vs FP32 for speed + stability | 🔲 open | usually speedup; stability varies |
| 36 | **Weight perturbation** | Add noise to weights during training (SWA-style) | 🔲 open | regularization effect |
| 37 | **Embedding non-negativity** | Constrain token embeddings to non-negative | ⚠️ speculative | changes the embedding space geometry |

**Banked: 7. Open queue: 8–37.** That's the whole near-term roadmap.

---

## Recommended first run

Stack the two highest-prior, known-independent open levers in one full run:

- **#18** LR warmup (~0.02) + decay-to-zero, plus
- **QK-gain init ≈ 2–4** (screens already show positive init beats the `init=0` baseline)

Highest probability of a real sub-5.015 result on the first full run.

---

## Beyond the finite set

Once items 8–37 are validated, you've earned the speculative tail: new attention variants,
state-space / SSM blocks, JEPA-style objectives, depth/width reshaping, data curricula.
These are real research — lower prior, higher variance. Screen them hard before spending a
full run, and the same rule holds: **no leaderboard row without a commit hash.**