# Output-head ablations — research plan

**For the implementing AI.** Self-contained. The whole model funnels through one
point: the final logits and the loss on them. Every experiment here is **one cheap
op on the logits, or one extra term on the loss** — no new matmuls in the trunk.

---

## The one point we're poking

```text
models/llm.py forward (~L300):
  x      = norm(x)
  z      = emb_proj^T @ x                 # factorized tied head
  logits = token_embedding @ z            # (..., vocab)
  # (logit_softcap #71 already applied here if enabled)
loss   = cross_entropy(logits, targets)   # training/trainer.py:232, evaluation.py:53
```

We change either `logits = OP(logits)` or `loss += aux_term`. Baseline = identity OP,
no aux term.

## Already in the repo (do NOT reimplement — reference + comparison)

| Existing | What | Where |
|---|---|---|
| `logit_softcap` (#71) | `c·tanh(logits/c)` Gemma cap | models/llm.py ~L314 |
| `output_adapter` | rank-32 additive head adapter | models/llm.py ~L310 |
| tied factorized head | `emb_proj^T` then `token_embedding^T` | models/llm.py ~L305 |

A *nonlinear* tied decode path is its own existing research plan —
[../../research-plans/tied-output-mlp/plan.md](../../research-plans/tied-output-mlp/plan.md).
Don't duplicate it; OH8 below just points there.

---

## ⚠️ Reporting rule (read before touching the loss)

The leaderboard number is **plain cross-entropy val_loss**. Any *training-only* aux
term (z-loss, label smoothing, confidence penalty) must **not** be added to the
reported val_loss — eval stays plain CE so runs are comparable. Levers that change
the *logits themselves* (temperature, per-vocab bias) ARE part of the model and DO
flow into eval CE legitimately.

| Lever type | Train loss | Eval/reported val_loss |
|---|---|---|
| aux term (OH1, OH2, OH3) | CE + aux | **plain CE only** |
| logit op (OH4, OH5) | CE on modified logits | CE on modified logits (same op) |

---

## Implementation contract

- aux terms → `training/trainer.py` (~L232) **and** mirror nothing into
  `training/evaluation.py` (keep eval CE plain). Gate with `if lambda > 0`.
- logit ops → `models/llm.py` forward, right after the softcap block.
- One `class Screen10M20M<Name>Config(Screen10M20MConfig)` per lever, one knob each.
- Run: `python train_llm.py --config <name> --seed 42`.
- **Identity-init:** λ=0 / ε=0 / τ=1 / b=0 → step-0 == baseline.
- **Optimizer routing:** τ scalar, per-vocab bias → AdamW. Confirm gradient flow.

## Protocol (what counts)

- Control = clean `Screen10M20MConfig` → **4.7984** (`s_ctrl_full`).
- tiny → screen 3-seed (42/43/44). "Live" = mean beats control by ≥0.01, seeds don't
  straddle zero. Winners re-run on the full ladder.

---

## Batch 1 — loss-side terms (0 model params, train-only)

| # | Name | Term added to train loss | Spec (step-0 == base) |
|---|---|---|---|
| OH1 | `ZLoss` | `+ λ·mean(logsumexp(logits)²)` | PaLM/Chinchilla stabilizer, **λ small (e.g. 1e-4), sweep**; pulls logit norm down |
| OH2 | `LabelSmooth` | CE with smoothing `ε` | **ε=0 == base**, sweep ε∈{0.01,0.05} |
| OH3 | `ConfPenalty` | `- β·entropy(softmax(logits))` | confidence penalty (anti-overconfidence), **β=0 == base** |

OH1 is the flagship — cheapest known stabilizer, may let LR run hotter. Remember:
all three are train-only; reported val_loss stays plain CE.

## Batch 2 — logit ops (cheap model params, flow into eval)

| # | Name | OP on logits | Spec | Params |
|---|---|---|---|---|
| OH4 | `OutputTemp` | `logits /= τ` | single learnable temperature, **τ=1 init** | 1 |
| OH5 | `VocabBias` | `logits += b_v` | per-vocab additive bias (learned unigram prior), **b=0 init** | vocab_size |

OH5 is many params but trivial compute (one add); it mostly re-learns token frequency,
a known small CE win. Worth a tied-vs-untied note: it's free-ish under the tied head.

## Batch 3 — head structure (costs params / overlaps — gated)

| # | Name | What | Note |
|---|---|---|---|
| OH6 | `LogitSoftcapSweep` | sweep the **existing** `logit_softcap` c | not new code — just run c∈{10,15,30} on this tier, 3-seed |
| OH7 | `UntieHead` | separate `lm_head` weights from the embedding | **costs params, not budget-matched** — a probe: is weight-tying load-bearing? |
| OH8 | `NonlinearDecode` | nonlinear tied decode (`z = φ(emb_proj^T x)`) | → see [tied-output-mlp plan](../../research-plans/tied-output-mlp/plan.md), don't duplicate |

---

## Run guidance

OH1 (z-loss) and OH5 (vocab bias) are the highest-prior bets; OH2/OH3 are cheap
regularizer sweeps. Tiny-screen first, promote clear movers to the 3-seed screen.
OH7 is a diagnostic, not a budget-fair claim.

## When a batch finishes

1. Numbers → [tutorial/results.md](tutorial/results.md) — **plain CE val_loss**, 3-seed mean + std.
2. Status → [tutorial/experiments.md](tutorial/experiments.md).
3. Clear story → draft [tutorial/README.md](tutorial/README.md) in the house style of
   [../../tutorials/qk_gain/README.md](../../tutorials/qk_gain/README.md).
4. Commit `metrics.json`, re-run `runs/make_evidence_index.py`.
