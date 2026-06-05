# Attention-logit ablations — research plan

**For the implementing AI.** Self-contained. Every experiment here is **one cheap
op on the attention score matrix, right before softmax** — `scores = QKᵀ/√d` → op →
`softmax`. No new matmuls beyond the existing QK; per-head scalars/vectors only.

---

## The one point we're poking

```text
scores = (Q · Kᵀ) / √d_k          # [B, H, T, T]
scores = OP(scores)               # <-- the only thing we change
scores = scores.masked_fill(causal/SWA, -inf)
attn   = softmax(scores) @ V
```

Baseline `OP = identity`. Every lever is a per-head scalar, a per-row reduction, or
an elementwise bound on `scores`.

## Critical wiring note (read first)

The **default attention path uses fused `F.scaled_dot_product_attention` (SDPA)** —
it never exposes `scores`, so you cannot edit logits there. The repo already has an
**eager scores path** (used by ALiBi `use_alibi_bias`, talking-heads
`use_talking_heads_q`, and NSA) that computes `scores` explicitly, masks, softmaxes,
and `@V`. See [models/layers.py](../../../models/layers.py) around the manual
`torch.matmul(Q, K.transpose) / √d_k ... torch.softmax(...)` blocks (~L968).

**Do this:**
1. Add one shared boolean `use_eager_logits` that's auto-enabled whenever any logit
   flag below is set. It routes attention through a single eager
   `scores → OP(scores) → mask → softmax → @V` branch (copy the existing eager block;
   keep causal + sliding-window masks intact).
2. Each lever is then **one line of `OP`** inside that branch, guarded by its flag.
3. Two levers (A1 temperature, A6 per-head scale) can *alternatively* be done by
   scaling Q before SDPA (no eager path needed) — cheaper. Implement those that way
   if it's a clean one-liner; everything else needs the eager branch.

Several query flags (`use_alibi_bias`, `use_cosine_attn`, etc.) are already declared
in `MultiHeadAttention.__init__` — reuse that param-declaration style.

---

## Implementation contract

- One `class Screen10M20M<Name>Config(Screen10M20MConfig)` per lever, one flag each.
- Run: `python train_llm.py --config <name> --seed 42`.
- **Identity/zero-init:** step-0 logits == clean baseline. Init each op to a no-op
  (τ=1, scale=0, bias=0, cap large). Where impossible, it's flagged → own control.
- **Optimizer routing:** all per-head scalars/vectors → AdamW. Confirm gradient flow.
- **Mask discipline:** the eager branch MUST keep the exact causal + SWA masking of
  the SDPA path, or the A/B is invalid (a masking change, not a logit change).

## Protocol (what counts)

- Control = clean `Screen10M20MConfig` → val_loss **4.7984** (`s_ctrl_full`).
- tiny (`Tiny1M3M*`) to rank → screen 3-seed (42/43/44) to claim. "Live" = mean
  beats control by **≥0.01**, seeds don't straddle zero. Winners re-run on full ladder.
- **Cross-cut control:** also run a *plain* `use_eager_logits` with `OP=identity` and
  confirm it matches SDPA (4.7984). This isolates the eager-path rewrite from the
  mechanism — every logit lever is measured against the eager-identity, not just SDPA.

---

## Batch 1 — temperature / scale (cheapest, identity-init)

| # | Name | OP on scores | Spec (step-0 == base) | Params/block |
|---|---|---|---|---|
| A1 | `LogitTemp` | `scores /= τ_h` | per-head learnable temperature, **τ=1 init** | n_heads |
| A2 | `LogitScale` | `scores *= (1 + s_h)` | additive cousin of temp, **s=0 init** | n_heads |
| A3 | `LogitSoftcap` | `c·tanh(scores/c)` | per-head cap `c`, init large (≈ no-op); sweep c | n_heads |
| A4 | `LogitTanhClip` | `b·tanh(scores/b)` fixed `b` | hard bound to kill blow-up at depth-24 | 0 |

A2/A3 closely related; A3 formalizes the one closed single-seed softcap run.

## Batch 2 — additive priors on the score (identity-init)

| # | Name | OP | Spec | Params/block |
|---|---|---|---|---|
| A5 | `LogitBiasHead` | `scores += b_h` | per-head constant added to every score, **b=0 init** | n_heads |
| A6 | `SelfBias` | `scores += d·I` | bias the diagonal (self-attention) only, **d=0 init** | n_heads |
| A7 | `OffDiagPenalty` | `scores -= p·(1-I)` | constant penalty on non-self scores, **p=0 init** | n_heads |

A5 is the eager-path twin of the shipped `attn_sink` — record the head-to-head A/B.

## Batch 3 — row-wise normalization of the logits (identity-ish)

| # | Name | OP (per query row) | Spec | step-0==base |
|---|---|---|---|---|
| A8 | `LogitCenter` | `scores -= mean_j(scores)` | subtract per-row mean before softmax (softmax-shift-invariant → pure null check / numerics) | yes |
| A9 | `LogitStdNorm` | `scores /= std_j(scores)·τ` | per-row standardize then learnable τ | ~yes |

A8 should be a *no-op* by softmax shift-invariance — it's the sanity anchor that
proves the eager path and the harness are wired right. If A8 moves loss, something
is wrong with the implementation, not the model.

## Batch 4 — regularizers (no params on the model, aux loss)

| # | Name | Mechanism | step-0==base |
|---|---|---|---|
| A10 | `EntropyReg` | aux loss = `+λ·H(attn)` or `-λ·H(attn)`; sweep sign and small λ | yes (λ→0) |

Tests whether attention wants to be pushed sharper or more diffuse. Cheapest of all —
no architecture change, just a term on the loss. Run last, gated on Batches 1–3.

---

## Run guidance

Breadth screen: tiny first, promote only clear tiny movers to the 3-seed screen.
A1/A3/A5/A10 are the highest-prior bets; A8 is the wiring sanity check, run it once
early. Don't pay for all 10 at full cost.

## When a batch finishes

1. Numbers → [tutorial/results.md](tutorial/results.md) (eager-identity control + each lever, 3-seed mean + std).
2. Status → [tutorial/experiments.md](tutorial/experiments.md).
3. Clear story → draft [tutorial/README.md](tutorial/README.md) in the house style of
   [../../tutorials/qk_gain/README.md](../../tutorials/qk_gain/README.md).
4. Commit `metrics.json`, re-run `runs/make_evidence_index.py`.
