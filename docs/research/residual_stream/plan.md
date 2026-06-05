# Residual-stream ablations — research plan

**For the implementing AI.** Self-contained. Every experiment here is **one cheap
scalar/vector knob on the residual add** — no extra matmuls, no new modules, fast to
compute. Implement in order, record numbers, fill the [tutorial scaffold](tutorial/).

---

## The one point we're poking

The block adds each sublayer back into the residual stream:

```text
pre-norm (default), models/layers.py ~L919:
  x = x + dropout(attn_out)     # the attention residual add
  x = x + dropout(ff_out)       # the FFN residual add
```

Every experiment changes **only** how the branch enters that add:

```text
baseline:   x = x + f(x)
general:    x = a·x + g ⊙ f(x)        # a, g are the knobs; baseline is a=1, g=1
```

That's it. Keep each lever to a scalar or a length-`d_model` / length-`n_heads`
vector. If an idea needs a matmul, it doesn't belong in this folder.

## Already in the repo (do NOT reimplement — use as reference + comparison)

| Existing flag | What | Where |
|---|---|---|
| `use_layerscale` | per-channel `(1+λ)` on each branch, `λ=0` init | block forward, L906/L914/L924 |
| `use_embed_residual` | `x = m0·x + m1·x0` per-dim mix with token embedding | block forward, L884 |
| `use_post_norm` | norm after the add | L900 |
| `use_parallel_block` | attn+ffn share one normed input, summed | L887 |

Copy the `use_layerscale` wiring pattern for every new flag below.

---

## Implementation contract

- Edit the `TransformerBlock` forward in [models/layers.py](../../../models/layers.py)
  (pre-norm branch ~L919; mirror into the post-norm/parallel branches only if trivial).
- One `class Screen10M20M<Name>Config(Screen10M20MConfig)` per lever in
  [configs/llm_config.py](../../../configs/llm_config.py), flipping one flag.
- Run: `python train_llm.py --config <name> --seed 42`.
- **Identity/zero-init rule:** step-0 logits must equal the clean baseline. Zero-init
  the new param (or init a gate to pass-through) so the A/B isolates the mechanism.
  Where a lever *can't* be identity-init, it's flagged — give it its own control.
- **Optimizer routing:** all of these are per-block scalars/vectors → AdamW. Confirm
  the new params actually get gradient (a zero-init scalar can silently stay zero).

## Protocol (what counts)

- Control = clean `Screen10M20MConfig` → val_loss **4.7984** (run dir `s_ctrl_full`).
- tiny (`Tiny1M3M*`, ~2 min) to rank → screen 3-seed (42/43/44) to claim.
- A lever is "live" only if the 3-seed mean beats control by **≥0.01** and seeds
  don't straddle zero. Screen winners re-run on the full ladder before any claim.

---

## Batch 1 — the cheap core (run all, identity-init, ~0 params)

| # | Name | Mechanism | Spec (step-0 == base) | Params/block |
|---|---|---|---|---|
| R1 | `ReZero` | per-sublayer scalar gate | `x = x + α·f(x)`, one `α` per sublayer, **α=0 init** | 2 |
| R2 | `ResidMix` | learned in/out mix | `x = a·x + b·f(x)`, scalars **a=1, b=1 init** | 4 |
| R3 | `HighwayGate` | sigmoid branch gate | `x = x + σ(β)·f(x)`, **β init large so σ(β)≈1** (~baseline) | 2 |
| R4 | `BranchGainHead` | per-head attn-branch gain | scale attention output per head before the add, **g=1 init** | n_heads |

## Batch 2 — init / schedule tricks (cheap, some shift the baseline)

| # | Name | Mechanism | Spec | step-0==base |
|---|---|---|---|---|
| R5 | `DepthScaledInit` | DeepNorm-style | multiply each branch by constant `1/√(2·n_layers)` (not learned) | no — own control |
| R6 | `FrozenLayerScale` | constant small scale | non-learned scalar `0.5` on each branch | no — own control |
| R7 | `StochDepth` | stochastic depth | drop a sublayer with prob `p≈0.1` at train; identity at eval | yes (eval) |
| R8 | `BranchDropout` | branch-only dropout sweep | dropout on `f(x)` before the add, sweep `p ∈ {0.05,0.1}` | yes (eval) |

## Batch 3 — tiny variants (only if Batch 1 shows the axis is alive)

| # | Name | Mechanism | Note |
|---|---|---|---|
| R9 | `ReZeroPerChannel` | ReZero but length-`d_model` `α` (the per-channel cousin) | direct A/B vs existing `use_layerscale` |
| R10 | `ResidMixVector` | `a, b` as length-`d_model` vectors instead of scalars | richer `ResidMix` |

**A/Bs worth recording:** R1 `ReZero` (scalar) vs existing `use_layerscale`
(per-channel) — does the residual want a scalar or a vector gate? And R5/R6
(fixed init scaling) vs R1/R2 (learned) — is the win in the *init* or the *learning*?

---

## When a batch finishes

1. Numbers → [tutorial/results.md](tutorial/results.md) (control + each lever, 3-seed mean + std).
2. Status → [tutorial/experiments.md](tutorial/experiments.md).
3. Clear story → draft [tutorial/README.md](tutorial/README.md) in the house style of
   [../../tutorials/qk_gain/README.md](../../tutorials/qk_gain/README.md).
4. Commit `metrics.json`, re-run `runs/make_evidence_index.py`.
