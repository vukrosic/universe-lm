# Muon optimizer ablations — research plan

**For the implementing AI.** Self-contained. This folder ablates the **Muon
optimizer** — which parts of it are load-bearing, and which can be cut for speed
without losing quality. The matching codebase: [optimizers/muon.py](../../../optimizers/muon.py)
and the routing in [training/trainer.py](../../../training/trainer.py) `setup_muon_optimizer` (L74).

---

## The thing we're poking

Current Muon update, per 2D param:

```text
buf  = lerp(buf, grad, 1-momentum)                     # momentum (β=0.95)
g    = lerp(grad, buf, momentum) if nesterov else buf  # nesterov
g    = zeropower_polar_express(g, steps=5)             # ORTHOGONALIZE (5 iters)
p   -= lr · max(1, fanout/fanin)^0.5 · g               # shape-scaled step
```

Routing today (`setup_muon_optimizer`):
`param.ndim==2 and 'token_embedding' not in name and 'norm' not in name → Muon`,
everything else → AdamW. Defaults: `muon_lr=0.024, momentum=0.95, ns_steps=5`,
`adamw_lr=0.006`.

The five sub-parts to ablate: **orthogonalization**, **the shape-scaling**, **momentum/
nesterov**, **routing**, **compute/precision**.

---

## ⚠️ Fairness rule (the optimizer analog of identity-init)

These levers are **not** step-0-identical to the baseline — they change the update
rule, so they change the whole trajectory. An optimizer change can look worse purely
because the **LR is now mistuned**, not because the mechanism is bad. So:

- For any lever that changes the *effective step size* (M2 no-ortho, M4 no-scale,
  M3 different coeffs), **sweep `muon_lr`** (≥3 points) and report the *best* LR per
  arm. Comparing at the baseline LR alone is invalid.
- For levers that don't touch step size (M1 fewer steps, M9 lazy ortho, precision),
  baseline LR is fine — but note it.
- Everything is **3-seed** at the chosen LR before any claim.

---

## Implementation contract

- Edit [optimizers/muon.py](../../../optimizers/muon.py) (add a flag/arg per lever)
  and/or `setup_muon_optimizer` for routing levers.
- Expose each knob on the config (`muon_ns_steps`, `muon_orthogonalize`,
  `muon_shape_scale`, `muon_momentum`, etc.) so a
  `class Screen10M20M<Name>Config(Screen10M20MConfig)` flips it.
- Run: `python train_llm.py --config <name> --seed 42`. There are already
  `--muon_lr` / `--adamw_lr` CLI overrides — use them for the LR sweeps.
- Control = clean `Screen10M20MConfig` default Muon → **4.7984** (`s_ctrl_full`).

## Protocol

tiny (`Tiny1M3M*`) to rank cheaply → screen 3-seed (42/43/44) at the per-arm best LR.
"Live" = 3-seed mean beats control by ≥0.01, seeds don't straddle zero. Several of
these are **speed** levers — also record `total_time_minutes` from `metrics.json`; a
wash on loss that's 20% faster is still a win for a compute-constrained lab.

---

## Batch 1 — orthogonalization (Muon's defining feature)

| # | Name | Change | LR re-sweep? | Why |
|---|---|---|---|---|
| M1 | `NSStepsSweep` | `ns_steps ∈ {1,2,3,5}` polar-express iters | no | how many iterations actually needed? fewer = faster. cheapest speed win |
| M2 | `NoOrtho` | skip `zeropower_polar_express` entirely (= momentum SGD on 2D) | **yes** | the headline A/B: is orthogonalization what makes Muon work, or just the LR/momentum? |
| M3 | `NSCoeffs` | swap polar-express coeffs for classic NS quintic `(3.4445,-4.7750,2.0315)` | **yes** | does the specific coeff schedule matter, or any contraction works? |
| M4 | `OrthoDtypeFp32` | run the iteration in fp32 not bf16 | no | accuracy vs speed of the orthogonalization itself |

## Batch 2 — the shape-scaling step factor

| # | Name | Change | LR re-sweep? | Why |
|---|---|---|---|---|
| M5 | `NoShapeScale` | drop `max(1, fanout/fanin)^0.5`, use flat lr | **yes** | is the aspect-ratio scaling load-bearing, or a constant LR works? |
| M6 | `SpectralScale` | modded-nanogpt scale `0.2·√max(dims)` instead | **yes** | a different principled update-norm target |
| M7 | `RMSMatchScale` | scale update so its RMS matches AdamW's | **yes** | makes Muon/AdamW steps commensurate — fairer LR coupling |

## Batch 3 — momentum / nesterov

| # | Name | Change | LR re-sweep? |
|---|---|---|---|
| M8 | `MomentumSweep` | `momentum ∈ {0.9, 0.95, 0.99}` | no (light) |
| M9 | `NesterovOff` | `nesterov=False` (plain heavy-ball) | no |

## Batch 4 — routing (which params Muon owns)

| # | Name | Change | Why |
|---|---|---|---|
| M10 | `EmbedToMuon` | route `token_embedding` / `emb_proj` to Muon (currently AdamW) | the embedding is 91% of params — does it want orthogonalized updates? |
| M11 | `PerGroupMuonLR` | separate `muon_lr` for attention vs FFN matrices | do the two matrix families want different step sizes? |
| M12 | `LRRatioSweep` | sweep `muon_lr / adamw_lr` ratio | the most basic knob — is the 0.024/0.006 split tuned? |

## Batch 5 — compute savings (speed levers, expect loss-neutral)

| # | Name | Change | Why |
|---|---|---|---|
| M13 | `LazyOrtho` | orthogonalize every N steps, reuse/skip between | cut the polar-express cost ~N× if loss holds |
| M14 | `Bf16Buffer` | momentum buffer in bf16 not fp32 | memory/speed; check it doesn't hurt at depth-24 |

---

## Run guidance

Run order: **M1 (ns_steps) and M2 (no-ortho) first** — M2 answers "does Muon even
need the orthogonalization here?", the single most informative result; M1 is the
easiest speed win. Then M12 (LR ratio, the cheapest tuning) and M5 (shape-scale).
Batch 5 only if you want to bank compute. Track wall-clock everywhere.

## When a batch finishes

1. Numbers → [tutorial/results.md](tutorial/results.md): 3-seed mean + std **and**
   `total_time_minutes`, at the per-arm best LR.
2. Status → [tutorial/experiments.md](tutorial/experiments.md).
3. Clear story → draft [tutorial/README.md](tutorial/README.md), house style of
   [../../tutorials/qk_gain/README.md](../../tutorials/qk_gain/README.md).
4. Commit `metrics.json`, re-run `runs/make_evidence_index.py`.
