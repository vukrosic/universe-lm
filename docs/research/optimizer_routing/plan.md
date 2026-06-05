# Optimizer-routing ablations — research plan

**For the implementing AI.** Self-contained. The one point we're poking is
**which params go to Muon vs AdamW, and at what LR**. This is C9 from
[../../research/README.md](../../research/README.md) §5 — routing sweep,
per-group LR, Muon for 1D params. Repo-specific. Lever to bank at:
[../../../research-plans/muon-optimizer/plan.md](../../../research-plans/muon-optimizer/plan.md) (Batch 4 routing).

---

## The one point we're poking

```text
setup_muon_optimizer in training/trainer.py:
  for name, param in model.named_parameters():
    if param.ndim == 2 and 'token_embedding' not in name and 'norm' not in name:
      muon_params.append(param)
    else:
      adamw_params.append(param)
```

Baseline is the rule above. Every lever here changes **ONE** routing decision.
`muon_lr=0.024, adamw_lr=0.006` are the LLMConfig defaults.

## Critical wiring note

The routing function (`setup_muon_optimizer`, [training/trainer.py](../../../training/trainer.py)
L74) decides group membership. Both optimizers receive **disjoint** param lists.
A param in both → double-step (the step-0 state is now wrong by the optimizer's
own update twice). A param in neither → no step (silent bug, the param appears
in the model but is frozen — only caught by loss not moving). Every lever MUST
preserve disjointness.

## Implementation contract

- Edit [training/trainer.py](../../../training/trainer.py) `setup_muon_optimizer`
  (around L74). One `getattr(config, "...", default)` boolean flag per routing
  lever — read **inside** the routing loop, fires when True.
- One `class Tiny1M3M<Name>Config(Tiny1M3MConfig)` per lever in
  [configs/optimizer_routing_ablations.py](../../../configs/optimizer_routing_ablations.py).
- Run: `python train_llm.py --config <name> --seed 42`.
- **Identity-check at step 0:** with the flag False, the routing must be
  byte-identical to the baseline rule. Verify by reading the diff — the lever
  must use `getattr(config, "...", False)` so default-False leaves the rule
  unchanged.
- **Disjointness-check:** after the change, the sum of Muon + AdamW param counts
  must equal the model's total trainable-param count. A 1-D norm param moved to
  Muon must NOT also stay in AdamW.

## Protocol (what counts)

- Control = clean `Screen10M20MConfig` (default routing) → **4.7984**
  (`s_ctrl_full`).
- Routing levers are **not** step-0-identical in trajectory — moving a param
  to a different optimizer changes the entire update path. See the
  [fairness rule](../../../research-plans/muon-optimizer/plan.md#fairness-rule):
  if a lever changes *effective step size* (it does — Muon's orthogonalized
  update has a different RMS than AdamW's), sweep `muon_lr` (≥3 points) and
  report the best LR per arm. Comparing at baseline LR alone is invalid.
- 3-seed (42/43/44) at the chosen LR. "Live" = mean beats control by ≥0.01,
  seeds don't straddle zero. Routing results transfer poorly between scales —
  re-screen on the full ladder before promoting.

## Batches (3 batches, 6 levers)

### Batch 1 — norm / gain routing (the cheapest clear hypothesis)

| # | Name | Routing change | step-0 == base | Note |
|---|---|---|---|---|
| R1 | `MuonFor1DNorm` | route 1-D `norm.weight` to Muon instead of AdamW | yes (zero grad step at init) | Orthogonalize the per-channel gain. Repo-specific. |
| R2 | `MuonForEmbed` | route `token_embedding` / `emb_proj` to Muon (currently AdamW) | yes | The embedding is most of the params at vocab=49k. Big question. Overlaps muon M10. |
| R3 | `MuonForOutput` | route `lm_head` / `out_proj` to Muon | yes | Output projection is 2-D but might be deliberately kept in AdamW. |

### Batch 2 — per-group LR

| # | Name | Routing change | step-0 == base | Note |
|---|---|---|---|---|
| R4 | `SeparateHeadLR` | separate `muon_lr` for attention vs FFN matrices | yes | A/B: do head/FFN want different step sizes? |
| R5 | `LRRatioSweep` | sweep `muon_lr / adamw_lr` ratio | yes | The basic knob. (Also in muon plan M12 — note overlap.) |

### Batch 3 — group routing rules

| # | Name | Routing change | step-0 == base | Note |
|---|---|---|---|---|
| R6 | `NoNormSkip` | drop the `'norm' not in name` rule (let 2-D norm matrices go to Muon) | yes | Simpler rule, may capture extra signal. |

## Run guidance

**R1 first** — cheapest, clearest hypothesis. **R2 second** — biggest payoff
if it works (the embedding is most of the params at vocab=49k). **R4-R5**
only if the routing axis is alive (sign > noise on R1/R2). **R3, R6** are
probes of the *current* default rule — useful negative results either way.

## When a batch finishes

1. Numbers → [tutorial/results.md](tutorial/results.md): 3-seed mean + std at
   the per-arm best LR. **Also** report param counts (`Muon: N, AdamW: M`)
   so disjointness is auditable.
2. Status → [tutorial/experiments.md](tutorial/experiments.md).
3. Clear story → draft [tutorial/README.md](tutorial/README.md), house style
   of an existing tutorial.
4. Commit `metrics.json`, re-run `runs/make_evidence_index.py`.
