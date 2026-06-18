# AttnRes Depth-Lever — Run Guide

Run a **paired 2-arm A/B** (control vs treatment), same box, seed 42, and report the
within-box delta. AttnRes is a **depth lever**: the result we care about is the *trend*
of Δ across rungs, not any single number. Pick a rung, run both arms, post your Δ.

> One-line hypothesis: replacing the unit-weight residual with softmax-attention over
> depth helps *more as the network gets deeper*. Δ ≈ 0 at shallow depth, grows with layers.

---

## 0. Setup (once)

```
git clone https://github.com/vukrosic/universe-lm && cd universe-lm
git checkout experiment/attn-res-v1
pip install -r requirements.txt
```

On consumer / non-datacenter GPUs, prefix every train command with `TORCHDYNAMO_DISABLE=1`.

## 1. Pick your rung

Each rung is a control config + its AttnRes treatment. **Δ = treatment_val − control_val.**
Negative Δ = the trick helped.

| Rung | Layers | Control config | Treatment config | Notes |
|------|--------|----------------|------------------|-------|
| 8M   | 8  | `Ladder8M155MConfig`  | `AttnResLadder8M155MConfig`  | local-runnable, noisiest — start here |
| 13M  | 8  | `Ladder13M252MConfig` | `AttnResLadder13M252MConfig` | depth fixed, width grows → Δ should stay ~flat |
| 23M  | 15 | `Ladder23M469MConfig` | `AttnResLadder23M469MConfig` | depth grows → Δ should grow |
| 52M  | 21 | `Ladder52M1042MConfig`| `AttnResLadder52M1042MConfig`| needs a real contributor GPU |

The depth story: **8M→13M** holds depth fixed (width control); **8M→23M→52M** grows
depth. If Δ climbs across the depth-growing rungs, it extrapolates to a real win at the
30-layer release target.

## 2. Run both arms (example: 8M rung)

```
# control
TORCHDYNAMO_DISABLE=1 python train_llm.py \
  --config_class configs.llm_config.Ladder8M155MConfig \
  --seed 42 --output_dir ./ckpt/ctrl

# treatment
TORCHDYNAMO_DISABLE=1 python train_llm.py \
  --config_class configs.llm_config.AttnResLadder8M155MConfig \
  --seed 42 --output_dir ./ckpt/trt
```

Swap in your rung's config classes for the other rungs.

**If you OOM on a 12GB card at the 8M rung**, drop the per-step batch and make it up with
accumulation (same effective batch):

```
... --batch_size 2 --gradient_accumulation_steps 4
```

## 3. Read the result

Each run writes `metrics.json` in its `--output_dir`. Read `final_metrics.val_loss` from
both:

```
ctrl=$(python -c "import json;print(json.load(open('ckpt/ctrl/metrics.json'))['final_metrics']['val_loss'])")
trt=$(python  -c "import json;print(json.load(open('ckpt/trt/metrics.json'))['final_metrics']['val_loss'])")
python -c "print('delta =', $trt - $ctrl)"
```

## 4. Report

Post a comment on the **Week-1 AttnRes deltas** GitHub issue with:

- GPU model
- rung you ran
- both val_loss numbers + the Δ
- whether you used the OOM batch override

Negative Δ = AttnRes helped. Each donor takes a **different rung** so we crowdsource the
full Δ-vs-depth ladder.

---

## Discipline (read before you trust a number)

- **Same-box paired runs only.** Absolute val_loss is not portable across GPUs — only the
  within-box Δ is. Never compare your control to someone else's treatment.
- **One seed (42).** Variance comes from the two-arm bracket, not seed sweeps.
- The 8M point is the weakest/noisiest (detection band ~0.02). Judge on the *trend* across
  rungs, not the single 8M Δ.

Mechanism + full write-up: `experiments/attn-res-depth/NOTES.md`.
Source: Attention Residuals, Kimi Team / MoonshotAI, arXiv:2603.15031.
