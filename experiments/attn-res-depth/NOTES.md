# AttnRes — Attention Residuals as a Depth Lever

**Status:** design (not yet implemented)
**Type:** ladder experiment (manual / donor-crowdsourced) — NOT a daemon tiny1m3m idea
**Owner:** vukrosic
**Goal it serves:** token-efficiency win at the 135M release scale → the SmolLM2-135M race.

---

## One-line hypothesis

Replacing the standard unit-weight residual with **softmax attention over depth**
(AttnRes) is a **depth lever**: Δ vs baseline is ~0 at shallow depth and grows with
layers. If Δ climbs across the ladder, it extrapolates to a real win at the 30-layer
release target.

## Source

Attention Residuals (AttnRes) — Kimi Team / MoonshotAI, arXiv:2603.15031 (Mar 2026).
Validated only at 48B / 1.4T tokens. **No small-scale results in the paper** — that
gap is exactly what this experiment fills.

- Official: https://github.com/MoonshotAI/Attention-Residuals
- Unofficial single-file (GQA+SwiGLU+RoPE): https://github.com/kyegomez/attn_res

## Mechanism

Baseline residual:
```
x_l = x_{l-1} + f_l(x_{l-1})            # unit-weight accumulation
```

AttnRes — each layer l has ONE learned pseudo-query w_l ∈ ℝ^d:
```
v_i   = output of layer i               # values = prior layer outputs
k_i   = RMSNorm(v_i)                    # keys: normalize magnitude across layers
α_i→l = softmax_i( w_l · k_i )          # weights over depth i = 0..l-1
h_l   = Σ_i  α_i→l · v_i                # aggregate
```

- **Params added:** per layer = 1 RMSNorm + 1 vector w_l ∈ ℝ^d. Negligible (~L·d).
- **Init:** `w_l = 0` → uniform softmax → starts as an **equal-weight average of all
  prior layers** (DenseFormer-like). NOT bit-identical to baseline at step 0.
- RMSNorm on keys is load-bearing: stops large-magnitude deep layers dominating softmax.

**Use FULL AttnRes (O(L·d)), skip Block AttnRes.** Block exists only to tame 48B-model
memory; at L ≤ 30 we store every layer output trivially. Halves the wiring.

Official Block pseudocode (for reference only — we do the Full version):
```python
def block_attn_res(blocks, partial_block, proj, norm):
    V = torch.stack(blocks + [partial_block])               # [N+1, B, T, D]
    K = norm(V)                                             # RMSNorm keys
    logits = einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)  # w_l · k_i
    h = einsum('n b t, n b t d -> b t d', logits.softmax(0), V)       # depth-softmax · values
    return h
```

## Why it should pay off more with depth

AttnRes fixes depth-wise hidden-state norm growth and layer-dilution under PreNorm.
The pathology it cures only appears once the stack is deep, so:

- tiny (~12 layers) → small, possibly sub-band Δ
- 135M target (30 layers) → where it should shine

This is the whole reason it's run as a **ladder**, not a single screen.

## The experiment: Δ vs depth across the ladder

Paired A/B at each rung, seed 42, control = AttnRes off (`LadderNxConfig`),
treatment = AttnRes on (`AttnResLadderNxConfig`).

| Rung | n_layers | non-embed N | control / treatment config | role |
|---|---|---|---|---|
| 8M  | **8**  | 1.45M | `Ladder8M155MConfig` / `AttnResLadder8M155MConfig`   | depth baseline |
| 13M | **8**  | 3.17M | `Ladder13M252MConfig` / `AttnResLadder13M252MConfig` | width-control (depth fixed vs 8M) |
| 23M | **15** | 10.9M | `Ladder23M469MConfig` / `AttnResLadder23M469MConfig` | depth +7 |
| 52M | **21** | 33.2M | `Ladder52M1042MConfig` / `AttnResLadder52M1042MConfig` | depth +6 |
| (target) | 30 | 106.8M | `Full135M2700MConfig` | extrapolation point |

**The ladder varies depth 8→8→15→21→30.** Two readings fall out:
- **8M → 13M** (depth FIXED at 8, width grows): Δ should stay ~flat if AttnRes is
  genuinely a depth lever and not a width/param effect. This is the control axis.
- **8M → 23M → 52M** (depth grows): Δ should grow. This is the win signal.

Plot Δ (treatment − control) vs n_layers. **The trend is the result**, not any
single point. A flat-or-noisy 8M point is expected — judge the slope.

## Success / kill criteria

- **Win:** Δ < 0 (lower loss) AND |Δ| grows monotonically with depth → extrapolate a
  real win at 30 layers → carry to the 135M release recipe.
- **Marginal:** Δ < 0 but flat across rungs → helps a constant amount, weak carry. Keep
  as optional lever, don't prioritize.
- **Kill:** Δ ≈ 0 or positive at 8M AND no upward trend by 13M/23M → doesn't carry to
  our scale. Drop it.

Detection band at 8M is ~0.02 (screen band). Expect the 8M point to be the noisiest /
weakest — judge on the trend.

## Variance / comparison discipline

- Same-box paired runs only. Absolute val_loss is NOT portable across donor GPUs;
  **only the within-box Δ is.** (See champion.json war stories re: cross-box false wins.)
- One seed (42), per house rule. Variance comes from the two-arm bracket, not seed sweeps.

## Implementation — DONE (branch `experiment/attn-res-v1`)

1. ✅ `use_attn_res: bool = False` added to base `LLMConfig` (`configs/llm_config.py`).
2. ✅ Full AttnRes wired in `models/llm.py`: per-layer pseudo-query
   `self.attn_res_query` (shape `(n_layers, d_model)`, zero-init, routes to Muon);
   parameter-free RMSNorm keys; depth-softmax aggregate replaces the inter-layer
   residual input inside `_run_post_embed`. Mutually exclusive with
   unet-skips / hyper-connections / GAU / YOCO / layer-tying (raises).
3. ✅ Treatment configs `AttnResLadder{8M,13M,23M,52M}Config` (subclass each rung,
   flip the flag). Control = the plain `Ladder*Config`.
4. ✅ Smoke verified at tiny1m3m: baseline + treatment both build, finite loss,
   `+768` params = `n_layers×d_model`, pseudo-query gets finite grads, optimizer
   builds, a real step updates the query. Baseline path (flag off) untouched.

**Run mechanism (no new runner needed):** the existing `train_llm.py
--config_class <path> --seed 42` trains and reports val loss + writes
`metrics.json`. Donor runs the control config then the treatment config; Δ =
treatment − control. See the branch README for the exact commands + the
paste-to-your-AI prompt.

## Distribution (donor workflow)

- Develop on a branch → review → merge to **main** → tag the merge commit
  `experiment/attn-res-v1` for provenance. **Donors always run main**; the tag is for us.
- One GitHub Issue per rung. Issue body = paste-into-your-LLM runbook pointing at
  `experiments/attn-res-depth/RUN.md`. Donor: clone main → install → download data slice
  → `run_donation.py --rung X` → paste `results.json` as an issue comment.
- Each donor takes a different rung → the Δ-vs-depth ladder is crowdsourced.

## Open questions

- Does the uniform-average init hurt early training at small depth? (watch first ~500 steps)
- Per-layer w_l vs per-head pseudo-queries — start per-layer (paper default).
- Is AttnRes redundant with existing V-embed / U-Net value-skip? Check in step 1.
