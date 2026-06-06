# U-Net skip ablations — Kaggle/T4 plan

**For the implementing AI.** Self-contained plan for turning the current U-Net
skip tutorial into a small, honest ablation study. The one point we are testing
is the bridge from early transformer blocks to mirrored late blocks.

---

## All knobs in this study

The full catalogue of what could be turned in this research. Split into three
groups: knobs we actively vary in Phase 1 (the research question), knobs we
deliberately defer to later phases (kept fixed early so the story stays clean),
and knobs we hold constant across every run (would confound the result).

### Group A — Phase 1 variables (the actual research question)

| Knob | Type | Phase 1 values | What it tests |
|---|---|---|---|
| `unet_gate_type` | str | `raw`, `sigmoid` | gate parameterization |
| `unet_gate_init` | float | `0.0`, `0.18`, `-1.5`, `-3.0` | strength at step 0 |
| `unet_skip_count` | int | `2`, `4`, `6` (best gate only) | how many bridges |

Hard-coded in current repo: `unet_skip_count = n_layers // 2`, raw zero gate.
The "raw 0.18" cell is what isolates *gate parameterization* from *nonzero start*.

### Group B — U-Net knobs deferred to later phases

These exist but are not in Phase 1. Add them only if Phase 1 finds a live gate,
otherwise the ablation table explodes. None are wired yet — they would be new
config flags.

| Knob | Default | Why deferred |
|---|---|---|
| `unet_apply_position` | `pre_block` | symmetry test (pre vs post the receiving block) |
| `unet_gate_granularity` | per-channel (current) | scalar / per-head / per-channel comparison |
| `unet_skip_transform` | identity | learnable linear vs identity bridge |
| `unet_skip_pattern` | mirror (`i → n-1-i`) | alt: every-other, dense, custom |
| `unet_skip_target` | residual stream | residual vs attn-input vs FFN-input |
| `unet_gate_lr_mul` | 1.0 | gate-specific LR multiplier |
| `unet_gate_wd` | inherits global | exempt gate from weight decay (decay shrinks toward 0) |

### Group C — knobs held FIXED across every Phase 1/2/3 run

If we vary these inside the U-Net study, we can't tell whether U-Net helped or
the new value of the knob did. They have known effects on val loss at this
scale and would dominate the signal. Drawn from `configs/llm_config.py`.

**Architecture (same model shape every run):**
`d_model`, `n_heads`, `n_layers`, `d_ff`, `n_kv_heads`, `emb_rank`,
`max_seq_len=2048` (locked to the downloaded data, do NOT change),
`vocab_size`, `ffn_variant`.

**Training schedule (same recipe every run):**
`seed=42`, `muon_lr`, `muon_momentum`, `adamw_lr`, `warmup_ratio`,
`schedule_type`, `weight_decay`, `grad_clip`, `dropout`, `batch_size`,
`gradient_accumulation_steps`, `train_tokens`, `compile_model`, `use_amp`.

**Other arch flags that interact strongly with skip topology:**
`tie_layer_groups` — hard-incompatible with U-Net (code raises); leave at `1`.
`use_post_norm` — flips where the skip lands relative to norm; leave at pre-norm
default. `norm_type`, `qk_norm_type` — same reason, leave at `rmsnorm`.

---

## Will other hyperparameters influence this result?

Short answer: yes, several would, which is exactly why Group C is fixed.
Itemized risk list, in rough order of how badly each one could fake or mask
a U-Net "win":

| Knob | Direction of risk | Why |
|---|---|---|
| `muon_lr` / `adamw_lr` | high | A higher LR can rescue a raw-zero gate by growing it faster; a lower LR can starve `sigmoid(-1.5)`. LR essentially tunes "how fast the gate wakes up." Fixing it is the only way to make the gate-init comparison fair. |
| `warmup_ratio`, `schedule_type` | high | Warmup decides when the gate gets useful gradient. A decay schedule that lands near zero LR right when the gate finally activates flips the result. |
| `weight_decay` | medium-high | Weight decay shrinks the gate parameter toward 0. With `gate=raw`, decay drags the bridge off. With `gate=sigmoid(init=-1.5)`, decay pulls the raw parameter toward 0, which moves the *effective* gate from 0.18 toward 0.5 — i.e. STRENGTHENS the bridge. The current value (`weight_decay=0.2`) is moderately aggressive; do not change it inside the study. |
| `seed` | medium | Tiny-scale val loss has ~0.01-0.02 seed noise. The same-seed control rows above already handle this; do not multi-seed Phase 1 — wait for Phase 2/3. |
| `grad_clip` | medium | A loose clip lets one early bad batch swing the gate; a tight clip slows its growth. The repo default `1.0` is reasonable; do not touch. |
| `dropout` | low at this scale | `0.0` default in repo; raising it would mask any small mechanism. Leave off. |
| `batch_size`, `gradient_accumulation_steps` | low | Move only to fit memory; effective batch is part of the LR scaling story, keep constant inside the study. |
| `compile_model` | none expected | Performance only, no numerical difference at FP32/AMP. |
| `use_amp` | low | AMP can clip tiny gate gradients via loss-scaling; leave on (default) consistently across all runs. |
| `max_seq_len`, data choice | very high but locked | Changing data invalidates RoPE cache and breaks training. Repo rule: do not change. |
| `tie_layer_groups`, `use_post_norm`, `norm_type` | high but structural | These change WHERE the skip lands. They are valid future studies (Group B), not Phase 1 confounds. |

**Headline rule for Phase 1:** every run differs from control in **at most one**
of `unet_gate_type`, `unet_gate_init`, `unet_skip_count`. If any other field of
`LLMConfig` differs, the row gets thrown out.

**Where other hyperparams DO matter — at the very end.** If U-Net survives
Phase 3, a single follow-up at full ladder may need to confirm that the
sigmoid-gate plateau is not LR-sensitive (one cheap LR sweep with the best
variant, at 10M shape). That goes in the "future work" line of the writeup, not
in this plan.

---

## Why this plan exists

The tutorial describes the nanoGPT-speedrun version:

```python
x = x + sigmoid(gate) * skip
gate init = -1.5  # sigmoid(-1.5) ~= 0.18
```

The current repo implementation is different:

```python
x = x + raw_gate * skip
raw_gate init = 0
```

That means the first research question is not "do U-Net skips work?" It is:

**Does the working trick depend on a small nonzero sigmoid gate?**

Until that is answered, do not spend time on broad model-size or hyperparameter
sweeps. Keep LR, schedule, tokenizer, sequence length, optimizer, and data fixed.

---

## Platform choice

Default to **Kaggle**, not Colab, for this batch.

Reasons:

- Kaggle notebook runs are easier to resume and collect as artifacts.
- Kaggle exposes accelerator selection through notebook metadata / CLI
  (`NvidiaTeslaP100`, `NvidiaTeslaT4`).
- Colab GPU availability and limits are dynamic, so it is fine for quick manual
  pokes but less pleasant for a multi-run ablation queue.

Use a single GPU. The repo training script is not DDP-wired, so a dual-T4
runtime only helps if one process uses one device and another process uses the
other device. Keep this plan simple: one run at a time.

---

## Time budget estimates

These are planning estimates for a free Kaggle/Colab-class 16GB GPU. Actual time
depends on whether the runtime gives T4 or P100, data-cache speed, and PyTorch
version.

| Tier | Config | Steps | Estimated time / run | Use |
|---|---:|---:|---:|---|
| tiny | `Tiny1M3MConfig` | ~700 | 5-10 min | mechanism ranking |
| short screen | `Screen10M1MConfig` | ~250 | 15-30 min | cheap 10M-shape smoke test |
| medium screen | `Screen10M5MConfig` | ~1220 | 60-120 min | promote only tiny winners |
| full screen | `Screen10M20MConfig` | ~4880 | 4-8 hr | claim-worthy screen, if still alive |
| full ladder | `Full10M200MConfig` | ~48,800 | 40-80 hr | not for free Kaggle/Colab |

Recommended Kaggle budget:

- Phase 1 tiny batch: 8-10 runs, about **1-2 hours**.
- Phase 2 short/medium screen: 3-4 runs, about **3-7 hours**.
- Phase 3 full screen: 2 runs, about **8-16 hours**.

Stop after Phase 1 if the best U-Net variant is not clearly ahead of the same
seed control by at least ~0.02 val loss at tiny scale.

---

## Implementation prerequisites

Before running this study, add these config knobs:

| Knob | Purpose |
|---|---|
| `unet_skip_count` | number of early-to-late bridge pairs; default `n_layers // 2` |
| `unet_gate_type` | `"raw"` or `"sigmoid"` |
| `unet_gate_init` | initial gate value, e.g. `0.0`, `-1.5`, `-3.0`, `0.18` |
| `unet_apply_position` | optional later knob: `"pre_block"` vs `"post_block"`; not Phase 1 |

Minimum code contract:

- `unet_skip_count` must actually control how many early outputs are saved and
  read back.
- `unet_gate_type="raw", unet_gate_init=0.0` must reproduce the current behavior.
- `unet_gate_type="sigmoid", unet_gate_init=-1.5` must match the tutorial.
- Metrics must record the active flags so old and new runs are distinguishable.

---

## Phase 1 — tiny mechanism check

Run on `Tiny1M3MConfig`, seed 42, fixed data and training defaults.

### Batch A — gate/init ablation

| Name | Gate | Init | Skip pairs | Question |
|---|---|---:|---:|---|
| `tiny_unet_ctrl` | none | n/a | 0 | same-seed control |
| `tiny_unet_raw0` | raw | 0.0 | 6 | current repo behavior |
| `tiny_unet_raw018` | raw | 0.18 | 6 | is nonzero strength enough? |
| `tiny_unet_sigmoid_m15` | sigmoid | -1.5 | 6 | tutorial / speedrun version |
| `tiny_unet_sigmoid_m30` | sigmoid | -3.0 | 6 | smaller nonzero start |

Decision:

- If `sigmoid_m15` beats `raw0`, the post is about **gate parameterization**.
- If all variants wash out, U-Net skips are probably not worth promoting.
- If `raw018` wins too, the key may be **nonzero skip strength**, not sigmoid.

### Batch B — skip-count ablation

Run only the best gate from Batch A:

| Name | Skip pairs for 12 layers | Question |
|---|---:|---|
| `tiny_unet_best_2` | 2 | only deepest late blocks read early features |
| `tiny_unet_best_4` | 4 | partial U |
| `tiny_unet_best_6` | 6 | full half-depth U |

Decision:

- If 2 or 4 beats 6, the story is "too many bridges can over-inject early
  features."
- If 6 wins, the full U pattern is plausible.

---

## Phase 2 — test inside the stronger tiny stack

Only do this if Phase 1 finds a live gate/count.

Use the current stronger tiny recipe:

```text
use_value_embed=True
use_q_gain=True
use_sliding_window=True
sliding_window_size=384
rope_base=250000
norm_type=pnorm1.5
```

Run:

| Name | U-Net? | Purpose |
|---|---|---|
| `tiny_stack_ctrl` | no | same-stack control |
| `tiny_stack_unet_best` | yes | does U-Net add on top of the current best stack? |
| `tiny_stack_multiscale_unet_best` | yes + multiscale heads | compare against previous promising combo |

Decision:

- If U-Net only helps in the stack, frame it as an **interaction effect**, not a
  standalone trick.
- If it helps clean and stacked, promote to screen.

---

## Phase 3 — 10M-shape confirmation

Promote only the best variant.

Run:

| Name | Config | Purpose |
|---|---|---|
| `screen1m_ctrl` | `Screen10M1MConfig` | cheap 10M-shape smoke test |
| `screen1m_unet_best` | `Screen10M1MConfig` + best U-Net | catches broken scaling cheaply |
| `screen5m_ctrl` | `Screen10M5MConfig` | stronger confirmation |
| `screen5m_unet_best` | `Screen10M5MConfig` + best U-Net | decide whether full screen is worth it |

Only run `Screen10M20MConfig` if the 5M screen delta is still positive and the
curve is not just an early transient.

---

## What counts as publishable

Good X/Skool post:

- control vs raw-zero vs sigmoid gate
- same model shape, same data, same seed
- one tiny table plus a gate diagram
- exact agent prompt to implement it
- caveat that this is screen-tier until the 10M-shape run confirms it

Do not claim a universal law from Phase 1.

Strongest possible claim:

> U-Net skips were not enough by themselves. The useful detail was a small
> nonzero sigmoid gate: raw-zero gates washed out, while sigmoid(-1.5) survived
> the same-token, same-seed ablation.

Null result is still useful:

> I tested the speedrun U-Net skip trick under a fair tiny LM screen. In this
> setup, the gate/init variants did not beat control, so I would not tell an
> agent to add this blindly.

---

## Stop rules

- Stop after tiny if all variants are within ~0.01-0.02 val loss of control.
- Stop after short screen if the sign flips.
- Do not run multi-seed until one variant survives at least `Screen10M5MConfig`.
- Do not tune LR/schedule to rescue U-Net. That turns the study into a
  hyperparameter search.

