# YouTube Architecture Ablation Log

Date: 2026-06-02

Purpose: keep a factual trail of the architecture ideas tested after the
embedding-factorization + depth record. This is source material for the YouTube
tutorial: what we tried, why it seemed plausible, what happened, and why we
kept or dropped it.

Baseline screen for comparison:

```text
run: logs/s_ctrl.log
config: Full10M200MConfig, seed 42, stop_at_step 4000
params: 7,682,832

step 0:    10.8125
step 500:   6.3972
step 1000:  5.8853
step 2000:  5.3856
step 4000:  5.0078
```

Promotion rule for these quick screens: do not promote a mechanism unless the
4000-step validation loss beats control by at least 0.01. The target is
`<= 4.9978`.

## 1. Embedding Residual

Run: `logs/s_embresid.log`

Mechanism: re-inject the original token embedding into each transformer block
with a learned per-channel residual mix.

Why we tried it: the new winning model is deep and narrow. A plausible failure
mode is that the token identity signal gets washed out through 24 layers. This
tries to keep lexical information available in every block.

Result:

```text
params: 7,689,888

step 0:    10.8125
step 500:   6.3841   better than control by 0.0131
step 1000:  5.8978   worse than control by 0.0125
step 2000:  5.4544   worse than control by 0.0688
```

Why it failed: it had a small warmup flash, then clearly degraded. The
extra embedding signal did not preserve useful information; it likely
interfered with the representation the deep stack was already learning.

Decision: killed early. Do not promote.

## 2. Zero-Init Residual Projections

Run: `logs/s_zeroinit.log`

Mechanism: initialize attention output projection and FFN down projection to
zero so each block starts as an identity function.

Why we tried it: a 24-layer narrow model might benefit from cleaner signal
propagation at the start of training. Zero-init residual branches are a common
stability idea.

Result:

```text
params: 7,682,832

step 0:    10.8125
step 500:   6.3953   better than control by 0.0019
step 1000:  5.8863   worse than control by 0.0010
step 2000:  5.3856   tied with control
```

Why it failed: it basically did nothing. The current warmup + decay schedule
already controls the deep stack well enough, so the identity start did not add
useful signal.

Decision: killed early. Do not promote.

## 3. Low-Rank Output Adapter

Run: `logs/s_outadapter.log`

Mechanism: keep the rank-48 tied factorized embedding/head, but add a rank-32
independent low-rank output correction to the logits. The adapter's final
projection starts at zero, so step 0 is exactly baseline.

Why we tried it: embedding factorization saved the record by compressing the
vocabulary table, but the tied output head is also bottlenecked through rank 48.
This asks whether the input embedding should stay cheap while the output
classifier gets a small escape hatch.

Result:

```text
params: 9,260,304

step 0:    10.8125
step 500:   6.3575   better than control by 0.0397
step 1000:  6.0775   worse than control by 0.1922
```

Why it failed: strong early shortcut, then collapse. The adapter adds a direct
logit path with a lot of leverage, and those matrices went through the same
aggressive optimizer grouping as hidden matrices. It learned something fast
but not something stable.

Decision: killed early. Interesting negative result for the video, not a
record path.

## 4. SmearGate

Run: `logs/s_smeargate.log`

Mechanism: add a learned per-channel amount of the previous token's embedding
before the transformer stack. It is causal, zero-init, and costs only
`d_model = 144` parameters.

Why we tried it: attention has to learn local token-pair features from scratch.
This gives the model a cheap input-side bigram hint without changing the
tokenizer, context length, data, or schedule.

Result:

```text
params: 7,682,976

step 0:    10.8125
step 500:   6.3416   better than control by 0.0556
step 1000:  5.8859   worse than control by 0.0006
step 2000:  5.3984   worse than control by 0.0128
step 4000:  5.0025   better than control by 0.0053
```

Why it only weakly worked: the step-500 gain mostly vanished. The final result
is a real small win, but below the promotion bar. It may help early lexical
features, but the transformer learns the same information soon enough.

Decision: keep as a teachable weak-positive result. Do not spend a full 200M
run unless later combined evidence justifies it.

## 5. U-Net Skips

Run: `logs/s_unetskip.log`

Mechanism: add zero-init learned bridges from early layer outputs to mirrored
late layers, like `layer 0 -> layer 23`, `layer 1 -> layer 22`, and so on.

Why we tried it: the record came from making the model much deeper. Deep narrow
models may lose early lexical details, so a visual U-Net bridge is a natural
architecture lesson: let early representations survive to the late stack.

Result:

```text
params: 7,684,560

step 0:    10.8125
step 500:   6.3891   better than control by 0.0081
step 1000:  5.8803   better than control by 0.0050
step 2000:  5.3894   worse than control by 0.0038
step 4000:  5.0081   worse than control by 0.0003
```

Why it failed: the skip bridges gave a tiny early benefit, then disappeared.
The late model did not need direct early-state reuse, or the zero-init bridges
were too weak to learn a useful routing pattern within this screen. The final
number is slightly worse than control.

Decision: drop. Good diagram, weak result.

## 6. Attention Output Gate

Run: `logs/s_attngate.log`

Mechanism: add a zero-init per-head multiplier on each attention output:
`attention_output *= (1 + gate)`. This costs `n_layers * n_heads = 144`
parameters and starts as exact baseline.

Why we are trying it now: full zero-init residuals did nothing, and U-Net skips
did not help. This tests a narrower stability/control idea: maybe attention
heads should be able to gradually adjust their contribution without also
muting the MLP path.

Current result:

```text
params: 7,682,976

step 0:   10.8125
step 500:  6.3700   better than control by 0.0272
step 1000: 5.8666   better than control by 0.0187
step 2000: 5.3894   worse than control by 0.0038
```

Why it failed: this was the best step-1000 signal of the follow-up batch, but
it still flipped negative by step 2000. The attention-only gate helped the
warmup trajectory but did not make the later validation curve better.

Decision: killed at step 2000. Do not promote.

## 7. LayerScale

Run: `logs/s_layerscale.log`

Mechanism: add learned per-channel residual-branch scales on both attention and
MLP outputs, initialized as exact baseline with `(1 + gate)`.

Why this is next: full zero-init residual projections did nothing, and
attention-only gating helped early but failed by step 2000. LayerScale is the
middle test: it gives every residual branch a learnable volume knob without
turning branches off at init.

Decision rule: run the same 4000-step screen. If step 1000 is not positive,
stop early. If it stays positive through step 2000, let it hit the 4000 gate.

Current result:

```text
params: 7,689,744

step 0:   10.8125
step 500:  6.3991   worse than control by 0.0019
step 1000: 5.8684   better than control by 0.0169
step 2000: 5.3812   better than control by 0.0044
step 4000: 4.9972   better than control by 0.0106  ✓ gate reached
```

## 8. Value Embeddings (#29)

Run: `logs/s_valembed.log`

Mechanism: inject the token embedding straight into the attention **values** at
every layer — `V += F.linear(ve, W)` — where `ve` is the existing low-rank
(factorized) token embedding, reused as the source. `W` is a raw zero-init Muon
matrix, so step 0 is an exact baseline AND it draws no RNG at init: every other
weight stays bit-identical to control, so the screen isolates the mechanism, not
a re-seed. Cost ~r·kv_size per layer (~55k total), inside the param-golf budget.
From modded-nanogpt speedrun records 55/63, adapted to reuse the factorized table.

Why this is next: the whole #20–#28 batch only ever *rescaled the residual
stream* — a small knob, every result inside the seed-noise band. Value embeddings
pull a different lever: they give attention direct access to token identity at
depth. First test off the residual-rescale corner.

Decision rule: same 4000-step schedule-matched screen. Promote only if it clears
control by ≫0.01 (above the noise band the residual levers lived in).

Current result:

```text
params: 7,738,128   (control + ~55k)

step 0:   10.8125              exact baseline (bit-identical to control)
step 500:  6.4059   worse than control by 0.0087   (zero-init ramp: not warmed up yet)
step 1000: 5.8800   better than control by 0.0053
step 2000: 5.3375   better than control by 0.0481
step 4000: 4.9381   better than control by 0.0697  ✓ gate reached — still widening
```

Decision: PROMOTE. First lever decisively above the noise band — ~7× LayerScale's
margin and ~7× the seed-noise band, and the gap is still widening at the gate
(0.048 → 0.070). New screen leader; not promoted to a full-length run in this
work (the screening is the deliverable).

## Current Takeaway

Value embeddings (#29) is the first follow-up lever that decisively beats
control: +0.0697 at the 4000 screen — ~7× LayerScale's margin and ~7× the
seed-noise band, with the gap still widening at the gate. New screen leader
for the 10M ladder; the 4,000-step result is the deliverable — no full-length
chase in this work.

The pattern across #20–#29 is the real lesson: every trick that only *rescaled
the residual stream* (embed-residual, zero-init-resid, attention gate,
LayerScale, SmearGate) landed inside the noise. The win came from pulling a
*different* lever — feeding token identity straight into the attention values.
When a whole family of tweaks lands in the noise, change levers; don't keep
tuning the same knob.

For the tutorial, the honest-screening method still holds:

1. Start from a clear failure mode (or a clearly different lever).
2. Make the change zero-init / baseline-equivalent so step 0 == control and the
   screen isolates the mechanism, not a re-seed.
3. Compare against fixed control checkpoints.
4. Kill ideas when the curve stops supporting the story — and promote the rare
   one that clears the noise band by a wide margin.
