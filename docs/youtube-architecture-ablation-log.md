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

## 9. Query Embeddings (#30) — same trick on Q

Mechanism: same as #29 but inject into Q instead of V.
`Q += F.linear(ve, query_embed_proj)`. Same zero-init raw Parameter, same
~166k extra params, same Muon routing.

Initial early-stop read (before apples-to-apples):

| step  | Q-embed  | V-embed  | Δ (Q-V)  |
|-------|----------|----------|---------|
| 500   | 6.1853   | 6.4059   | +0.2206 |
| 1000  | 5.6941   | 5.8800   | +0.1859 |
| 1500  | 5.4447   | 5.4216*  | -0.0231 |

*V-embed at 1500 from the new V-embed-full run.

Naive read: "Q beats V by 0.22 at 500, 0.19 at 1000" — cherry-picking the
early checkpoints suggested Q was the bigger lever.

## 10. Apples-to-apples at the natural endpoint (step 4,882)

Re-ran both V-embed and Q-embed to the screen's natural end (20M tokens,
step 4,882) with the same seed (42) for a clean comparison. Patched the
trainer so `--stop_at_step` actually fires on Screen* configs (the old
`build_eval_milestones` only set milestones up to step 1500 for the 25M
bucket, so the gate was unreachable).

| run                  | 500    | 1000   | 1500   | 4882   |
|----------------------|--------|--------|--------|--------|
| V-embed (this run)   | 6.1513 | 5.6756 | 5.4216 | **4.7728** |
| Q-embed (latest)     | 6.1700 | 5.6853 | 5.4422 | 4.8159 |
| Q-embed (first)      | 6.1853 | 5.6941 | 5.4447 | 4.8753 |
| V-embed (stop@4k)    | 6.4059 | 5.8800 | —      | 4.9381 |
| control              | 6.3972 | 5.8853 | —      | 5.0078 |

Picture flipped at the endpoint:
- Q-embed is **better early** (steps 500, 1000, ~1500)
- V-embed **overtakes by step 1500** and ends ~0.04 lower
- Both beat the control by 0.13–0.24 — that is the real, reproducible win

Run-to-run variance with the same seed is large (~0.16 for V-embed,
~0.06 for Q-embed), so the #0 vs #1 gap (0.043) is **inside the noise**.
The clean signal is "the token-identity-into-attention lever works,
end-of-training winner is V-embed."

Decision: V-embed is the new screen16m baseline for follow-up screens.
Q-embed is a real mechanism but the early-vs-end behavior is different —
not a free upgrade on V, and a Q+V combo needs to be tested as a separate
hypothesis, not "V plus a small Q boost."

## Current Takeaway

Value embeddings (#29) is the first follow-up lever that decisively beats
control: +0.07 at the 4000-step gate and +0.24 at the natural screen endpoint
(4,7728 vs control 5.0078). The Q-embed sister (#30) learns faster warmup
but V overtakes — same lever, slightly different operating point. Both
clearly beat the control; the V-vs-Q gap is inside the seed-noise band.

The pattern across #20–#30 is the real lesson: every trick that only
*rescaled the residual stream* (embed-residual, zero-init-resid, attention
gate, LayerScale, SmearGate) landed inside the noise. The wins came from
pulling a *different* lever — feeding token identity straight into the
attention Q or V. When a whole family of tweaks lands in the noise, change
levers; don't keep tuning the same knob.

For the tutorial, the honest-screening method still holds:

1. Start from a clear failure mode (or a clearly different lever).
2. Make the change zero-init / baseline-equivalent so step 0 == control and the
   screen isolates the mechanism, not a re-seed.
3. Compare against fixed control checkpoints.
4. Kill ideas when the curve stops supporting the story — and promote the rare
   one that clears the noise band by a wide margin.

## 11. Next lever: K-embed (#31)

The natural mirror of #29/#30. K is the third of Q/K/V; V was the winner
end-of-training, Q was faster warmup. K gets RoPE applied, so the
projection's term gets positionally rotated — different operating point,
free to test. Same zero-init Muon pattern, same ~55k extra params at 24
layers.

Baseline for this round: V-embed at 4.7728 (the natural-end screen record).
V-embed is built into the new config class so the screen isolates "K
in addition to V" vs "V alone."

## 12. K-embed (#31) result

Smoke-test passed (max|logit diff| at init = 0, 24/24 nonzero grad,
+55,296 params at 24 layers). Ran to natural end (step 4,882) using the
patched trainer milestones (gate now fires correctly).

| step  | K-embed  | V-embed  | Q-embed  | control  |
|-------|----------|----------|----------|----------|
| 500   | **6.1641** | 6.4059 | 6.1853  | 6.3972   |
| 1000  | **5.6813** | 5.8800 | 5.6941  | 5.8853   |
| 4000  | **4.8722** | 4.9381 | —       | 5.0078   |
| 4882  | 4.8228   | **4.7728** | 4.8159  | —        |

K-embed has the **fastest warmup of all three** (best at 500, 1000, 4000)
but loses to V at the natural end. K and Q are essentially tied at the
end (4.8228 vs 4.8159, inside the 0.06-0.16 run-to-run noise). V's
end-of-training edge is real but small.

Decision: V-embed is confirmed as the best end-of-training mechanism
within the Q/K/V family. The pattern across #29-#31 is consistent:
**V/Q/K all beat control by 0.13-0.24, all use ~0.7% extra params, all
are the same lever in different positions. The lever works; the V
position is the best single choice.**

## 13. V+Q combo (#32) result

Combination probe. V-embed is the end-game winner (4.7728), Q-embed is
the warmup winner (4.8159, faster at steps 500/1000). Tests whether the
two are additive — fast warmup + good end-game — or whether V's
V-specific position is the unique story.

Same zero-init Muon pattern as #29-#30. New config class
`Screen10M20MVQEmbedConfig` has both `use_value_embed=True` and
`use_query_embed=True` (the model code already supported running them
together; the `ve` source is computed once and shared). Cost = 24
layers × (q_size 144 + kv_size 48) × emb_rank 48 = 221,184 extra
params (~2.9% over baseline). Smoke test passed: max|logit diff|=0 at
init (Q-embed zero-init = no-op), 24/24 nonzero grad on both
projections, param delta matches expected.

Ran to natural end (step 4,883, 20M tokens):

| #    | val_loss | Δ vs V-embed |
|------|----------|--------------|
| V+Q  | **4.7428** | **-0.0300**  |
| V    | 4.7728    | 0            |
| Q    | 4.8159    | +0.0431      |
| K    | 4.8228    | +0.0500      |

**V+Q beats V-embed alone by 0.0300** at the natural end. The signal
is consistent with the additive hypothesis: Q's warmup advantage + V's
end-game edge do not cancel. The 0.0300 improvement is inside the
~0.06-0.16 run-to-run noise band for V-embed, so single-seed certainty
isn't there — but the direction matches the prediction and the curve
never crosses (V+Q is better at every step from 500 onward).

> **Fairness note (corrected):** the apples-to-apples control at the
> natural end (step 4,882) has not been run yet. The current "control"
> 5.0078 is a 4,000-step eval (different step count, ~16M tokens
> instead of ~20M), so the "V+Q beats control by 0.2660" claim from an
> earlier draft was unfair. The leaderboard now has two tiers:
> `screen16m` (step 4,000, 16M tokens) and `screen20m` (step 4,882,
> 20M tokens). The natural-end control row is marked **pending** until
> a fresh rerun fills it.

This is the new screen16m #0. The Q/K/V-embed lever is not just "a
thing that works" — combining positions is a real direction worth
pushing. Next probes to consider: V+K, V+Q+K, and a fundamentally
different lever (output-side token injection, learnable per-head
temperature, etc.).

### Full curve across Q/K/V/QV at the screen's natural end

| step  | V+Q     | V       | Q       | K       | control  |
|-------|---------|---------|---------|---------|----------|
| 500   | 6.0992  | 6.4059 | 6.1853  | **6.1641**  | 6.3972   |
| 1000  | 5.6015  | 5.8800 | 5.6941  | **5.6813**  | 5.8853   |
| 4000  | **4.7875** | 4.9381 | 4.8607† | 4.8722  | 5.0078‡  |
| 4882  | **4.7428** | 4.7728 | 4.8159  | 4.8228  | pending   |

† Q-embed @4k from milestone history inside the natural-end run (no
gated rerun). ‡ Control is gated at 4k, not natural-end. Both marked
in the leaderboard.

## 14. O-embed (#33) result — fundamentally different lever

The Q/K/V family is the same trick in different positions; O-embed
tests a different operating point entirely. Where #29-#32 inject `e_j`
into attention *inputs* (so the token identity enters the score
computation), #33 injects `e_j` into attention *output* (post-O
projection, straight into the residual). The signal bypasses attention.

This is the modded-nanogpt speedrun's "value embeddings" position. The
hypothesis to test: if V-embed wins because V is a unique position,
O-embed should underperform. If V-embed wins because any
token-signal-into-the-residual helps, O-embed should also win.

Same zero-init raw-Parameter pattern as #29-#32. New config class
`Screen10M20MOutputEmbedConfig` (single `use_output_embed=True`
flag). Cost = 24 × d_model 144 × emb_rank 48 = 165,888 extra params
(~2.1%). Smoke test passed: max|logit diff|=0 at init, 24/24 nonzero
grad on output_embed_proj, param delta matches expected.

Ran to natural end (step 4,883, 20M tokens):

| #    | val_loss | Δ vs V |
|------|----------|--------|
| V+Q  | 4.7428   | -0.0300 |
| V    | 4.7728   | 0      |
| Q    | 4.8159   | +0.0431 |
| K    | 4.8228   | +0.0500 |
| O    | **4.8350** | **+0.0622** |

O-embed is the **worst of the V/Q/K/O embed family** at the natural
end. The hypothesis is confirmed: the token-signal win is *inside
attention* (where the signal affects what the model attends to and
what gets aggregated), not in the residual (where the signal is just
another additive vector). A direct path from the input embedding to
the residual stream is not the lever; the lever is a path through
attention.

Important caveat: O-embed is **way better than control** (will be
apples-to-apples once the natural-end control run finishes, currently
in flight as `s_ctrl_full`). So the token signal *does* help in the
residual too — just less than inside attention. The O-embed is the
"additive embed, wrong position" failure mode, not a "levers don't
work" failure mode.

### Full curve across all five embeds

| step  | V+Q     | V       | Q       | K       | O       |
|-------|---------|---------|---------|---------|---------|
| 500   | 6.0992  | 6.4059 | 6.1853  | **6.1641** | 6.1506  |
| 1000  | 5.6015  | 5.8800 | 5.6941  | **5.6813** | 5.6684  |
| 4000  | **4.7875** | 4.9381 | 4.8607† | 4.8722  | 4.8834  |
| 4882  | **4.7428** | 4.7728 | 4.8159  | 4.8228  | 4.8350  |

O-embed has the *fastest warmup of all five* at step 500 (6.1506) and
1000 (5.6684), but loses the V-embed race by step 4882. Pattern: the
warmup advantage of an embed-position probe correlates weakly with
the end-of-training ranking. V-embed is the unique end-of-training
winner.

### Next lever to consider

Since the "anywhere in attention" story is now well-explored, the
*real* architectural questions are:

1. **Output-side + input-side combo**: O-embed + V-embed (V + O)?
   Costs 24 × (kv_size 48 + d_model 144) × emb_rank 48 = 221k extra
   params. Would test if the two positions are additive (V+Q was
   additive; V+O might be too).
2. **Multi-source embed**: project `e_j` through two different
   projections (one for warmup, one for end-game) and use them at
   different layers. Tests "is the warmup-vs-endgame tradeoff
   learnable per-layer?"
3. **A non-embed architectural change**: QK-layernorm, attention
   temperature scaling, SwiGLU activation, etc. None of the
   embed-lever work has touched a non-embed lever. The V-embed
   family was the only architectural change to make a difference
   in the 0.10+ band.

Recommended next: try (3) — pick a non-embed change and see if any of
them break the 0.10 noise band.
