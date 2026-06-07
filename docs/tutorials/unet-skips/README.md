# U-Net Skip Connections Make Small LLMs Train Faster - Until You Train Them Longer

By Vuk Rosić

We added this trick to a small model and validation loss improved at first:

![val loss curves](images/01_val_loss_curves.png)

A U-Net skip connects an early transformer layer to a matching late one.

It is just a small learned bridge that the model can scale up or down.

The bridge starts almost off, so it never hurts the model at the start of training.

Late layers can reach back to the simple, local features the early layers saw, and gradients get a shorter path back to the start.

![u-net skip architecture](images/unet_architecture.png)

## How it works, step by step

A deep transformer processes tokens one layer at a time, and each layer only reads the layer right below it.

Early layers capture simple, local patterns.

By the late layers those early details can get washed out.

A U-Net skip saves each early layer's output and adds it back into a matching late layer - first to last, second to second-to-last, and so on.

```text
layer 0  ->  layer 7
layer 1  ->  layer 6
layer 2  ->  layer 5
layer 3  ->  layer 4
```

Each bridge is gated with a sigmoid, so the model decides how much to pull in.

The sigmoid keeps the gate in [0, 1] - stable to train.

The gate weight starts at -1.5, so `sigmoid(-1.5)` is about 0.18.

A small nonzero start matters: a gate at exactly zero gets almost no gradient and can fail to ever turn on.

![the sigmoid gate scales the skip before adding it](images/unet_gate.png)

## In code

Keep one gate vector per bridge, initialized to -1.5.

```python
# n_skips bridges, one gate value per embedding dimension
gate = nn.Parameter(torch.full((n_skips, d_model), -1.5))
# sigmoid(-1.5) ~ 0.18  ->  the skip starts small but nonzero
```

In the first half of the layers, save each layer's output.

```python
skips = []
for i, block in enumerate(blocks):
    x = block(x)
    if i < n_skips:
        skips.append(x)        # remember early outputs
```

In the second half, before each layer, add its matching early output through the sigmoid gate.

```python
for i, block in enumerate(blocks):
    if i >= n_layers - n_skips:
        j = n_layers - 1 - i               # matching early layer
        x = x + torch.sigmoid(gate[j]) * skips[j]
    x = block(x)
```

The first half writes its outputs, and the second half reads them back scaled by a learned per-dimension gate.

## Phase 1 ablation

Eight runs on a tiny ~1M-param model (12 layers, d_model 64), 733 steps (~3M tokens), all with seed 42.

The three "raw0" rows below are bit-identical runs - the skip-count flag was ignored, so all three used the default k=6.

| run | use_unet_skips | gate_type | gate_init | skip_count | val_loss | val_ppl |
|---|---|---|---|---|---|---|
| `tiny_unet_ctrl` | false | - | - | 0 | 6.4422 | 627.78 |
| `tiny_unet_raw0_k2_real` | true | raw | 0.0 | **2** | 6.4503 | 632.90 |
| `tiny_unet_raw0` | true | raw | 0.0 | 6 (default) | 6.4203 | 614.20 |
| `tiny_unet_raw018` | true | raw | 0.18 | 6 (default) | 6.4272 | 618.43 |
| `tiny_unet_sigmoid_m15` | true | sigmoid | -1.5 | 6 (default) | **6.3816** | **590.85** |
| `tiny_unet_sigmoid_m30` | true | sigmoid | -3.0 | 6 (default) | 6.3831 | 591.77 |

![final val_loss bar chart - sigmoid wins by 0.04-0.07](images/02_final_loss_bar.png)

### Findings

At the full k=6, the sigmoid gate beats the raw gate by 0.046 val_loss even at a matched start (`sigmoid(-1.5)` and `raw_init=0.18` begin at the same effective weight), which suggested the [0, 1] bound was doing the work.

Phase 2 below tests that idea at other bridge counts and at scale, and the picture turns out to be narrower than this first read.

![sigmoid vs raw - same init, sigmoid ends lower](images/03_sigmoid_vs_raw.png)

The raw-gate skip count also looked non-monotonic here: k=2 was *worse* than no skips while k=6 was *better*.

Phase 2 finds the same zig-zag for sigmoid gates too, so this is better read as single-seed noise than as a real "minimum bridge count" law.

![skip count sweep - only one k=2 point is real (see caveats)](images/04_skip_count_scatter.png)

### Caveats

The original skip_count sweep was incomplete - the flag was ignored, so several runs were bit-identical at the default k=6. Phase 2 below re-ran it with the flag actually wired, and all runs are single-seed, so treat small gaps as directional.

## Phase 2: going deeper

We then filled the gaps and tested the two questions Phase 1 left open.

### Does skip count have a clean trend?

We ran the full sigmoid curve from k=1 to k=6 bridges, all at 3M tokens, seed 42.

The curve zig-zags: k=3 is the worst point and k=6 is the best, with no smooth trend in between.

At a single seed these wiggles are within the run-to-run noise, so the honest read is "more bridges trend slightly better, but pick k=6 and move on."

![skip count curve k=1 to 6 - noisy, k=6 best](images/05_skipcount_curve.png)

### Is it the bound or the starting point?

`sigmoid(-1.5)` and a raw gate at `0.18` start at the same effective weight, so any difference is the [0, 1] bound, not the start.

We compared them at k=2, k=4, and k=6.

The bound only clearly helps at k=6 - at k=2 the edge is tiny and at k=4 it disappears.

So "the sigmoid bound wins" is really a k=6 effect, not a general law.

![bound vs init at matched start - bound only helps at k=6](images/07_bound_vs_init.png)

### Does the win survive more tokens?

This is the one that matters for whether you should use it.

We re-ran the best config (sigmoid, k=6) against the no-skip control at 3M, 10M, and 20M tokens.

The advantage shrinks fast: +0.061 val_loss at 3M, +0.009 at 10M, then -0.007 at 20M.

Past about 10M tokens the U-Net skip stops helping and becomes a small net negative.

So the skip buys faster early convergence, not a better final model - and if you train long enough it slightly hurts.

![U-Net advantage shrinks then reverses with more training](images/06_scale_washout.png)

### Bottom line

Use U-Net skips when your model is small or undertrained and you want faster early progress.

Drop them once you train near convergence - here the gain reverses past 10M tokens.

This is also a reminder to always re-test a "win" at more tokens before you trust it.

### Want to go deeper in AI research? I'll coach you 1-on-1

Bring whatever you're stuck on - picking a direction, your first experiment, a paper you can't crack, your training setup, or a career move.

📆 **$20 (80% OFF) for the founding cohort - first 8 spots.** Not a fit? I'll
refund in the first 10 minutes, no hard feelings.
→ https://cal.com/vuk-ai/60-min

### Not ready for a call? Start free in the Skool

Every experiment I post comes with the scaffolded code and a step-by-step
protocol, so you can reproduce it yourself and then run your own variant. You also get the weekly research thread and a community of people doing real AI research.
Free to try.
→ https://www.skool.com/become-ai-researcher-2669/about
