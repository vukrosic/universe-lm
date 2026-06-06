# Bridge Early Layers to Late Layers to Improve Your LLM Training: U-Net Skips

By Vuk Rosić

U-Net skips connect the early layers of a transformer straight to its late layers, so the model can reuse early features deep in the stack.

![u-net skip architecture](images/unet_architecture.png)

This gives the late layers a shortcut back to the simple, local information the early layers saw.

It also gives gradients a shorter path to the early layers, which helps deep models train.

The whole thing is a handful of learned numbers, and it can start as almost a no-op.

## How it works, step by step

A deep transformer processes tokens one layer at a time, and each layer only reads the layer right below it.

Early layers tend to capture simple, local patterns, like which token is next to which.

By the time information reaches the late layers, those early details can get washed out.

A U-Net skip fixes this by saving the output of each early layer and adding it back into a matching late layer.

The pairing is symmetric, like the two sides of a letter U: the first layer connects to the last, the second to the second-to-last, and so on.

```text
layer 0  ->  layer 7
layer 1  ->  layer 6
layer 2  ->  layer 5
layer 3  ->  layer 4
```

Each bridge is gated, so the model decides how much of the early output to pull in.

The gate runs through a sigmoid, which keeps its strength between 0 and 1 and stable to train.

![the sigmoid gate scales the skip before adding it](images/unet_gate.png)

The gate weight starts at -1.5, so `sigmoid(-1.5)` is about 0.18, small but not zero.

A small nonzero start matters: a gate that starts at exactly zero gets almost no gradient and can fail to ever turn on.

So the skip begins faint, and training raises or lowers it per dimension as the model learns how useful the bridge is.

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

Put together, the first half writes its outputs and the second half reads them back, scaled by a learned per-dimension gate.

This is the same skip pattern used in the record-setting nanoGPT speedrun, where the sigmoid gate and the small nonzero start are what make it train well.
