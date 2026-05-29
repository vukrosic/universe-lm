# QK-Gain Hyperparameter Search Plan

## What is QK-Gain

A learnable per-head scalar on attention logits before softmax:

```
logits = gain[n_heads] * (Q @ K.T) / sqrt(d_k)
gain = nn.Parameter(torch.ones(n_heads))  # init 1.0
```

Literature (PRs #1394, #1413, #1953) shows monotonic improvement as gain rises from 4.0 to 5.25. Init matters — starting too high or too low changes the optimization trajectory.

## What we need to find quickly

1. **Optimal init value** — does QK-Gain work better with init=1.0, 4.0, 5.0?
2. **LR interaction** — does QK-Gain need a different LR than the rest of the model?
3. **Shape** — per-head (n_heads,) vs per-layer (n_layers, n_heads)?

## Fast hyperparam search design

**Budget:** ≤1M tokens per config, single seed (42). No baseline comparison.

**Configs to test:**

```
gain_init ∈ {0.5, 1.0, 4.0, 5.25}
lr_mult   ∈ {1.0, 0.5}   # gain LR relative to rest of model
shape     ∈ {per_head, per_layer}
```

Full grid = 4 × 2 × 2 = 16 configs. Too many for quick search.

**Reduced grid (8 configs):**
- gain_init={1.0, 4.0, 5.25} × lr_mult={1.0, 0.5} — per_head only (6 configs)
- best gain_init × per_layer shape (2 configs)

At 5M tokens per config (~2-3 min each on 3090) → ~20-25 min total.

**Evaluation metric:** smoothed_min_val_loss (EMA α=0.3) at 5M tokens.

## Implementation

### Changes to `models/layers.py` — `MultiHeadAttention.__init__`

```python
# QK-Gain parameter (per-head)
self.qk_gain = nn.Parameter(torch.ones(n_heads))  # init=1.0

# forward: apply gain before softmax
scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
scores = scores * self.qk_gain.view(1, 1, -1, 1)  # broadcast over B,T,seq
```

### Changes to `configs/llm_config.py`

```python
qk_gain_init: float = 1.0
qk_gain_lr_mult: float = 1.0  # gain LR = muon_lr * this
qk_gain_per_layer: bool = False  # if True, shape (n_layers, n_heads)
```

### Changes to `optimizers/muon.py`

Split qk_gain params into their own group with scaled LR.

## Sweep YAML

```yaml
base_preset: "5m"
train_tokens: 5_000_000
seed: 42
variants:
  - name: gain1.0_lr1.0
    overrides: {qk_gain_init: 1.0, qk_gain_lr_mult: 1.0}
  - name: gain4.0_lr1.0
    overrides: {qk_gain_init: 4.0, qk_gain_lr_mult: 1.0}
  - name: gain5.25_lr1.0
    overrides: {qk_gain_init: 5.25, qk_gain_lr_mult: 1.0}
  - name: gain4.0_lr0.5
    overrides: {qk_gain_init: 4.0, qk_gain_lr_mult: 0.5}
  - name: gain5.25_lr0.5
    overrides: {qk_gain_init: 5.25, qk_gain_lr_mult: 0.5}
  - name: gain4.0_perlayer
    overrides: {qk_gain_init: 4.0, qk_gain_per_layer: true}
  - name: gain5.25_perlayer
    overrides: {qk_gain_init: 5.25, qk_gain_per_layer: true}
```

## Decision

After 8-config search at 5M tokens:
- Pick best config by smoothed_min_val_loss
- If best is >0.01 better than init=1.0 baseline, proceed to 3-seed eval
- If best is <0.005 better, QK-Gain likely not worth pursuing at scale