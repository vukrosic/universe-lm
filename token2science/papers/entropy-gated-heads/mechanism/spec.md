# entropy-gated-heads

Each attention head output is scaled by a gate:

`gate_h = 1 - alpha_h * normalized_attention_entropy_h`

where `alpha_h` is a per-head learnable scalar initialized to `0`.

At initialization, every gate is exactly `1`, so the model is identical to the
baseline on the first forward pass.

Heads with confident, peaked attention have low normalized entropy, so they
keep close to full weight.
Heads with diffuse, high-entropy attention are down-weighted as `alpha_h`
learns away from zero.

The normalized entropy term should be computed from each head's attention
distribution over keys, with the usual entropy normalization so values stay on
a comparable scale across sequence lengths.
