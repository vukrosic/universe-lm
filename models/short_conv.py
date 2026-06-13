"""ShortConv — pre-attention depthwise 1D convolution (Hyena ShortConv variant).

A depthwise causal Conv1d applied to the residual stream *before* the
attention sublayer, providing a cheap local-context pass before the
global attention pass. From the Hyena hierarchy family
(Poli, Massaroli, et al. 2023, "Hyena Hierarchy: Towards Larger
Convolutional Language Models", arXiv:2302.10866) — specifically the
ShortConv variant: a single depthwise 1D conv with kernel size 3 or 4,
used as a pre-attention local aggregator.

Mechanism
---------
A depthwise Conv1d with kernel_size=k, LEFT-padded by (k-1) zeros
along the time axis (so the conv is causal — no future leak). One
filter per channel, no bias. Weights are initialized as the IDENTITY
kernel: LAST tap = 1.0, all other taps = 0.0. This means at
initialization, `ShortConv1D(x) = x` — the conv is a true identity
on the input at step 0. (The LAST tap reads the current token's value
after left-padding by (k-1); the center tap would read a previous
token and give a shifted output, not identity.)

Causality
---------
Enforced by left-padding the time axis by (kernel_size-1) BEFORE the
conv. A perturbation at input position t can only affect output
positions ≥ t because the conv kernel is size k and the leftmost
inputs to the conv are always zeros (the pad). The output length
matches the input length.

Step-0 identity
---------------
The conv weights are identity-initialized (last tap = 1, rest = 0).
For a causal left-padded input, the last tap of the kernel reads
the *current* token's value, so the output equals the input bit-for-
bit at step 0. The lever is applied as `x = x + g · ShortConv1D(x)`
where `g` is a per-block scalar gate (init 0). The `g=0` gate
ensures the contribution is zero at step 0 regardless of the conv's
internal state — bit-identical to the no-conv baseline at step 0.

Why a per-block scalar gate (and not just identity init)
--------------------------------------------------------
Even though ShortConv1D(x) = x at init, the block applies
`x = x + g · ShortConv1D(x)`. If we set g = 1 (no gate), the block
becomes `x = 2x` at step 0 — NOT bit-identical. The per-block scalar
gate g=0 is the cleanest way to make the lever a baseline-identity
trick while preserving the conv's identity init as a good starting
point for training (the conv has a meaningful structure the moment g
moves off 0).

Differentiation from CanonConv (023)
------------------------------------
- Placement: both are pre-attention, pre-LN on the residual stream.
- Init: CanonConv uses Kaiming-uniform init (nn.Conv1d default);
  ShortConv1D uses identity init (center=1, rest=0).
- Gate: both use a per-block scalar gate init 0.
- Kernel: CanonConv is pinned to k=3; ShortConv1D accepts k=3 or k=4
  via `short_conv_kernel`.
- The lever is "pre-attention identity-init conv" vs "pre-attention
  kaiming-init conv" — tests whether the conv's starting structure
  matters for the local-mixing lever.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortConv1D(nn.Module):
    """Causal depthwise 1D convolution with identity initialization.

    Args:
        d_model: channel dim of the residual stream (B, T, d_model).
        kernel_size: conv kernel width (3 or 4; default 3 per spec pin).

    Forward:
        x: [B, T, d_model]
        Returns: [B, T, d_model] = DWConv(left_pad(x, K-1, dim=time))
    """

    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        # Depthwise = one filter per channel. bias=False (the gate
        # absorbs any constant offset, like CanonConv).
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=0, groups=d_model, bias=False,
        )
        # Identity init: LAST tap = 1, rest = 0. nn.Conv1d weight
        # shape for depthwise is [d_model, 1, kernel_size] (in_channels/
        # groups = 1 per filter). The last tap (position kernel_size-1)
        # is the one that reads the *current* token's value after
        # left-padding by (kernel_size-1) — so output[t] = padded[t +
        # kernel_size-1] = x[t]. This gives true identity for a causal
        # left-padded conv. (The spec sketch mentioned the center tap
        # k//2, but with left-padding the center reads a *previous*
        # token — the output would be a shifted version, not identity.
        # The last tap is the correct position for "conv output = input"
        # on a causal left-padded conv.)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, 0, kernel_size - 1] = 1.0
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model] → conv1d expects [B, d_model, T]
        h = x.transpose(1, 2)
        # Left-pad (kernel_size - 1) along the time axis for causality.
        # We do NOT set padding=kernel_size-1 on Conv1d (that would pad
        # BOTH sides → leaks future).
        h = F.pad(h, (self.kernel_size - 1, 0))
        h = self.conv(h)  # [B, d_model, T]
        # Back to [B, T, d_model]. The block applies this through a
        # per-block scalar gate g=0 (set in TransformerBlock), so the
        # contribution is zero at step 0 regardless of the conv output.
        return h.transpose(1, 2)
