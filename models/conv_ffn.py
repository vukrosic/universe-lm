"""157 — Depthwise Conv inside FFN (post-activation, pre-down-projection).

A depthwise Conv1d applied to the FFN's pre-down-projection hidden
state (or, equivalently in this implementation, to the FFN output
applied in `TransformerBlock.forward` between `feed_forward(...)` and
the layerscale/sub_ln/residual-add chain). The conv mixes each
token's representation with its immediate neighbors before the
down-projection (or before the residual add), giving the FFN a free
local-mixing step at near-zero parameter cost.

Mechanism
---------
A depthwise Conv1d with `kernel_size=k` (default 3) and SYMMETRIC
padding `padding=k//2` on both sides of the time axis (this is a
non-causal local aggregator — the FFN-output mixing stage is allowed
to look at neighbors on both sides because the attention sublayer
has already integrated the full causal context). One filter per
channel, no bias.

Step-0 identity
---------------
Conv weights are identity-initialized (center tap = 1, all other
taps = 0). For symmetric padding and kernel size k, the center tap
reads the *current* token's value, so the conv output equals the
input bit-for-bit at step 0. The block applies this conv to the FFN
output with NO scaling gate, so the conv IS the identity function
at step 0 ⇒ forward is byte-identical to the no-conv baseline.

Different from ShortConv (143)
------------------------------
- Placement: 143 sits on the residual stream BEFORE the attention
  sublayer's pre-LN; 157 sits on the FFN's post-activation (or
  post-FFN) tensor — i.e., the FFN *output* side, not the attention
  *input* side.
- Causality: 143 is causal (left-pad only) because pre-attention
  leakage of future tokens is forbidden; 157 is symmetric (look at
  both neighbors) because the FFN output is already a function of
  the full causal context (the attention sublayer has run).
- Init: 143 uses last-tap identity init (for causal left-padding);
  157 uses center-tap identity init (for symmetric padding).

Why no per-block output gate?
-----------------------------
The center-tap symmetric identity init makes the conv a *strict*
identity function at step 0 (`y = x` exactly, with no residual
addition). There is no `2x` artifact to absorb. So no gate is
needed — the conv goes in front of the down-projection (or the
residual add) as a plain pre-existing op, and the lever is "do you
want a k-neighbor local mixer in the FFN pipeline?"

From ConvBERT (Jiang et al. 2020, arXiv:2008.02496) and ConvNeXt
(Woo et al. 2020) — depthwise conv inside FFN for parameter-
efficient local mixing. Tested at 110M+ BERT and at all ConvNeXt
scales; transfer risk is low (≥100M source scale, multiple
replications).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFFN(nn.Module):
    """Symmetric depthwise 1D convolution with identity initialization.

    Args:
        d_model: channel dim of the FFN output tensor
            ([B, T, d_model] in this implementation — applied to the
            FFN's output tensor in `TransformerBlock.forward`).
        kernel_size: conv kernel width (default 3 per spec pin;
            pinned by `conv_ffn_kernel` config flag, range 3-7).

    Forward:
        x: [B, T, d_model]
        Returns: [B, T, d_model] = DWConv(pad(x, (k//2, k//2), dim=time))

    The conv is a strict identity at step 0 (center-tap = 1, all
    other taps = 0 with symmetric padding ⇒ output[t] reads x[t]
    at the center tap and 0 at the side taps, yielding `y = x`).
    """

    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        # Depthwise = one filter per channel. bias=False because the
        # identity init absorbs any constant offset (a constant bias
        # would shift the FFN output by a learnable per-channel
        # amount; we don't want that here, the lever is purely the
        # local-mixing filter).
        assert int(kernel_size) >= 3, (
            f"conv_ffn_kernel={kernel_size} must be >= 3 (odd, for "
            f"symmetric center-tap identity init)"
        )
        assert int(kernel_size) % 2 == 1, (
            f"conv_ffn_kernel={kernel_size} must be odd for symmetric padding"
        )
        # Construct the conv weight as a raw `Parameter` (NOT
        # `nn.Conv1d`) so the construction does NOT consume RNG.
        # This keeps the RNG state aligned with the baseline path
        # for the step-0 byte-identity test (any RNG advance
        # between the two constructions would shift the next-
        # block qkvo_proj random init and break the comparison).
        # Same raw-`Parameter` pattern as 156-moa (`moa_extra_kv`,
        # `moa_router_weight`).
        weight = torch.zeros(d_model, 1, int(kernel_size))
        weight[:, 0, kernel_size // 2] = 1.0
        self.weight = nn.Parameter(weight)
        # Cache kernel size and pad for the manual F.conv1d call.
        self.kernel_size = int(kernel_size)
        self._pad = self.kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model] → conv1d expects [B, d_model, T].
        # Use F.conv1d with explicit symmetric padding
        # (left = right = kernel_size // 2) so we control the
        # padding behavior precisely. `groups=d_model` makes it
        # depthwise.
        h = F.conv1d(
            x.transpose(1, 2),
            self.weight,
            bias=None,
            stride=1,
            padding=self._pad,
            groups=self.weight.shape[0],
        )  # [B, d_model, T]
        # Back to [B, T, d_model]. At step 0 this equals the input
        # bit-for-bit (center-tap identity with symmetric padding).
        return h.transpose(1, 2)
