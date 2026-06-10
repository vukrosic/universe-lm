"""Canon conv (gated depthwise causal Conv1d on the residual stream).

The local-mixing half of the Griffin / Mamba hybrid playbook
(De/Smith/Fernando 2024, "Griffin: Mixing Gated Linear Recurrences
with Local Attention for Efficient Language Models", arXiv:2402.19427;
Allen-Zhu et al., "Physics of Language Models" Canon-layer line, 2024-2025).

Mechanism
---------
A depthwise Conv1d with kernel_size=3, left-padded by (kernel_size-1)=2
zeros along the time axis (NOT `padding=2` on the conv — that would
pad BOTH sides and leak future tokens). Output is a single scalar
learnable gate `g` per block, init 0 so step 0 ≡ no-conv baseline:

    y = x + g · DWConv(left_pad(x, 2, dim=time))

Step-0 identity
---------------
`g` is a `nn.Parameter(torch.zeros(1))`, so `g·DWConv(x) = 0` for
every input at init. The conv weights are nn.Conv1d's default
Kaiming-uniform init, but with `g=0` the contribution is zero
regardless. The block forward is bit-identical to a no-conv build
at step 0 (test (e) in `idea.md:118-120`).

Causality
---------
Enforced by left-padding the time axis by (kernel_size-1)=2 BEFORE
the conv. A perturbation at input position `t` can only affect
output positions `≥ t` because the conv kernel is size 3 and the
two leftmost inputs to the conv are always zeros (the pad).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CanonConv(nn.Module):
    """Gated depthwise causal Conv1d on the residual stream.

    Args:
        d_model: channel dim of the residual stream (B, T, d_model).
        kernel_size: conv kernel width (default 3, per spec pin).

    Forward:
        x: [B, T, d_model]
        Returns: [B, T, d_model] = x + g · DWConv(left_pad(x, K-1, dim=time))
    """

    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        # Depthwise = one filter per channel. bias=False because the
        # gate absorbs any constant offset.
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=0, groups=d_model, bias=False,
        )
        # Scalar output gate. Init 0 → step-0 ≡ no-conv baseline.
        self.gate = nn.Parameter(torch.zeros(1))
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model] → conv1d expects [B, d_model, T]
        h = x.transpose(1, 2)
        # Left-pad (kernel_size - 1) along the time axis. (left, right) pad
        # on the last dim of a [B, d_model, T] tensor is time. We do NOT
        # set padding=2 on Conv1d (that would pad both sides → leaks future).
        h = F.pad(h, (self.kernel_size - 1, 0))
        h = self.conv(h)  # [B, d_model, T]
        # Back to [B, T, d_model] and gate.
        return x + self.gate * h.transpose(1, 2)
