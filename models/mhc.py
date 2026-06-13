"""Multi-stream residual (Hyper-Connections, mHC).

Xie et al., "Hyper-Connections" (arXiv:2409.19606, Sept 2024). Used as the
residual-stream backbone of DeepSeek-V3 (671B MoE, ~37B activated).

This module wraps a standard `TransformerBlock` and maintains `n_resid`
parallel residual streams of width `d_l = d_model // n_resid` each.
Before each block, a learnable mixing matrix `B_l ∈ R^{n_resid × n_resid}`
mixes the streams into the block's input; after each block, a second
mixing matrix `A_l ∈ R^{n_resid × n_resid}` mixes the block's residual
contribution; a third matrix `C_l ∈ R^{n_resid × n_resid}` mixes the
combined stream into the next layer's input.

Identity at step 0: `A_l = B_l = C_l = I_{n_resid}` ⇒ `x_out = block(x)`
bit-identical to the pre-norm baseline.

With `n_resid = 1`, all three matrices are `1×1` identity ⇒
construction collapses to standard residual.
"""
import torch
import torch.nn as nn


class MultiStreamResidual(nn.Module):
    """Per-position wrapper: maintains its own (A_l, B_l, C_l) mixing.

    The block's parameter count is unchanged (the wrapper is a thin
    pass-through). Extra parameters: `3 × n_resid²` scalars per
    position. At n_resid=4, n_layers=6 (tiny1m3m): 3·16·6 = 288 scalars,
    negligible vs the 0.94M base.
    """

    def __init__(self, block: nn.Module, n_resid: int, d_model: int):
        super().__init__()
        if d_model % n_resid != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_resid ({n_resid})"
            )
        # Hold the block as a non-registered attribute. If we used
        # `self.block = block`, the same block would appear twice in the
        # module tree (once via `transformer_blocks`, once via the
        # wrapper), and `self.apply(_init_weights)` would re-initialize
        # its parameters with a different RNG draw, breaking
        # step-0 bit-identity with the baseline. The block is owned by
        # `self.transformer_blocks` (so `model.to(device)` and
        # `state_dict()` both find it through that path); the wrapper
        # just calls it.
        object.__setattr__(self, "block", block)
        self.n_resid = n_resid
        self.d_model = d_model
        self.d_l = d_model // n_resid
        # Identity init — no stream mixing at step 0 ⇒ baseline path.
        # `nn.Parameter` (not `nn.Linear`) so `_init_weights` (which only
        # touches Linear/Embedding) does not overwrite the identity.
        self.A = nn.Parameter(torch.eye(n_resid))
        self.B = nn.Parameter(torch.eye(n_resid))
        self.C = nn.Parameter(torch.eye(n_resid))

    def _mix(self, x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Apply (M ⊗ I_{d_l}) · x. x: (B, T, d_model), M: (n_resid, n_resid).

        Reshape x into (B, T, n_resid, d_l), contract M on the stream axis,
        reshape back. Equivalent to a per-stream dense linear layer where
        M mixes streams but each stream's channels are preserved.
        """
        B, T, D = x.shape
        x_s = x.view(B, T, self.n_resid, self.d_l)
        # einsum: M[i,j] * x_s[b,t,j,d] -> out[b,t,i,d]
        out = torch.einsum("ij,btjd->btid", M, x_s)
        return out.contiguous().view(B, T, D)

    def forward(
        self,
        x: torch.Tensor,
        x0=None,
        ve=None,
        v_residual=None,
        layer_index=None,
    ) -> torch.Tensor:
        """Apply B mix → run block → extract residual contribution → A mix
        on residual → C mix → next input. Signature matches
        `TransformerBlock.forward` so the existing model loop calls it
        with no signature change.

        At init (A=B=C=I): x_out = block(x). Bit-identical to baseline.
        """
        x_pre = self._mix(x, self.B)
        x_block_out = self.block(
            x_pre,
            x0=x0,
            ve=ve,
            v_residual=v_residual,
            layer_index=layer_index,
        )
        # Residual contribution = block_out − input. With B=I, this is
        # exactly the standard sublayer contribution (attention + FFN
        # outputs, before any post-attention residual add mixes in).
        # `A` learns to scale/permute that contribution across streams.
        sublayer = x_block_out - x_pre
        x_with_A = x_pre + self._mix(sublayer, self.A)
        # `C` mixes the combined stream (input + scaled sublayer) before
        # handing off to the next position. With C=I at init, this is a
        # no-op.
        return self._mix(x_with_A, self.C)