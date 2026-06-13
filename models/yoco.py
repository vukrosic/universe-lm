"""129 — YOCO: You Only Cache Once (Sun et al. 2024, arXiv:2405.05254,
ICLR 2024 workshop).

Decoder-decoder cross-layer KV reuse: split the transformer into a
lower half (standard sliding-window self-attention) and an upper half
where each layer's attention reads a SHARED `(K_g, V_g)` cache
projected from the lower half's final residual stream — instead of
computing per-layer K, V from the input. Saves ~50% of the upper-half
K/V projection params (the W_K, W_V slices of `qkvo_proj` are unused on
the upper half) and at inference collapses the KV cache from
`O(L·d·T)` to `O(d·T)`. The lever is the cross-layer information flow
itself, not the cache saving (which doesn't affect the tiny1m3m
val-loss A/B).

Concretely:
  - Lower half (layers 0..yoco_split-1): standard
    `TransformerBlock` with `use_sliding_window=True` and
    `sliding_window_size=yoco_lower_window` (default 512).
  - At the boundary: a single `GlobalKVHead` projects the lower-half
    final residual stream to `(K_g, V_g)`, each of shape
    `[B, T, kv_size]`.
  - Upper half (layers yoco_split..n_layers-1): each layer is a
    `YOCOLlamaBlock` (a `TransformerBlock` whose inner MHA has
    `use_shared_kv=True`). The MHA skips its W_K, W_V projections
    and reads `(K_g, V_g)` instead — Q is still computed per-layer
    from the input.

Identity at step 0: the `GlobalKVHead` projections are standard
`nn.Linear(d_model, kv_size)` with normal init std=0.02 (matching the
rest of the model). K_g, V_g are `O(0.02)` per element, `O(0.02·√d_model)`
per kv_size row. The upper-half attention reads these and produces
`O(0.02²)` attention output per token — same magnitude order as the
standard per-layer W_K, W_V std=0.02 init path. NOT byte-identical to
the standard 12L self-attention baseline at step 0, but the deviation
is bounded and well within the NULL band.

Default off (`use_yoco=False`): `GlobalKVHead` is never built and the
baseline forward graph is bit-identical. See
`autoresearch/ideas/129-yoco/idea.md`.
"""

import torch
import torch.nn as nn

from .layers import TransformerBlock


class GlobalKVHead(nn.Module):
    """Project the lower-half residual stream to a shared (K_g, V_g) cache.

    Each of K_g, V_g is shape `[B, T, kv_size]` (matching the MHA's
    K/V-projection output). Both projections are std=0.02 normal init —
    same scheme as the merged qkvo_proj elsewhere in the model.

    Args:
        d_model: residual-stream width.
        kv_size: per-head dim summed over kv heads (= n_kv_heads · d_k).
    """

    def __init__(self, d_model: int, kv_size: int):
        super().__init__()
        self.k_proj = nn.Linear(d_model, kv_size, bias=False)
        self.v_proj = nn.Linear(d_model, kv_size, bias=False)

    def forward(self, x: torch.Tensor):
        return self.k_proj(x), self.v_proj(x)


class YOCOLlamaBlock(TransformerBlock):
    """TransformerBlock whose inner MHA reads shared K, V (cross-attention
    to the lower-half KV cache).

    Forwards all kwargs through to the parent `TransformerBlock.forward`,
    including the new `shared_kv=(K_g, V_g)` kwarg. The MHA's
    `use_shared_kv=True` flag (set in `__init__`) makes it skip the
    W_K, W_V slices of the merged qkvo_proj and use the supplied
    `shared_kv` tensors as K and V. Q is still computed per-layer from
    the input, then q_norm + RoPE applied per the standard path. K_g
    goes through k_norm + RoPE inside the MHA; V_g is used as-is (no
    norm, no RoPE — V has no positional semantics).
    """

    def __init__(self, *args, **kwargs):
        kwargs["use_shared_kv"] = True
        super().__init__(*args, **kwargs)
