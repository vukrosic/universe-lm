"""Autoresearch 163 — Post-Attention V-Mix Depthwise Convolution.

Symmetric depthwise Conv1d on the time axis over the post-attention
tensor `[B, T, d_model]` BEFORE the W_O output projection (post-
SDPA, post-reshape, pre-W_O). Conv weights are identity-initialized
(center tap = 1, rest = 0) via a raw `nn.Parameter(zeros(d_model, 1, k))`
with the center tap set inline ⇒ the conv is a strict identity at
step 0 ⇒ the block's attention output is bit-identical to baseline
at step 0 (within fp32 rounding noise of the conv arithmetic).

Third axis of a deliberate 3-axis locality test:
  143-shortconv (pre-attn, closed borderline-WIN-rule)
  157-conv-ffn (post-FFN-activation, closed null)
  163-v-mix-conv (post-attention on V, this one)

NB: imports `Tiny1M3MVMixConvConfig` directly rather than declaring
its own `class C(Tiny1M3MConfig): use_v_mix_conv: bool = True`,
because the latter would NOT override the parent's dataclass field
default (the parent's `False` is inherited verbatim without a
`@dataclass` re-decoration on the subclass, so the annotation is
ignored and `C().use_v_mix_conv` resolves to `False`). The
canonical `Tiny1M3MVMixConvConfig` is `@dataclass`-decorated in
`configs/llm_config.py` and correctly sets `True`. (Same pitfall
documented in `_arq_161-dyt-temp.py`.)
"""
from configs.llm_config import Tiny1M3MVMixConvConfig as C


if __name__ == "__main__":
    import sys
    import train_llm
    sys.modules["__main__"].C = C
    sys.argv = [
        "train_llm.py",
        "--config_class", "__main__.C",
        "--seed", "42",
        "--dataset_path", "processed_data/pretrain_1B",
        "--warmup", "false",
    ]
    train_llm.main()