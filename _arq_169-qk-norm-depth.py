#!/usr/bin/env python
"""Bootstrap for 169-qk-norm-depth. Subclass Tiny1M3MQKNormConfig
with use_qk_norm_depth=True.

Depth-conditional QK-Norm: keeps the 016 WIN shape (per-head
LayerNorm on Q, K) and adds one scalar `qk_norm_scale =
nn.Parameter(torch.ones(()))` per MHA. The scalar is applied AFTER
the per-head norm and BEFORE the QK matmul, so `Q ← Q ·
qk_norm_scale; K ← K · qk_norm_scale`. At init `qk_norm_scale =
1.0`, so the multiplicative gain is exactly the identity in fp32 ⇒
step-0 forward is byte-identical to 016's step-0 (max-abs-diff = 0.0
vs the 016 control — no tolerance needed).

Mirrors NormFormer's per-layer attention-output gains (Shleifer et
al. 2021, arXiv:2110.09456) applied to the QK-norm output. The
hypothesis: 016's WIN uses a single shared per-head scale; different
blocks may want different normalization strengths (e.g. shallow
blocks with broader attention vs deep blocks with sharper
attention). If 169-WIN > 016-WIN, depth-conditional is the binding
axis; if 169 ≈ 016, the per-head-shared scale is sufficient at 0.94M.

Wiring lives in `models/layers.py` (the `MultiHeadAttention.
use_qk_norm_depth` kwarg + `self.qk_norm_scale` registration + the
forward multiply after the norm+RoPE block + the MoA `extra_K`
multiply), threaded through `TransformerBlock.__init__` and read
by `MinimalLLM.__init__` in `models/llm.py`. Mutually exclusive
with `use_q_only_norm` / `use_k_only_norm` / `use_qk_norm_post_rope`
(asserted at MHA forward). Default off ⇒ baseline path bit-identical.

NB: imports `Tiny1M3MQKNormDepthConfig` directly rather than declaring
its own `class C(Tiny1M3MConfig): use_qk_norm_depth: bool = True`,
because the latter would NOT override the parent's dataclass field
default (the parent's `False` is inherited verbatim without a
`@dataclass` re-decoration on the subclass, so the annotation is
ignored and `C().use_qk_norm_depth` resolves to `False`). The
canonical `Tiny1M3MQKNormDepthConfig` is `@dataclass`-decorated in
`configs/llm_config.py`, inherits from `Tiny1M3MQKNormConfig`
(preserving `use_qk_layernorm=True` from the 016 WIN), and sets
`use_qk_norm_depth=True`.

Caches are at /root/universe-lm/_arq_169-qk-norm-depth.py on the box.
Run via the runner harness (see autoresearch/prompts/runner.md).
"""
from configs.llm_config import Tiny1M3MQKNormDepthConfig as C


if __name__ == "__main__":
    import sys
    import train_llm

    sys.modules["__main__"].C = C
    sys.argv = [
        "train_llm.py",
        "--config_class",
        "__main__.C",
        "--seed",
        "42",
        "--dataset_path",
        "processed_data/pretrain_1B",
        "--warmup",
        "false",
    ]
    train_llm.main()