"""Autoresearch 150 — trt: Cross-Layer Feedback Attention
(Holtzman et al. 2020, arXiv:2002.09402; lean "previous K=2 layers"
variant).

A/B vs the plain tiny1m3m baseline (`_arq_150-xlayer-feedback_ctrl.py`).
Each block reads from a cache of the previous K=2 blocks' pre-FFN
residual states via a small `XLayerCrossAttn` head (1 head, head_dim=16,
Q/K/V all d_model → 16, out 16 → d_model), and adds the result as a
*gated* residual branch. Per-block scalar `xlayer_gate` is init 0 ⇒
contribution is exactly 0 at step 0 ⇒ forward is bit-identical to the
no-feedback baseline. See `autoresearch/ideas/150-xlayer-feedback/idea.md`.
"""
import sys
from configs.llm_config import Tiny1M3MXLayerFeedbackConfig


class C(Tiny1M3MXLayerFeedbackConfig):
    pass


if __name__ == "__main__":
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
