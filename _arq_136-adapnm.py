"""Autoresearch 136 — trt: AdaPNM: Adaptive Positive-Negative Momentum
(Ding, Zhou, Zhu, Ye, Jiao 2019, arXiv:1906.01520, NeurIPS 2019).

A/B vs the plain tiny1m3m baseline. Replaces the AdamW 1-D / embedding /
norm / head path with `AdaPNM`: a thin AdamW subclass that maintains
TWO parallel momentum buffers — one for the positive gradient part
and one for the negative — and combines them as `m+ − m−` (which is
algebraically equal to the standard EMA `m_t` since
`max(g,0) − max(-g,0) = g` element-wise). The 2-D Muon path is
unchanged. Shared `v` (Adam's second-moment estimator), decoupled WD.
"""
import sys
from configs.llm_config import Tiny1M3MAdaPNMConfig


class C(Tiny1M3MAdaPNMConfig):
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
