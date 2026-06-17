"""Autoresearch 116 — trt: Hyper-Connections / Multi-Stream Residual (Xie et al. 2024).

A/B vs the plain tiny1m3m baseline. Splits the residual stream into
`hc_n_resid=4` parallel streams of width `d_l = 16` each, with per-position
(A_l, B_l, C_l) ∈ R^{4×4} mixing matrices. Identity init ⇒ step-0 forward
graph is bit-identical to the pre-norm residual path.
"""
import sys
from configs.llm_config import Tiny1M3MHyperConnectionsConfig


class C(Tiny1M3MHyperConnectionsConfig):
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
