"""Autoresearch 113 — trt: GaLore: Gradient Low-Rank Projection (Zhao et al. 2024).

A/B vs the plain tiny1m3m baseline. For each 2-D weight matrix, the
gradient is projected into a rank-4 subspace via orthonormal P, Q; AdamW
runs in the 4×4 projected space; the update is projected back. P, Q
refresh from the SVD of a running gradient EMA every 200 steps (paper
defaults). 1-D / embedding / norm stay on plain AdamW. The forward
graph is unchanged, so step-0 val_loss is bit-identical to baseline.
"""
import sys
from configs.llm_config import Tiny1M3MGaLoreConfig


class C(Tiny1M3MGaLoreConfig):
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
