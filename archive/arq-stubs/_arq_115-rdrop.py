"""Autoresearch 115 — trt: R-Drop: Regularized Dropout (Liang et al. 2021).

A/B vs the plain tiny1m3m baseline. For every train step, run the model
forward **twice** with different dropout masks, take the mean of the two
next-token CE losses, and add a symmetric KL penalty
`α · 0.5·(KL(p_1‖p_2)+KL(p_2‖p_1))` to regularize the model toward
dropout-invariant logits. `α` is linearly warmed from 0 → 1.0 over the
first 1000 steps so step-0 is bit-identical to the single-CE baseline
(modulo the doubled forward, which is runtime not math).
"""
import sys
from configs.llm_config import Tiny1M3MRDropConfig


class C(Tiny1M3MRDropConfig):
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
