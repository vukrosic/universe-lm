"""Autoresearch 117 — trt: Soft MoE FFN replacement (Puigcerver et al. 2024).

A/B vs the plain tiny1m3m baseline. Replaces the standard dense FFN with
`SoftMoEFFN` (E=4 parallel narrower FFNs + softmax dispatch/combine).
Each expert has width `d_ff / 4 = 64`. Dispatch/combine zero-init ⇒
uniform softmaxes at step 0. Fully differentiable: no top-k, no
balancing loss.
"""
import sys
from configs.llm_config import Tiny1M3MSoftMoEConfig


class C(Tiny1M3MSoftMoEConfig):
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
