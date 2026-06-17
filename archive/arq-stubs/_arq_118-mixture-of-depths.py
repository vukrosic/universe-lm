"""Autoresearch 118 — trt: Mixture-of-Depths (Raposo et al. 2024).

A/B vs the plain tiny1m3m baseline. Each transformer block gets a
`MoDRouter` (2-layer MLP) that scores every token; top-k tokens
(`k = mod_capacity · T`) get the block's residual update. Skipped tokens
pass through unchanged. W_1, W_2 zero-init ⇒ σ(0) = 0.5 uniform scores.
"""
import sys
from configs.llm_config import Tiny1M3MMixtureOfDepthsConfig


class C(Tiny1M3MMixtureOfDepthsConfig):
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
