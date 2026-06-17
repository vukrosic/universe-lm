#!/usr/bin/env python
"""Bootstrap for 157-conv-ffn. Subclass Tiny1M3MConfig with use_conv_ffn=True."""
from configs.llm_config import Tiny1M3MConfig


class C(Tiny1M3MConfig):
    use_conv_ffn: bool = True
    conv_ffn_kernel: int = 3


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
