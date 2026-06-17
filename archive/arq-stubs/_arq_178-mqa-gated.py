#!/usr/bin/env python
"""Bootstrap for 178-mqa-gated. Subclass Tiny1M3MConfig with use_mqa_gated=True.

Gated Multi-Query Attention: per-KV-head scalar gate β_k, β_v blends
between the head-local K, V projection and a single shared K, V
projection. β init 0 ⇒ step-0 forward is byte-identical to the
no-flag baseline.

Caches are at /root/universe-lm/_arq_178-mqa-gated.py on the box.
Run via the runner harness (see autoresearch/prompts/runner.md).
"""
from configs.llm_config import Tiny1M3MMQAGatedConfig


class C(Tiny1M3MMQAGatedConfig):
    pass


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
