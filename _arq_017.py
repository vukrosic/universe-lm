"""Autoresearch 017 — Sub-LN / Sandwich block (LN_post on each sublayer output)."""
import sys
from configs.llm_config import Tiny1M3MConfig


class C(Tiny1M3MConfig):
    use_sub_ln: bool = True


if __name__ == "__main__":
    import subprocess
    rc = subprocess.call([
        sys.executable, "train_llm.py",
        "--config_class", "__main__.C",
        "--seed", "42",
        "--dataset_path", "processed_data/pretrain_1B",
        "--warmup", "false",
    ])
    sys.exit(rc)
