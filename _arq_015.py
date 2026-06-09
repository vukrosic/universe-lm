"""Autoresearch 015 — Moonlight Muon RMS rescale."""
import sys
from configs.llm_config import Tiny1M3MConfig


class C(Tiny1M3MConfig):
    use_moonlight_muon: bool = True
    moonlight_muon_c: float = 0.2


if __name__ == "__main__":
    # Run train_llm.py with the subclass as the active config.
    import subprocess
    rc = subprocess.call([
        sys.executable, "train_llm.py",
        "--config_class", "__main__.C",
        "--seed", "42",
        "--dataset_path", "processed_data/pretrain_1B",
        "--warmup", "false",
    ])
    sys.exit(rc)
