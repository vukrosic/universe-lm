"""Autoresearch 015 — Moonlight Muon RMS rescale."""
from configs.llm_config import Tiny1M3MConfig


class C(Tiny1M3MConfig):
    use_moonlight_muon: bool = True
    moonlight_muon_c: float = 0.2


if __name__ == "__main__":
    import sys
    import train_llm
    # Inject this module's C as the active config so train_llm.main()
    # picks it up via --config_class __main__.C.
    sys.modules["__main__"].C = C
    sys.argv = [
        "train_llm.py",
        "--config_class", "__main__.C",
        "--seed", "42",
        "--dataset_path", "processed_data/pretrain_1B",
        "--warmup", "false",
    ]
    train_llm.main()
