"""Autoresearch 146 — trt: Switch FFN (Fedus, Zoph, Shazeer 2022, arXiv:2101.03961).

A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). Replaces the
dense FFN with `SwitchFFN` — E parallel full-width FFNs (default E=4)
routed by a learned top-1 router. Identity at step 0: router zero-init
⇒ all tokens route to expert 0 ⇒ output = expert_0(x) = standard FFN.
"""
from configs.llm_config import Tiny1M3MConfig


class C(Tiny1M3MConfig):
    use_switch_ffn: bool = True
    n_ffn_experts: int = 4
    expert_capacity_factor: float = 1.25


if __name__ == "__main__":
    import sys
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