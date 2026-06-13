"""Autoresearch 150 — ctrl: plain tiny1m3m baseline
(`Tiny1M3MConfig`) — paired with `_arq_150-xlayer-feedback.py` to form
the A/B test for the cross-layer feedback attention lever. With
`use_xlayer_feedback=False` (default) the baseline is byte-identical
to the standard pre-norm 12L tiny1m3m path (no extra params, no
extra RNG). See `autoresearch/ideas/150-xlayer-feedback/idea.md`.
"""
import sys
from configs.llm_config import Tiny1M3MConfig


class C(Tiny1M3MConfig):
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
