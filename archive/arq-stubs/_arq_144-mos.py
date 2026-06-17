"""Autoresearch 144 — trt: Mixture of Softmaxes (MoS) output head
(Yang, Chen, et al. 2017, arXiv:1711.03953).

A/B vs the plain tiny1m3m baseline (`_arq_144-mos_ctrl.py`). Replaces
the single output softmax with `n_mos_components=4` parallel
vocab-sized heads mixed by per-token `π = softmax(W_π · h)`. At step
0 the mix projection is initialized to a one-hot (`W_π.bias =
[+1e4, -1e4, -1e4, -1e4]`) so the mixture reduces to `softmax(W_0 ·
h)` — bit-identical to the standard tied head. See
`autoresearch/ideas/144-mos/idea.md`.
"""
import sys
from configs.llm_config import Tiny1M3MMoSConfig


class C(Tiny1M3MMoSConfig):
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
