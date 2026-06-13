"""Autoresearch 133 — trt: SeqMix: Token-Level Mixup for LM
(Guo, Mao, Zhang 2019, arXiv:1908.02951).

A/B vs the plain tiny1m3m baseline. Each step samples a second sequence
from the batch, mixes the two sequences at the embedding level via
`λ · x_a + (1−λ) · x_b` with `λ ~ Beta(α, α)`, and trains on the mixed
input with `λ`-weighted CE against both targets. The `use_seqmix` flag
is on the base `Tiny1M3MConfig`; we flip it True here.
"""
import sys
from configs.llm_config import Tiny1M3MConfig


class C(Tiny1M3MConfig):
    use_seqmix: bool = True
    seqmix_alpha: float = 0.4


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
