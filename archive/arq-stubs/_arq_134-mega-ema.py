"""Autoresearch 134 — trt: Mega EMA on V (Ma et al. 2022, arXiv:2209.10655).

A/B vs the plain tiny1m3m baseline. The V stream is concatenated with
`V_ema = W_V @ u` where `u_t = β·u_{t-1} + (1-β)·x_t` is a per-channel
EMA over the input residual stream. β ∈ [0, 1] is parametrized via
sigmoid of a learnable per-channel scalar (init 0 ⇒ β=0.5 at step 0).
The concat doubles the V stream and the head reshape treats it as
2·n_kv_heads heads (asserted == n_heads at tiny1m3m). NOT byte-identical
to baseline at step 0 — the EMA is half-smoothed and the V stream is
doubled.
"""
import sys
from configs.llm_config import Tiny1M3MMegaConfig


class C(Tiny1M3MMegaConfig):
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
