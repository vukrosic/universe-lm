"""Autoresearch 143 — trt: pre-attention ShortConv (Hyena ShortConv variant).

Poli, Massaroli, et al. 2023, "Hyena Hierarchy: Towards Larger
Convolutional Language Models" (arXiv:2302.10866) — ShortConv variant
(single depthwise Conv1d, kernel 3 or 4, pre-attention local aggregator).

A/B vs the plain tiny1m3m baseline. Adds one depthwise causal Conv1d
per transformer block on the residual stream, applied pre-attention,
gated by a per-block scalar `g` init 0. At step 0 `g = 0` so
`x + g·ShortConv1D(x) = x` — bit-identical to the no-conv baseline.
The conv's internal weight init is identity (last tap = 1, rest = 0)
so the conv is `x → x` at step 0 too — a "pass-through" starting
structure the gate smoothly grows into as training proceeds.

Strictly orthogonal to 023-canon-conv (which is post-attention concat
with kaiming init) and to all attention-side levers. Default off in
`Tiny1M3MConfig`; on via `Tiny1M3MShortConvConfig` subclass.
"""
import sys
from configs.llm_config import Tiny1M3MShortConvConfig


class C(Tiny1M3MShortConvConfig):
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
