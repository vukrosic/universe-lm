"""Autoresearch 110 — trt: Polyak-Ruppert Weight EMA (Polyak 1990).

A/B vs the plain tiny1m3m baseline. Maintains a shadow copy of every
trainable parameter and updates it after each optimizer step as
`θ_ema ← μ·θ_ema + (1−μ)·θ_live`, with `μ` ramping linearly from 0
to 0.999 over the first 100 steps. At step 0 `μ=0` ⇒ `θ_ema = θ_live`,
so the val score at step 0 is bit-identical to the baseline. The
shadow is swapped into the live model before `evaluate_model(...)`
and restored on exit; training and checkpointing stay on the live
trajectory (the lever is the eval point, not the model itself).
"""
import sys
from configs.llm_config import Tiny1M3MEMAConfig


class C(Tiny1M3MEMAConfig):
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
