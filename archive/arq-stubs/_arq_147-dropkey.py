"""Autoresearch 147 — trt: DropKey (Xu, Zhao et al. 2022, arXiv:2207.01058).

A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). Replaces the
standard attention with a DropKey variant: per-head, per-token
Bernoulli mask on the K tensor before the QKᵀ dot product. Mask
shape `[B, n_heads, T, 1]`, elements drawn i.i.d. Bernoulli(1 - p)
with inverted-dropout rescale `K ← K * M / (1-p)`. Identity at step
0: `use_drop_key=False` (default) skips the branch entirely, so the
forward graph is bit-identical to the no-DropKey baseline. With
the flag on and `drop_key_rate=0.0`, the mask is all-ones and
`K = K * 1 / 1 = K` — also bit-identical.
"""
from configs.llm_config import Tiny1M3MConfig


class C(Tiny1M3MConfig):
    use_drop_key: bool = True
    drop_key_rate: float = 0.1


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
