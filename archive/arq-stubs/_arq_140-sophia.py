"""Autoresearch 140 — trt: Sophia second-order optimizer
(Liu, Wang, et al. 2023, arXiv:2305.14342, ICML 2023).

A/B vs the plain tiny1m3m baseline. Replaces the AdamW 1-D / embedding /
norm / head path with `Sophia`: diagonal-Hessian-aware update
`update = clip(g, ±ρ) / max(h, ε)` where the diagonal Hessian `h` is
sampled every k=10 steps via Hutchinson's trace estimator (one extra
backward on a `g·u` scalar where `u ~ Rademacher(±1)`). The 2-D Muon
path is unchanged. Defaults: lr=6e-3, β1=0.965, β2=0.99, ρ=0.04,
k=10, update_clip=1.0 (cold-start safety guard on the `h≈0`
amplification). The trainer's Hutchinson block fires when
`Sophia._step_count % k == 0`; with k=10 and 92 update steps at
tiny1m3m, the diagonal Hessian is refreshed ~9 times, matching the
paper's amortization (~1.1× backward cost).
"""
import sys
from configs.llm_config import Tiny1M3MSophiaConfig


class C(Tiny1M3MSophiaConfig):
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
