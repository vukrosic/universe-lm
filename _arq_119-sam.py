"""Autoresearch 119 — trt: SAM Sharpness-Aware Minimization (Foret et al. 2020).

A/B vs the plain tiny1m3m baseline. Wraps the AdamW path (1-D / embedding /
norm) with `AdamSAM`: on every step, do an adversarial ascent to `w + ε̂`
(ε̂ = ρ · ∇L(w) / ‖∇L(w)‖), re-run a forward+backward at the perturbed
point, then apply AdamW to the perturbed-point gradient. The 2-D Muon
path is unchanged. `sam_rho=0.05` paper default.
"""
import sys
from configs.llm_config import Tiny1M3MSAMConfig


class C(Tiny1M3MSAMConfig):
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
