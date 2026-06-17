"""Autoresearch 148 — trt: Focal Modulation Networks (Yang et al. 2022,
arXiv:2203.11926, NeurIPS 2022).

A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). Replaces the
attention sub-block with `FocalModulationBlock` — hierarchical
depthwise-conv context aggregation + gated linear modulation. At step
0 the gate sigmoid is init to 0 (`gather` and `h_proj` are zero-init)
⇒ modulation contribution is exactly 0 ⇒ output = x bit-identical to
MHA path.
"""
from configs.llm_config import Tiny1M3MConfig


class C(Tiny1M3MConfig):
    use_focal_mod: bool = True
    focal_mod_kernels: tuple = (3, 5, 7)


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
