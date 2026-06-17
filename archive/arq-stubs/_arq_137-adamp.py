"""Autoresearch 137 — trt: AdamP (Adam with Projection-Based Update).

A/B vs the plain tiny1m3m baseline. Replaces the AdamW 1-D / embedding
/ norm / head path with `AdamP` (He et al. 2020, arXiv:2006.08217).
Before each Adam step, projects the Adam update Δ = m̂/√v̂ onto the
orthogonal complement of w (subtracts (Δ·w / ‖w‖²)·w), so the update
no longer rotates the weight — only scales its magnitude. The L2 reg
is applied as the paper's λ·‖w‖·ŵ (magnitude shrinkage, no rotation).

Identity at step 0: for symmetric inits the projection removes an
O(1/√d) component of Δ_0, so the first AdamP step ≈ the first AdamW
step modulo that small correction. With `use_adamp=False` (default)
plain `torch.optim.AdamW` is used — baseline bit-identical. The 2-D
Muon path is unchanged.
"""
import sys
from configs.llm_config import Tiny1M3MAdamPConfig


class C(Tiny1M3MAdamPConfig):
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