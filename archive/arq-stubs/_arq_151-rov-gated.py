"""Autoresearch 151 — trt: Gated Rotary Value Embeddings (RoV)
(Su et al. 2024, Hunyuan-DiT arXiv:2403.13257 §2.3 / RoV for ViT
arXiv:2407.07282).

A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). Applies the
same rotary position embedding already used on Q,K to the value
vector V as well, mixed via a per-block scalar gate
`rov_gate = nn.Parameter(torch.zeros(1))`. Init 0 ⇒
`V_combined = V + 0·V_rot = V` ⇒ step-0 forward graph bit-identical
to baseline (within fp32 rounding noise of one extra rotary call +
one elementwise add per block per forward). The model has to *learn*
to mix the position-aware V back in during training. Cost: 1
scalar/block (12 at tiny1m3m) + one extra rotary call per block per
forward.
"""
from configs.llm_config import Tiny1M3MRoVGatedConfig


class C(Tiny1M3MRoVGatedConfig):
    pass


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