"""Autoresearch 177 — Talking-Heads Attention (Shazeer et al. 2020,
arXiv:2003.02436).

Cross-head H×H linear mix on **both** axes:
- pre-softmax attention scores (`talking_heads_M`)
- post-softmax attention output (`talking_heads_out_M`)

Both M matrices are `nn.Parameter(torch.eye(n_heads, n_heads))` ⇒
literal `M @ x = x` and `M_out @ x = x` ⇒ forward is byte-identical
to the bare baseline at step 0 (max-abs-diff = 0.0).

A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`,
val ≈ 6.4306 cached). WIN ≤ -0.025, NULL |Δ| < 0.015, DRIFT
|Δ| > 0.04 wrong-sign. See `autoresearch/ideas/177-talking-heads/`.
"""
from configs.llm_config import Tiny1M3MTalkingHeadsConfig


class C(Tiny1M3MTalkingHeadsConfig):
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
