"""Autoresearch 149 — trt: TTT-Linear FFN replacement (Sun, Yang,
et al. 2024, arXiv:2407.04620, §3.2).

A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). Replaces the
FFN's up-projection with `TTTLinear` — a per-input closed-form fast-
weight linear that updates its own weight from the input on the fly
via a single Newton-style gradient step on the auto-encoding loss
`||W·x − x||²`. The down-projection stays a standard `nn.Linear` so
the FFN output side is unchanged. Identity at step 0: `ttt_lr=0.0`
(default) ⇒ the `TTTLinear` short-circuits to `F.linear(x, weight, b)`
with the same `kaiming_uniform_` weight as `nn.Linear` ⇒ the FFN is
bit-identical to a vanilla `SquaredReLUFeedForward` at step 0.
"""
from configs.llm_config import Tiny1M3MTTTLinearConfig


class C(Tiny1M3MTTTLinearConfig):
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
