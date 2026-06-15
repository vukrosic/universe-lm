"""Autoresearch 172 — trt: per-head learnable RoPE base frequency
(RoFormer / Su et al. 2024 arXiv:2104.09864, extended for head
specialization).

Each attention head learns a multiplicative scale on top of the
global `rope_base` (closed-winner 500000) for its rotary frequency
spectrum. The mechanism is already in `models/layers.py:1779-1781`
(init of `per_head_rope_log`) and `:2057-2060` (use:
`head_scale = exp(per_head_rope_log); freqs *= head_scale`); with
`per_head_rope_log = 0` at step 0, `head_scale = 1.0` for every head
and the forward is bit-identical to the `rope_base=500000` baseline
(max-abs-diff = 0.0). The optimizer then grows the per-head scale
during training. Total new params: n_heads × n_layers = 4 × 12 = 48
scalars (+0.005% of 0.94M).

A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4306
cached at 6.4394±0.04 for this box, see
`autoresearch/baseline-cache.json`). Daemon-owned ctrl is
`Tiny1M3MConfig` (rope_base=10000); trt is `rope_base=500000 +
use_per_head_rope_base=True` — isolates "trt (500k + per-head
learning) vs ctrl (10000)" so the run answers "is per-head frequency
learning a real axis beyond the closed-winner global base?"
"""
from configs.llm_config import Tiny1M3MPerHeadRopeConfig


class C(Tiny1M3MPerHeadRopeConfig):
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