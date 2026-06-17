#!/usr/bin/env python
"""Bootstrap for 168-av-output-carry. Use the canonical
`Tiny1M3MAVOutputCarryConfig` (defined in `configs/llm_config.py`
with `@dataclass` so `use_av_output_carry=True` properly overrides
the parent's default — the dataclass-inheritance pitfall documented
in `_arq_161-dyt-temp.py`).

Cross-block AV-output carry: for each block l >= 1, augment the
post-SDPA / post-merge-reshape / pre-W_O attention output with a
learnable α_l-scaled carry from the previous block's same-stage
tensor. α_l init 0 ⇒ step-0 forward is bit-identical to the no-
carry baseline (within fp32 rounding noise of one extra
multiply-add).

This is the third axis of the cross-block carry family: 021 carries
V (pre-AV), 164 carries Q (pre-AV), 168 carries AV-output (post-AV,
pre-W_O). Distinct from 116-hyper-connections (residual-stream mix,
post-block) and 150-xlayer-feedback (full cross-block attention,
rejected). The wiring lives in `models/layers.py`
(`MultiHeadAttention.use_av_output_carry` + `_av_carry` stash +
post-merge-reshape carry branch) and `models/llm.py`
(`MinimalLLM.use_av_output_carry` + the `_run_post_embed` stash/
loop plumbing). Default off ⇒ baseline path bit-identical.

Caches are at /root/universe-lm/_arq_168-av-output-carry.py on the
box. Run via the runner harness (see autoresearch/prompts/runner.md).
"""
from configs.llm_config import Tiny1M3MAVOutputCarryConfig as C


if __name__ == "__main__":
    import sys
    import train_llm

    sys.modules["__main__"].C = C
    sys.argv = [
        "train_llm.py",
        "--config_class",
        "__main__.C",
        "--seed",
        "42",
        "--dataset_path",
        "processed_data/pretrain_1B",
        "--warmup",
        "false",
    ]
    train_llm.main()
