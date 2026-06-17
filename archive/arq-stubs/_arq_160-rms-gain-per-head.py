"""Autoresearch 160 — trt: Per-head RMS gain on the attention value output
(Gemma 2 / Qwen 2.5 pattern). A learnable `g_h ∈ R^H` (n_heads=4 at
tiny1m3m) multiplies each head's AV-aggregated output before the O
projection, so each head controls the *magnitude* of its contribution
to the residual stream without changing its direction.

Init `g_h = 1` ⇒ `o_h *= 1 = o_h` byte-for-byte at step 0 (baseline
behavior). Applied BEFORE the existing per-head gates so it composes
cleanly with `use_attn_output_gate`, `use_attn_output_channel_gate`,
`use_gated_attn`, etc. (they multiply through).

A/B vs the plain tiny1m3m baseline (Tiny1M3MConfig, val 6.4306).
NULL band |Δ| < 0.005 expected (predicted mathematical null — the
post-AV magnitude axis is plausibly redundant given W_O).
"""
from configs.llm_config import Tiny1M3MHeadGainConfig


class C(Tiny1M3MHeadGainConfig):
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
