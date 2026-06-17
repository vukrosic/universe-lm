"""Autoresearch 175 — trt: Learnable per-head ALiBi-style linear-distance
bias on attention scores (Press, Smith, Lewis, ICLR 2022,
arXiv:2108.12409; validated at BLOOM-176B by BigScience 2022).

The original ALiBi paper uses *fixed* per-head slopes from a geometric
sequence `1/2^(8k/H)`. 175 makes the slopes *learnable* per head
(48 scalars total at 0.94M/12L/4H).

A learnable `m_h ∈ R^H` (n_heads=4 at tiny1m3m) is added to the
attention scores pre-softmax as `score[b,h,t,s] -= m_h · (t − s)`
(for s ≤ t — the causal mask handles the rest).

Init `m_h = 0` ⇒ `softmax(scores − 0·(t−s)) = softmax(scores)`
byte-for-byte at step 0. Forces the manual attention path so SDPA's
flash / efficient backends don't perturb step-0 numerics.

A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`,
val mean 6.4447 ± 0.0244). Expected Δval ∈ [-0.005, -0.025]
(per-head linear-distance decay is a fresh axis in the
locality-prior family that already wins at this tier: 009-FIRE,
154-rebased-attn, 023-canon-conv, 143-shortconv). NULL band |Δ| <
0.02 expected. PASS ≤ ctrl − 0.02.
"""
from configs.llm_config import Tiny1M3MAlibiConfig


class C(Tiny1M3MAlibiConfig):
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
