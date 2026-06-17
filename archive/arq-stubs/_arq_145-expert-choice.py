"""Autoresearch 145 — trt: Expert-Choice MoE FFN replacement
(Zhou, Lei, et al. 2022, arXiv:2202.09368,
"Mixture-of-Experts with Expert Choice Routing").

A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`). Replaces the
dense FFN with `ExpertChoiceMoE` — E parallel full-width FFNs (default
E=4) where each expert picks its own top-k tokens (k = ceil(N/E))
instead of each token picking its top-k experts. Load balance is by
construction (every expert processes exactly k tokens) so NO
auxiliary load-balancing loss is required. Identity at step 0: router
zero-init ⇒ all expert-token scores are 0 ⇒ every expert processes
the same set of k tokens with uniform softmax weights ⇒ output ≈
uniform mean of E identically-init'd FFNs (NOT byte-identical to a
single FFN at step 0 — documented caveat mirroring 117-soft-moe).
"""
from configs.llm_config import Tiny1M3MExpertChoiceConfig


class C(Tiny1M3MExpertChoiceConfig):
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
