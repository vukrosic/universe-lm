"""Autoresearch 130 — trt: ReZero residual scaling (Bachlechner et al. 2020,
arXiv:2003.04887).

A/B vs the plain tiny1m3m baseline. Each block builds two learnable
scalars `α_attn` and `α_ffn` (init 0) that gate the residual add
`x = x + α·f(x)`. At step 0 α=0 ⇒ the model is the identity
function (baseline forward graph unchanged because the α·f term is
exactly 0). The optimizer grows α during training; the lever is
whether layer-specific residual scaling helps at 12L.

Transfer-risk: high. ReZero's paper wins are at 100L; at 12L the
deep-network init pathology is mild and the slow α ramp-up may cost
more than it saves. The high transfer-risk is honest about the depth
mismatch. See `autoresearch/ideas/130-rezero/idea.md`.
"""
import sys
from configs.llm_config import Tiny1M3MReZeroConfig


class C(Tiny1M3MReZeroConfig):
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
