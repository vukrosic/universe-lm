"""Autoresearch 142 — trt: LayerScale (Touvron, Cord, et al. 2021,
"Going Deeper with Image Transformers", arXiv:2103.17239, ICCV 2021).

A/B vs the plain tiny1m3m baseline. Each block learns two per-channel
diagonal scales `gamma_attn`, `gamma_ffn ∈ R^{d_model}` that multiply
the residual branch: `x = x + gamma * sub_block(x)` (direct form, NOT
the reparam `(1+γ)` form used by the closed-#21 `use_layerscale` flag).
Init `gamma = 1e-4 * ones(d_model)` (paper default) → at step 0 the
residual contribution is `1e-4 × sub_block(x)`, four orders of
magnitude smaller than the residual stream magnitude, so the val loss
at step 0 is within fp32 noise of baseline. The per-channel
selectivity is qualitatively different from the scalar ReZero (130,
NULL at 12L) and the whole-residual Sub-LN (017, NULL at 12L) — both
prior depth-conditional levers null at 12L, but LayerScale's
per-channel diagonal is a structurally different mechanism that has
not been tested in this pipeline. Cost: 2 × d_model = 128 extra
params at tiny1m3m (negligible).

Transfer-risk: med — paper's headline wins are at depth ≥ 50 in the
original; independent replications report small gains (0.1–0.3%) on
shallow LMs. NULL band |Δ| < 0.01. DRIFT > +0.01. PASS ≤ −0.01.
See `autoresearch/ideas/142-layerscale/idea.md`.
"""
import sys
from configs.llm_config import Tiny1M3MLayerScaleConfig


class C(Tiny1M3MLayerScaleConfig):
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
