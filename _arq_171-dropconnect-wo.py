"""Autoresearch 171 — trt: DropConnect on W_O (Wan, Zeiler, Zhang,
LeCun, Fergus, "Regularization of Neural Networks using DropConnect",
ICML 2013, arXiv:1304.3174).

Per-weight Bernoulli mask on the attention output projection W_O
during training. At each training forward, every block's
`MultiHeadAttention` samples a fresh mask `M ∈ {0,1}^{d_model ×
d_model}` with `M_ij ~ Bernoulli(1 - p)` and uses `W_O_masked = W_O
⊙ M / (1 - p)` (inverted-dropout rescale, matches `F.dropout` and
the 147-DropKey sibling's rescale). The mask is sampled per forward
pass and shared across all batch elements and positions (one mask
per layer per call, NOT per-token).

**Warmup ramp.** The effective rate is linearly ramped from 0.0 →
0.05 over the first 100 training forwards (the
`dropconnect_wo_warmup_steps` schedule). At step 0 the effective
rate is 0.0 ⇒ the mask branch is short-circuited before any RNG is
consumed ⇒ the trt forward is **bit-identical to baseline at step 0**
(max-abs-diff = 0.0 across the full forward, no parameter modified,
no RNG consumed). At step 100 the effective rate reaches 0.05 and
holds there for the remaining ~92 steps.

Distinct from the closed regularizer family at 0.94M:
- 147-dropkey (per-token K-mask, NULL at 0.94M): signal-space
  regularizer; Q's gradient is computed against a clean K-derivative
  so the optimizer can absorb the noise into Q's update.
- 111-drop-path (per-branch residual mask, DRIFT/NULL): surviving
  branches still receive clean gradients; optimizer up-weights them.
- 115-rdrop (KL-divergence loss regularizer, NULL): loss-shape, not
  stochastic masking.

DropConnect on W_O is a *parameter-space* regularizer: zeroing
`W_O[i,j]` also zeros that weight's gradient, so the parameter
receives a smaller-magnitude update AND the gradient tells the
optimizer to up-weight surviving paths in W_O. The optimizer cannot
route around the noise by adjusting a *different* parameter. W_O is
a single dense `d_model × d_model = 64 × 64 = 4096` matrix at
tiny1m3m; weight masking forces redundant paths and prevents
co-adaptation onto a narrow subspace over the 3M-token training
horizon.

A/B vs the plain tiny1m3m baseline (`Tiny1M3MConfig`, val 6.4216
cached at 6.4394±0.04 / 6.4504±0.0558 for this box — see
`autoresearch/baseline-cache.json`). NULL band |Δ| < 0.04. PASS ≤
ctrl − 0.020. DRIFT > +0.020.

Wiring lives in `models/layers.py`
(`MultiHeadAttention.use_dropconnect_wo` + `self.dropconnect_wo_rate`
+ `self.dropconnect_wo_warmup_steps` + `self._dc_step_count` stored
at construction; the warmup-ramped mask-and-rescale branch at the
W_O application site in `MultiHeadAttention.forward`), threaded
through `TransformerBlock.__init__` and read by `MinimalLLM.__init__`
in `models/llm.py`. The flag is read at MHA forward via the standard
boolean-gated pattern
(`self.use_dropconnect_wo and self.training`), with the effective
rate computed as
`dropconnect_wo_rate * min(step, warmup) / warmup`. Eval mode and
`rate=0.0` (and step=0) all skip the branch ⇒ baseline path
bit-identical at step 0. See
`autoresearch/ideas/171-dropconnect-wo/idea.md`.
"""
from configs.llm_config import Tiny1M3MDropConnectWOConfig


class C(Tiny1M3MDropConnectWOConfig):
    pass


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
