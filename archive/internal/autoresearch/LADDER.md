# The release ladder — scaling-law architecture search toward a 135M model that beats SmolLM2-135M

## Goal
Find the **architecture** that, trained at the 135M / 2.7B-token release budget,
beats a SmolLM2-135M-class baseline — and prove it *before* spending the
multi-day full run, by **extrapolating a scaling law** fit on cheap small models.

We do **not** train 2.7B tokens per experiment (infeasible — days/run). Instead:

```
broad cheap SCREEN  ->  carry survivors up the LADDER  ->  fit L(N) -> extrapolate  ->  ONE full 135M run
(tiny 1M, ~3x tok)      (same arch at 4 sizes, 20x tok)    (predict target loss)        (the winner only)
```

## Why scaling laws, not direct 135M
For a power law `L(N) = E + A·N^(-α)`, the **log(N) lever arm** — how many orders
of magnitude the rungs span — is what pins the exponent and the extrapolation.
Loss depends mainly on **non-embedding N** and only weakly on shape
(Kaplan et al. 2020), so we **vary shape freely to span ~1.9 decades of N
log-uniformly** rather than holding shape rigid (which clusters the rungs and
wastes the lever arm). The research prize is an architecture with a **steeper α**
— an advantage that *grows* with scale — not just a lower loss at one small size.
Most tiny-tier wins share the baseline's α (a constant intercept shift) and wash
out at scale; finding the few that bend α is the point.

## The rungs (in `configs/llm_config.py`, counts verified by build)
Same architecture FAMILY as the target (`Full135M2700MConfig`): full tied
embeddings, head_dim 64, `d_ff = 4·d_model`, RoPE + RMSNorm + squared-ReLU +
Muon, Chinchilla 20×-total-params tokens. Shape varies to spread N.

| config | d_model | layers | heads/kv | total | **non-embed N** | log₁₀N | tokens | runs on |
|---|---|---|---|---|---|---|---|---|
| `Ladder8M155MConfig`   | 128 | 8  | 2/1 | 7.7M  | 1.45M  | 6.16 | 155M | your box |
| `Ladder13M252MConfig`  | 192 | 8  | 3/1 | 12.6M | 3.17M  | 6.50 | 252M | your box |
| `Ladder23M469MConfig`  | 256 | 15 | 4/2 | 23.5M | 10.9M  | 7.04 | 469M | your box |
| `Ladder52M1042MConfig` | 384 | 21 | 6/2 | 52.1M | 33.2M  | 7.52 | 1.04B | contributor GPU |
| `Full135M2700MConfig`  | 576 | 30 | 9/3 | 135M  | 106.8M | 8.03 | 2.70B | contributor GPU (release) |

The cheap low end extends the lever arm for almost free; the expensive points
cluster near the target. Three rungs run locally; the top two need the
**distributed contributor system** — that's where many-people-contributing earns
its keep.

## What to run on the ladder (carry-architectures)
Train each of these *at every rung* (same config flags, scaled size):
1. **`baseline`** — the plain target architecture (no extra levers). The control.
2. **`deepnet`** — the tiny champion's **clean** structural carry: DeepNet-α
   residual init (`use_deepnet_alpha`). It's an init/stability mechanism, not
   positional — it does **not** touch attention range, so it's long-context-safe
   under D002. (The 323 optimizer knobs — momentum/LR — are per-tier engineering,
   NOT carried; re-tune per rung. Per EXPERIMENT-DESIGN RULE 0 only structural
   levers are carried.)
3. **Long-context-safe novel levers** from `autoresearch/LONG-CONTEXT-IDEAS.md`
   — start with **RoPE base/θ scaling** and **QK-norm**, then **differential
   attention** (flags already wired). These are next, *after* baseline+deepnet
   produce the first scaling points.

> **CUT: the `champion`/`polyalibi` arm.** The tiny champion's biggest lever is
> alibi → poly-alibi, which lowers loss precisely by **punishing distant
> attention** — banned by `DECISIONS.jsonl` **D002** (long context is a
> first-class release objective; no distance-penalty mechanisms qualify on loss
> alone). The tiny loss-game and the release-capability goal diverge here, so the
> alibi family does not ride up this ladder.

A lever only earns a 135M slot if its fitted curve sits **below** `baseline` at
the target N — ideally via a steeper α, not just a lower intercept — **and**
clears the long-range eval (D002): a loss win that degrades long context is
disqualified, not promoted.

## How to run a rung and log the result
Each `(arm × rung)` is an **`_arq` stub** that subclasses the rung config and
turns on exactly the levers for that arm — the same idiom as the tiny tier. Use
a stub, NOT `run_experiment.py` directly: on the operator machine
`run_experiment.py` auto-folds the *tiny* champion (its structural levers **and**
its tiny-tuned optimizer LRs/momentum) into every run via `champion.json`, and
the merge can't cleanly clear those keys. The stub's direct `train_llm` invoke
bypasses that, giving a clean, fully-explicit config.

`baseline` arm (plain target architecture, no levers) — `_arq_ladder8m_baseline.py`:
```python
from configs.llm_config import Ladder8M155MConfig
class C(Ladder8M155MConfig):
    pass
if __name__ == "__main__":
    import sys, train_llm
    sys.modules["__main__"].C = C
    sys.argv = ["train_llm.py", "--config_class", "__main__.C", "--seed", "42",
                "--dataset_path", "processed_data/pretrain_1B", "--warmup", "false"]
    train_llm.main()
```

`deepnet` arm — same, but the subclass must be a **re-decorated** `@dataclass`
with a typed field default (a plain subclass body silently no-ops on a dataclass
field — verified):
```python
import dataclasses
from configs.llm_config import Ladder8M155MConfig
@dataclasses.dataclass
class C(Ladder8M155MConfig):
    use_deepnet_alpha: bool = True
```
Do **not** carry the tiny optimizer knobs (muon_momentum 0.9 / ×2 LR) — those are
per-tier engineering, re-tuned per rung, not the architecture axis (per
`EXPERIMENT-DESIGN.md` RULE 0). All arms at a given rung must share the SAME
optimizer settings; only the structural lever differs. (RoPE has no toggle — it's
always on; long-context-safe levers like RoPE-base/QK-norm/diff-attn add to this
arch, they don't replace attention range.)

Then append one JSONL line per run to `autoresearch/ladder/results.jsonl`
(`N` = **non-embedding** params for that rung — see the table):

```json
{"arch": "deepnet", "N": 1450000, "tokens": 155000000, "val_loss": 4.91, "seed": 42, "rung": "Ladder8M155MConfig"}
```

## Fit and extrapolate
```bash
python3 autoresearch/bin/scaling_fit.py --baseline baseline
```
Per architecture it fits `L = E + A·N^(-α)`, prints α / R² / the predicted loss
at the 135M target N (106.8M non-embed) with a 95% bootstrap CI (needs ≥4 rungs),
and ranks architectures by predicted target loss. `--selftest` verifies the
fitter on synthetic data.

**Caveat:** with 4 rungs, `E` and `α` trade off individually (under-determined),
though the prediction at/near the target is reliable. Adding rungs or widening
the span tightens the `E`-vs-`α` split and shrinks the CI.

## Success bar (budget-matched, falsifiable)
- **Primary:** val loss at the 135M / 2.7B point vs a SmolLM2-135M-arch baseline
  trained on the **same 2.7B tokens**. (Absolute parity with the real
  SmolLM2-135M, ~2T tokens, is out of scope — ~1000× our budget.)
- **Release check:** a light downstream eval (HellaSwag / ARC) at the matched budget.

## Distributed contributors
Every rung run is logged as `(arch, N, tokens, val_loss, seed)` — the shared
ladder dataset *is* the research asset. Contributors run the heavy rungs (52M,
135M) on their own GPUs and append to the same dataset; the scaling fit pools
everyone's points. voidbase should expose a ladder-results endpoint mirroring
`results.jsonl` (and gain a `params`/`N` column — it currently logs `val_loss`
and `tokens_seen` but not N).
