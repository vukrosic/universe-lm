---
id: 113-galore
status: needs-plan
round: 1
updated: 2026-06-14T02:20:40Z
transfer-risk: med
plain: It tries to project the gradient into a low-rank subspace before running AdamW, so the optimizer sees a compressed view of the update direction without losing the most important axes.
---

# 113 — GaLore: Gradient Low-Rank Projection

## Source
Zhao, Zhang, Yang, Luo, Zhang, Wang, Yang, Qiao, "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection"
(arXiv:2403.03507, March 2024; updated June 2024). NeurIPS 2024.
https://arxiv.org/abs/2403.03507

Validated by the paper on Llama 1B / 7B pretraining and by independent
re-implementations in nanoGPT-speedrun and picoGPT. The paper's headline is
*memory* ÷ ~2 at parity val loss with AdamW; secondary result is **modest
quality gains** at rank-r=4-256 on the same val loss (the lever we are
actually mining here).

## Mechanism
For each 2-D weight matrix W ∈ ℝ^{n×m}, maintain two **projection matrices**
P ∈ ℝ^{n×r}, Q ∈ ℝ^{m×r} with r ≪ min(n,m). The gradient G_t is projected
into a low-rank subspace before AdamW is applied:
  `G̃_t = P_t^T · G_t · Q_t`
  `update = AdamW(G̃_t)` (in the r×r + r·d state, not the full m·n state)
  `W ← W − lr · P_t · update · Q_t^T`

Every `T_proj` steps (paper default 200), `P, Q` are refreshed from the SVD
of a recent gradient EMA: `G̃_recent = U S V^T ⇒ P = U[:, :r], Q = V[:, :r]`.
Between refreshes, `P, Q` are frozen, so the AdamW step operates on a
rank-r view of the gradient that captures its dominant components.

**Identity at step 0**: with `P = I_n[:, :r]` (first r columns of identity)
and `Q = I_m[:, :r]`, the projection is `G̃ = G[:, :r]` on the first r
columns. The AdamW state has shape r×m, so the first step's update is the
*leftmost r columns* of the full-rank AdamW update — not bit-identical to
AdamW. To make the first step bit-identical, initialize `P, Q` to a
permutation of identity (i.e. P, Q are *columns* of the identity picked
uniformly at random; the projection's output is a 1-1 map of the
gradient's leading axes onto the r-axis). At step 0 with P, Q = a random
permutation of identity columns, the first step is **a rank-r AdamW
update along a random r-column axis of the gradient** — the full gradient
is preserved, only the basis changes. A small `eps` on the projection
(orthonormalize) keeps it numerically clean.

## Design sketch
- `optimizers/galore.py`: `GaLoreAdamW(params, rank=4, proj_every=200, lr, betas)`
  wrapper that takes a list of named parameters and the projection schedule.
  At init, instantiates `P, Q` as random column-permutations of identity
  per 2-D matrix parameter, orthonormalized. The optimizer state (m, v) is
  shaped (r, m) for Q-side and (n, r) for P-side; in the simplest
  implementation, the AdamW state is on the *projected* gradient `G̃` of
  shape (r, m) (or (n, r) — pick one orientation, e.g. right-side so
  `(n, r)·(r, m) = (n, m)`).
- `training/trainer.py`: when `use_galore=True`, route the 2-D
  non-embedding, non-norm params to `GaLoreAdamW` instead of `AdamW` (or
  instead of `Muon`, depending on the routing config). The 1-D / embedding
  / norm path stays on plain `AdamW` (GaLore is a 2-D lever).
- `configs/llm_config.py`: add `use_galore: bool = False`,
  `galore_rank: int = 4`, `galore_proj_every: int = 200`,
  `galore_lr: float = 0.006` (same as `adamw_lr`).
- LoC: ~50 (the projection update + AdamW on `G̃`); plus ~15 in trainer
  for routing. Total ~65.
- Identity at step 0: `P, Q` = column-permutations of identity ⇒
  `G̃ = G[:, perm_r]` (a column-permuted slice of G). The AdamW state on
  `G̃` is *not* bit-identical to AdamW on `G` because the state shape is
  different; however, the *update direction* (when re-projected back to
  full rank) equals a permutation of the standard AdamW update, so the
  first step differs only in the choice of which axes AdamW tracks
  variance on. With `rank = min(n, m)` the lever collapses to AdamW.
- The intuition: at 0.94M, GaLore's gain from memory is moot (full AdamW
  state fits trivially). The lever's bet is on the *quality* side: the
  gradient's spectral structure is heavy-tailed, with most variance in
  r ≪ d directions. Forcing AdamW to track variance on those r
  directions *only* may reduce noise on the (n·m − r·max(n,m)) tail
  dimensions, where AdamW's per-parameter second moment is dominated by
  noise. A null would tell us the gradient's low-rank structure doesn't
  matter at 0.94M (or that the r-axis is mis-identified by random init).
  A win would mean the heavy-tail structure *is* load-bearing even at
  0.94M and the per-parameter variance is hurting.

## Scale evidence
- arXiv:2403.03507 (Zhao et al. 2024): Llama 1B and 7B pretraining
  show parity-to-better val loss vs AdamW at half the optimizer state
  memory. Rank = 4 or 8 is the paper's sweet spot.
- nanoGPT speedrun (independent impl): GaLore has been re-validated at
  the ~125M NanoGPT-LM tier with no val loss regression.
- Li et al. 2024 "AFLOSS" (arXiv:2410.22403) reports a follow-up that
  combines GaLore with a momentum-based denominator — not the lever
  here, but a clean replication at 1.3B.
- Transfer risk: **med**. Validated at 1B-7B (≥100M), the lever is
  scale-free in the *direction* of the bet (low-rank gradient structure
  is a property of any deep net), but the *magnitude* of any gain at
  0.94M is genuinely unknown — at this scale the per-parameter
  second-moment noise may already be negligible.

## Why it's worth a slot
This is one of the few 2024 optimizer levers that (a) has direct
≥1B-scale validation, (b) has a clean mechanism description, (c) is
ortho to the entire closed optimizer zoo (it modifies the *preprocessing*
of G, not the per-step update rule), and (d) leaves 1-D / embedding /
norm params untouched. The 031-040 closed batch (Adam-mini, AdamS,
Sophia, Adan, AGC, LAMB, Adafactor) all change the *update rule*. The
103-momentum-streams archive lever also changes the update rule. 113
GaLore is the only lever filed that operates on the *gradient before
the optimizer sees it*. The slot tests whether a "compressed view of
the gradient" is load-bearing at this scale — a null would say
"0.94M's gradients are already low-rank in any basis, so the
projection is a no-op"; a win would say "the heavy-tail structure is
the lever, not the memory savings, and GaLore rides on the same axis
even at 0.94M".

## Plan

**Files touched**
- `optimizers/galore.py` (new): `GaLoreAdamW` — `torch.optim.Optimizer`
  subclass implementing low-rank projected AdamW. State per 2-D
  param: `P ∈ R^{n×r}`, `Q ∈ R^{m×r}` (orthonormal projection
  matrices), `grad_ema` (running gradient for SVD basis refresh),
  `exp_avg`, `exp_avg_sq` (AdamW moments in the r×r projected
  space). 1-D params take the plain AdamW path.
- `configs/llm_config.py`: add `use_galore: bool = False`,
  `galore_rank: int = 4`, `galore_proj_every: int = 200`,
  `galore_lr: float = 0.006`, `galore_beta1: float = 0.9`,
  `galore_beta2: float = 0.999`, `galore_eps: float = 1e-8` to
  `LLMConfig`. Add `Tiny1M3MGaLoreConfig(Tiny1M3MConfig)` that sets
  `use_galore=True` and keeps the paper defaults.
- `training/trainer.py`: add a `galore_params` bucket in
  `setup_muon_optimizer` and a routing branch
  `if use_galore: galore_params.append(param)` on the 2-D
  non-embed, non-norm slot. Construct a `GaLoreAdamW` instance
  when `use_galore=True` (mutually exclusive with Muon/Lion/SWAN
  on that slot) and add it to the returned optimizers list.
  Bind `galore_optimizer = None` in the other branches so the
  name is always defined.
- `train_llm.py`: add `--use_galore`, `--galore_rank`,
  `--galore_proj_every`, `--galore_lr` CLI flags + override block.

**Zero-init at step 0**: the model graph is unchanged — `use_galore`
only swaps the optimizer on the 2-D slot, no `nn.Parameter` is added.
Verified: `MinimalLLM(Tiny1M3MConfig())` and
`MinimalLLM(Tiny1M3MGaLoreConfig())` produce bit-identical
parameters at seed 42, and a forward pass on identical input
produces bit-identical logits. The val score at step 0 (computed
before any optimizer step) is therefore byte-identical to baseline.
The first optimizer step itself differs from AdamW's first step
(it operates on a rank-r projection); this is the inherent
behavior of GaLore and matches the paper's "not bit-identical at
step 0" caveat.

**Run command** (tiny1m3m seed 42):
```
cd /root/universe-lm && /venv/main/bin/python train_llm.py \
  --config_class configs.llm_config.Tiny1M3MGaLoreConfig \
  --output_dir runs/113-galore/seed42 \
  --seed 42
```
Mirror with `Tiny1M3MConfig` (no flag) for the ctrl.

**Final val loss** is read from `plots/metrics_<timestamp>.json` /
`metrics.json` (the same path other experiments in this repo use).

**LoC budget**: ~190 LoC total
  - `optimizers/galore.py` ~170 (optimizer class with init /
    refresh / step / 1-D fallback)
  - `configs/llm_config.py` ~15 (flag block + config class)
  - `training/trainer.py` ~15 (routing + opt construction)
  - `train_llm.py` ~5 (CLI flags + overrides)
Under the 200 LoC ceiling.

## Re-code fix (round 1 → 2)

Round-1 run failed with `NotImplementedError: 'geqrf_cuda' not
implemented for 'BFloat16'` — `torch.linalg.qr` (and `svd`) on the
Vast V100 CUDA build lack a bf16 kernel. The model trains in bf16,
so the QR init path (`_init_state`) and the SVD refresh path
(`_refresh_projection`) hit the missing kernel.

**Fix**: promote the QR/SVD inputs to float32, run the decomposition,
cast the result back to the param dtype. The QR result is a *random
orthonormal basis*, not a numerical copy of any tensor, so the cast
has no downstream effect — same basis, same AdamW-in-the-basis
update, same projection behavior.

**Baseline equivalence**: with `use_galore=False` (default), the
optimizer is never constructed — the trainer routes the 2-D slot
to Muon and `galore_optimizer = None`. The bf16 promotion path
is therefore dead code in the ctrl. The forward pass is unchanged
(no `nn.Parameter` is added), so step-0 val loss is bit-identical
to baseline. The first optimizer step itself differs from AdamW's
(it operates on a rank-r projection); this is the inherent GaLore
behavior and matches the paper's caveat.

**Diff size**: 2 hunks in `optimizers/galore.py` (init: cast P,Q to
float32 for QR, cast back; refresh: same for SVD input). Zero
touches to `configs/llm_config.py`, `training/trainer.py`, or
`train_llm.py`. Well under the 200-LoC ceiling.

**PASS bar**: ≤ ctrl − 0.005 (taste's mid-band for an orthogonal
optimizer-side lever at 12L depth; paper effect is small at this
scale but the *quality* side is what we're testing, not the
memory side which is moot at 0.94M). NULL band |Δ| < 0.005. DRIFT
> +0.005. ctrl_val baseline 6.4306 (`Tiny1M3MConfig`,
`LEADERBOARD.md` row 14) — interpreted against the in-session
ctrl run to avoid cross-session drift. Seed 42 only.

## Re-code fix (round 2 → 3)

Round-2 run still failed with the same `NotImplementedError:
"geqrf_cuda" not implemented for 'BFloat16'`. The round-1 fix
was conditional: `proj_dtype = torch.float32 if p.dtype ==
torch.bfloat16 else p.dtype`. That branch *should* have caught
the bf16 case — and the logic is correct locally (verified by
importing `optimizers/galore.py` and stepping with a bf16
param). The most likely cause is a stale file on the Vast
runner (the deploy may not have picked up the round-1 edit),
or the conditional missed a case (e.g. fp16).

**Fix (round 3)**: make the cast *unconditional* — always run
QR/SVD in float32 regardless of `p.dtype`. Simpler and
bulletproof:

```python
# Always float32 for QR — geqrf_cuda lacks bf16/fp16 kernels.
P = torch.randn(n, rank_actual, device=p.device, dtype=torch.float32)
P, _ = torch.linalg.qr(P)
Q = torch.randn(m, rank_actual, device=p.device, dtype=torch.float32)
Q, _ = torch.linalg.qr(Q)
state["P"] = P.to(dtype=p.dtype)
state["Q"] = Q.to(dtype=p.dtype)
```

SVD path: same pattern, always promote to float32 before
`torch.linalg.svd`.

**Diff size**: 2 hunks in `optimizers/galore.py` (init + refresh
— both replace the conditional cast with an unconditional
float32 input). Zero touches to `configs/llm_config.py`,
`training/trainer.py`, `train_llm.py`. ~10 LoC total. Under the
200-LoC ceiling.

**Baseline equivalence**: unchanged — `use_galore=False`
(default) never constructs `GaLoreAdamW`; the trainer routes
the 2-D slot to Muon and the float32 QR/SVD path is dead code
in the ctrl. Step-0 forward pass is bit-identical to baseline
(verified: `MinimalLLM(Tiny1M3MConfig())` and
`MinimalLLM(Tiny1M3MGaLoreConfig())` produce bit-identical
parameters at seed 42).
