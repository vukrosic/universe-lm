---
id: 116-hyper-connections
status: done
round: 1
updated: 2026-06-13T14:29:54Z
transfer-risk: med
plain: It splits the residual stream into several parallel lanes that get mixed between layers, so the network has more ways to move information forward without widening the hidden size.
---

# 116 — Hyper-Connections (mHC): Multi-Stream Residual

## Source
Zhu, Xie, Zhang, Sun (DeepSeek-AI), "Multi-Head Attention: Collaborate
Instead of Compete", but for residual: the *Hyper-Connections* line
documented in the DeepSeek-V3 technical report (DeepSeek-AI 2024) and
the dedicated paper Xie et al., "Hyper-Connections" (arXiv:2409.19606,
Sept 2024). https://arxiv.org/abs/2409.19606
Used as the residual-stream backbone of DeepSeek-V3 (671B MoE,
~37B activated) with consistent +loss / +benchmark gains reported
in the V3 tech report; ablation tables in the dedicated paper validate
the *mechanism* independently of MoE at ≤3B.

## Mechanism
Replace the standard single residual stream `x_l = x_{l-1} + f_l(x_{l-1})`
with `n_resid` parallel streams of width `d_model / n_resid` each. Before
each block, a learnable mixing matrix `B_l ∈ R^{n_resid × n_resid}` mixes
the streams into the block's input; after each block, a second mixing
matrix `A_l ∈ R^{n_resid × n_resid}` mixes the block's residual contribution
back out. With `n_resid = 1`, `A_l = B_l = 1` and the construction collapses
to standard residual. With `n_resid = 4` and `A_l, B_l` initialized to
`I_{n_resid}`, the streams do not mix — each block reads its own stream
and writes its own stream — and the model is *bit-identical* to a
4-way split of the residual stream where each stream is processed by the
same block weights (the streams differ only in their *initial content*,
set by a learnable per-stream embed at the input).

Concretely for tiny1m3m:
  `x_in  = concat([x_in_1, ..., x_in_{n_resid}])  ∈ R^{B, T, d_model}`
  `x_mixed = (B_l ⊗ I_{d_l}) · x_in`                 # mix streams pre-block
  `x_block = f_l(x_mixed)`                            # standard block
  `x_resid = x_mixed + (A_l ⊗ I_{d_l}) · x_block`     # mix streams post-block
  `x_out = (C_l ⊗ I_{d_l}) · x_resid`                 # final output mixing
`A_l, B_l, C_l ∈ R^{n_resid × n_resid}` — three small `(n_resid² × L)`
parameters, total `3 × n_resid² × n_layers = 3 × 16 × 6 = 288` extra
scalars. Negligible param cost.

**Identity at step 0**: with `n_resid = 4`, `A_l = B_l = C_l = I_4`, the
mixing is the identity permutation — streams pass through unmodified.
Each block receives its own slice of the input and adds its own slice of
the residual. The model is equivalent to a *tied* 4-way split of the
residual stream at step 0. After init, `A_l, B_l, C_l` learn to mix
information across streams.

## Design sketch
- `models/mhc.py` (new): `MultiStreamResidual` class — wraps a standard
  block, maintains `n_resid` stream views, and applies the per-layer
  `A_l, B_l, C_l` mixing. ~80 LoC.
- `models/llm.py`: when `config.use_hyper_connections=True`, replace
  the standard residual path in `Block.forward` with a wrapped version
  that splits `x` into `n_resid` chunks along the feature dim, runs each
  chunk through the same attention+FFN (weights shared across chunks, so
  the block's parameter count is *unchanged*), and re-concatenates. The
  mixing matrices `A_l, B_l, C_l` are registered as `nn.Parameter` per
  block. Init: `I_{n_resid}` for all three.
- `configs/llm_config.py`: add `use_hyper_connections: bool = False`,
  `hc_n_resid: int = 4`. Default off → baseline path bit-identical.
- LoC: ~100 (mhc.py) + ~15 (plumbing in llm.py and config) = ~115.
- Identity at step 0: `A_l = B_l = C_l = I_{n_resid}` ⇒ no stream mixing
  ⇒ each block operates on its own slice ⇒ equivalent to a tied
  4-way parameter-shared ensemble of the standard block at step 0. The
  pre-norm baseline path stays bit-identical when the flag is off.
- The intuition: at 0.94M with 6L, the residual stream is the *only*
  highway between layers; a single bottleneck that all 6 layers must
  share. mHC gives the model 4 parallel highways that can carry
  *different* information and mix only when it's useful. The paper's
  bet is that this extra capacity is the lever — at 100L+ it's clearly
  load-bearing; at 6L it's a real question whether the bottleneck is
  the residual stream or something else. A null says "6L doesn't have
  a residual-bottleneck problem and the multi-stream overhead is just
  noise"; a win says "even at 6L the stream-width is the binding
  constraint".

## Scale evidence
- DeepSeek-V3 (671B MoE, ~37B activated, 61L): mHC is the *default*
  residual-stream construction. V3 reports consistent improvements
  over DeepSeek-V2 baselines (which used standard residual) across
  benchmarks at this scale.
- Hyper-Connections paper (arXiv:2409.19606): ablations on ≤3B
  non-MoE transformers show mHC at `n_resid=4` improves loss vs
  standard residual.
- Transfer risk: **med**. The mechanism is scale-free (multi-stream
  residual is well-defined at any depth), but the paper's headline
  wins are at 100L+. At 6L the residual-stream bottleneck may not be
  binding — sub-ln-sandwich (017, closed as null at 6L) and
  DropPath (111, closed as null at 6L) already established that
  *residual-stream* levers often don't fire at our depth. The slot
  tests whether the multi-stream *expansion* lever (which is
  qualitatively different from sub-LN/DropPath — it adds capacity,
  not regularization) survives the 6L regime.

## Why it's worth a slot
mHC is the **only** mechanism filed that *expands the residual
stream itself* — every other residual-side lever in the repo
(canon-conv, unet-skips, zero-init-resid, drop-path, sub-LN) operates
on a *single* stream. Sub-LN is closed (null at 6L), DropPath is
closed (null at 6L), canon-conv won (+0.06 after stripping FIRE).
mHC is the next unexplored axis: capacity, not regularization.
A null would mean "6L's residual stream is already wide enough and
the overhead of stream-mixing outweighs any capacity benefit"; a win
would mean "even at 6L the stream-width is the binding constraint
and parallel highways unlock loss". The slot is ortho to every
closed and active lever in the queue.

## Plan

**Files**:
- `models/mhc.py` (new, ~80 LoC): `MultiStreamResidual` class — wraps a
  standard `TransformerBlock`, maintains `n_resid` parallel stream views
  of width `d_l = d_model // n_resid`, and applies per-block
  `A_l, B_l, C_l ∈ R^{n_resid × n_resid}` mixing matrices at the block
  boundary.
- `models/llm.py`: when `config.use_hyper_connections=True`, build a
  per-position `MultiStreamResidual` wrapper around each tied-block slot
  in `self.transformer_blocks`. Default off → baseline path untouched.
- `configs/llm_config.py`: add `use_hyper_connections: bool = False`
  (default off) and `hc_n_resid: int = 4`. Add a `Tiny1M3MHyperConnectionsConfig`
  preset for the A/B.

**Flag name**: `use_hyper_connections` (off by default → baseline path
bit-identical).
**Hyperparameter**: `hc_n_resid: int = 4` (number of parallel residual
streams; n_resid=1 collapses to standard residual via a 1×1 identity mixing).

**Identity at step 0**: `A_l = B_l = C_l = I_{n_resid}`. The wrapper
computes `x_pre = B @ x`, runs the standard block, then mixes the residual
contribution `sublayer = block_out − x_pre` via `A_l`, then mixes the
combined stream via `C_l`:
```
x_out = (C ⊗ I_d) · (x_pre + (A ⊗ I_d) · sublayer)
```
With B=A=C=I: `x_pre = x`, `sublayer = block(x) − x` (the standard
sublayer contribution), `x_pre + sublayer = block(x)` (the standard
residual), `x_out = block(x)`. Bit-identical to the pre-norm baseline.

**Wrapper signature** matches the block: `(x, x0, ve, v_residual, layer_index)`
so it slots into the existing forward loop with no signature change.

**Per-position wrappers**: when `tie_layer_groups > 1`, the unique block
is shared across positions, but each position still owns its own
`(A_l, B_l, C_l)` mixing matrices (the paper's formulation is per-position,
not per-unique-block). `nn.ModuleList` of wrappers indexes by position `i`.

**Math**:
- `(M ⊗ I_d) · x`: reshape `x ∈ R^{B,T,D}` to `(B,T,n_resid,d_l)`, contract
  `M` on the stream axis via einsum `'ij,btjd->btid'`, reshape back. Cheap
  (one matmul per mixing × 3 = 3 small `(n×n)` matmuls per block per step).

**Run command** (A/B at tiny1m3m, seed 42):
```bash
/venv/main/bin/python train_llm.py --config_class configs.llm_config.Tiny1M3MHyperConnectionsConfig
```
Control run:
```bash
/venv/main/bin/python train_llm.py --config_class configs.llm_config.Tiny1M3MConfig
```

**Read val loss**: `runs/<run_name>/metrics.jsonl` (final val_loss or any
`val_loss` milestone). Compare against `Tiny1M3MConfig` ctrl val 6.4306
(± 0.01 box-noise floor from `LEADERBOARD.md`).

**Pass criterion**: ctrl − 0.005 (mid-band for an architectural expansion
lever at 6L; the bet is at the small end given the prior closed-nulls of
sub-LN and DropPath). NULL band |Δ| < 0.005. DRIFT > +0.005.
