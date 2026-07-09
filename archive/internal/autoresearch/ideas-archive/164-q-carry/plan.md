# Plan — 164 Q-Carry (Cross-Block Q Residual)

## Mechanism

For each block l (l ≥ 1), augment the Q projection with a learnable
cross-block carry from block l-1's MHA sublayer input:

```
Q_l = W_Q(x_l) + α_l · W_Q(prev_x)
```

where `prev_x = LN(x_{l-1})` is the residual stream entering block l-1
*after* `norm1` (i.e. the actual tensor the previous block's MHA saw
in its pre-norm / parallel / post-norm forward), and `α_l` is a
per-block learnable scalar (init 0).

- K, V, O projections are unchanged. The carry is added BEFORE
  q_norm / RoPE / q_only_norm (so 016 and 162 still rescale Q +
  α·Q_carry consistently; at step 0 both terms are zero, the
  post-rescale is the standard pre-trained norm).
- `prev_x` is `.detach()`-ed (mirroring 021's V-residual contract at
  `models/layers.py:2250-2257`): only `α_l`'s gradient flows through
  the carry path, not layer-l-1's W_Q / norm / block params. The
  cross-block gradient is structurally bounded to α_l's 12 scalars.
- At layer 0, no previous block exists ⇒ `q_carry=None` ⇒ the carry
  branch is a no-op. The stash `MHA._q_carry = x.detach()` (where
  `x` inside `MHA.forward` is the MHA sublayer input, i.e.
  `norm1(x_0)` in pre-norm) is set on layer 0 and read back by the
  model loop.

This is the Q-side dual of 021's V-residual. 021 stashes V from
layer 0 and blends `(1-λ)·V + λ·V_1` later; 164 stashes the
previous block's MHA sublayer input and adds `α·W_Q(prev_x)` to Q at
layer l. Same scaffolding, different tensor.

## Flag

- `use_q_carry: bool = False` (default off) on `LLMConfig` —
  `configs/llm_config.py`. Placed next to `use_value_residual` (line
  396 area), same comment-block style documenting the mechanism +
  identity claim.
- A/B subclass: `class C(Tiny1M3MConfig): use_q_carry: bool = True`
  at the top of `_arq_164-q-carry.py`. The canonical daemon stub
  pattern (per `autoresearch/RUN-CONTRACT.md`).

## Wiring

- `configs/llm_config.py` — adds `use_q_carry: bool = False` next
  to `use_value_residual`. ~12 lines of comment block.
- `models/layers.py` —
  - `MultiHeadAttention.__init__` accepts `use_q_carry: bool = False`
    (declared alongside `use_value_residual` at line 760), and
    when on, registers `self.alpha_q = nn.Parameter(torch.zeros(()))`
    and initializes `self._q_carry = None` (the forward-pass-local
    stash). 1-dim scalar per block, init 0.
  - `MultiHeadAttention.forward` signature gains `q_carry=None`
    kwarg. At the top of `forward`, when `use_q_carry=True` and
    `q_carry is None` (the layer-0 stash branch), set
    `self._q_carry = x.detach()`. After the QKV split (post the
    `if self.use_shared_kv / use_tied_qk / use_mla / default` block
    where Q is shape `[B, T, q_size]`), inject the carry:
    ```
    if self.use_q_carry and q_carry is not None:
        if self.use_tied_qk:
            q_carry_w = self.qk_proj[:self.q_size]
        else:
            q_carry_w = self.qkvo_proj[:self.q_size]
        Q = Q + self.alpha_q * F.linear(q_carry, q_carry_w)
    ```
    `qk_proj[:q_size]` is the W_Q slice in the tied-QK branch
    (where W_Q == W_K); the standard `qkvo_proj[:q_size]` is the
    W_Q slice in all other branches (default, shared_kv, MLA —
    MLA still uses the merged qkvo_proj's Q slice, the latent is
    K, V only).
  - `TransformerBlock.__init__` accepts `use_q_carry: bool = False`,
    pass-through to inner `MultiHeadAttention(...)` constructor
    (parallel to `use_value_residual` at line 3084).
  - `TransformerBlock.forward` signature gains `q_carry=None`
    kwarg; pass it to all three `self.attention(...)` call sites
    (parallel_block, post_norm, pre_norm) at lines 3874, 3909, 3947
    (with the same `q_carry=q_carry` plumbing as 021's v_residual
    at lines 3874/3909/3947).
- `models/llm.py` —
  - `MinimalLLM.__init__` reads
    `self.use_q_carry = getattr(config, "use_q_carry", False)`
    (next to `self.use_value_residual` at line 337).
  - `MinimalLLM._run_post_embed` initializes
    `q_carry = None` next to the existing `v_residual = None` at
    line 1306, passes it through to both `block(...)` call sites
    (the HyperConnections wrapper and the standard path), and after
    layer 0 captures `q_carry = block.attention._q_carry` (mirrors
    the 021 v_residual capture at line 1390-1399).
  - `MinimalLLM.__init__` pass-through into both
    `TransformerBlock(...)` constructor sites (standard at line 879
    and YOCO upper at line 634): `use_q_carry=self.use_q_carry`.

## Step-0 identity claim (the §5 self-check)

- **Flag OFF path (the baseline reproduction).** When
  `use_q_carry=False`, no `alpha_q` parameter is registered, no
  `_q_carry` attribute is created, no `q_carry` kwarg is consulted,
  no `F.linear(q_carry, ...)` is computed. The MHA forward graph
  is byte-identical to the existing baseline. Verified locally
  via the daemon's build-smoke (`MinimalLLM(Tiny1M3MConfig())`
  constructs on CPU without error).
- **Flag ON at step 0.** `alpha_q = 0.0` init ⇒ `α_l · W_Q(prev_x) =
  0` exactly in fp32 (the scalar multiply with zero zeroes every
  element). The `F.linear(q_carry, q_carry_w)` matmul still runs
  (the branch is taken), but its contribution is `0 * W_Q(prev_x) =
  0` — bit-identical to the no-carry baseline within fp32 rounding
  noise of one extra multiply-add. (Mirror 021's claim at
  `models/layers.py:2249-2250`.) The stash path on layer 0 still
  fires (`self._q_carry = x.detach()`) but the stash is a
  no-side-effect attribute write — the model only reads it back
  AFTER the layer-0 block returns, so it cannot perturb layer 0's
  forward graph.
- **Explicit fp32 max-abs-diff ≤ 1e-6 verification (per
  review.md r1).** When `use_q_carry=True` and `alpha_q=0` for ALL
  blocks, the treatment's step-0 forward must match the
  flag-OFF baseline to within fp32 max-abs-diff ≤ 1e-6 across all
  12 blocks. (The extra `F.linear(q_carry, q_carry_w)` matmul
  contributes `0` exactly, but an extra matmul in the graph can
  perturb numerics at the trailing bits if the path is not
  branch-skip-elided — same bug class as 150-xlayer-feedback's
  inplace `xlayer_mem.append`.) The runner's `isolate_effects`
  audit will catch any deviation.

## Pass bar (numeric Δ vs fresh ctrl, per review.md r1)

The bar is **tightened** to be resolvable at 92-step tiny1m3m
(one-seed-only rule, no scale-up retry possible):

- **WIN:** `Δ ≤ -0.01` at tiny1m3m seed 42 against a fresh ctrl
  (cached-mean `6.4394±0.04` per
  `autoresearch/baseline-cache.json` — box `5b8a7fea8963`, but
  the daemon re-baselines against the current-box control). The
  magnitude matches 021's recorded `Δ = -0.034` (the WIN that
  motivated this axis) at half-bar — a Q-side win at `Δ = -0.01`
  is a clear dual-axis signal (residual-stream mixing generalizes
  from V to Q). Win message: "Q carry matches V-carry's
  magnitude ⇒ cross-block residual mixing is the binding axis,
  not V-specific; opens door to generalized 116-mHC at deeper
  tiers."
- **NULL:** `|Δ| < 0.01` — V is special; Q's information source
  is the in-block residual stream, not the previous block's.
  Null message: "Q carry does not bind at 0.94M; 021's WIN was
  V-specific. Closes the dual axis."
- **DRIFT (hostile):** `Δ > +0.01` — the carry actively hurts.
  Strong null (021's WIN was in the other direction, so a
  wrong-sign move at 10× the noise band is a meaningful signal).
  Drift message: "the cross-block gradient coupling (even with
  `.detach()`) disturbs the Q-learning path."
- **CRASH / NaN / OOM** → `needs-recode` (round 1, inside
  budget). The build-smoke + step-0 identity check should catch
  most crash classes; runtime NaN/grad-explode would route here.

## Cost

- **Params:** +12 scalars total (one `alpha_q` per block, 12 blocks
  at tiny1m3m, +0.001% of 0.94M). Negligible.
- **FLOPs:** +1 W_Q matmul per block (one extra `d_model × q_size`
  matmul on `prev_x` shape `[B, T, d_model]`). At tiny1m3m this
  is `B·T·d_model·q_size = 2·2048·64·64 = 16.8M FLOPs/layer`,
  × 12 layers = ~200M FLOPs/step, vs the ~1.8B FLOPs/step baseline
  ⇒ **+11% per-step FLOPs**. Acknowledged cost — mirrors 150's
  extra cross-attn but structurally lighter (single Q projection
  vs full K/V/O cross-attention).
- **Memory:** +1 stash tensor per forward pass (the detached
  `LN(x_0)` of shape `[B, T, d_model]`, 16 KB at tiny1m3m
  fp32). Negligible.

## Composition matrix

Orthogonal / composable with (single-flag diff is the A/B; no
multi-flag composition in this round):

- 021 (V-residual): composes — different tensor (V vs Q),
  different stash site, different blend math. Together: 12 α_q
  + 12 λ_v scalars, two independent cross-block pathways. The
  composed forward at step 0 is byte-identical to baseline
  (both contributions are zero-init).
- 016 (QK-norm): composes — Q carry is added BEFORE q_norm,
  the norm still rescales Q + α·Q_carry.
- 155 (per-head temp): composes — temperature is applied
  post-softmax logits, not on Q directly.
- 161 (per-layer temp): composes — same as 155, post-softmax
  logit multiplier.
- 162 (Q-only norm): composes — same as 016, applied to
  Q + α·Q_carry.
- 020 (FoX), 022 (softpick), 025 (SSMax): composes — all live
  on the softmax / post-softmax path, not on the Q projection.

This round enables **only** `use_q_carry=True` (single-flag
diff against the bare `Tiny1M3MConfig` ctrl).

## Run

- **Artifact:** `_arq_164-q-carry.py` (repo root) defines
  top-level `class C(Tiny1M3MConfig): use_q_carry: bool = True`
  and dispatches `train_llm.main()` with `--config_class
  __main__.C --seed 42 --dataset_path
  processed_data/pretrain_1B --warmup false`.
- **Descriptor:** `autoresearch/ideas/164-q-carry/run.json` —
  `{"name": "164-q-carry", "arq_file":
  "_arq_164-q-carry.py", "job_timeout": "12m"}`. (Default
  12m cap is fine — the +11% per-step FLOPs lifts wall-clock
  proportionally but 12m is the same order as the baseline
  12m cap; if the box reports a timeout, bump in a recode
  round.)
- **Daemon (`autoresearch/bin/queue-daemon.sh`):** scp's the
  stub, runs the CPU build-smoke (`MinimalLLM(C())` constructs
  without error), then launches the run in the `arq` tmux.
- **Reference:** 021-value-residual (the closed WIN at
  tiny1m3m, `Δ = -0.034` vs shared fire-ctrl 6.3419 — caveat:
  in-isolation re-test pending, but the magnitude shapes the
  bar). Cached baseline: `6.4394±0.04` (per
  `autoresearch/baseline-cache.json` box `5b8a7fea8963`,
  measured `2026-06-14T05:29:18Z`).
