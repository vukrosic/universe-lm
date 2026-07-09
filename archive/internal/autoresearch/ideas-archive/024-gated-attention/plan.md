# Plan — 024 gated-attention

## Flag
- `use_gated_attn: bool = False` — new config field, default OFF, added at
  `configs/llm_config.py` next to `use_sub_ln` (line 325 area). Plumbed through
  `LLM.__init__` (line 215 area, alongside `use_fire_pe`) → `TransformerBlock`
  ctor (line 1886 area) → `MultiHeadAttention` ctor (line 465 area) → applied
  in `MultiHeadAttention.forward` (line 1732 area, just before `o_proj` merge).
- Wires **no other flag**. The new `use_gated_attn` does NOT touch the existing
  `use_attn_output_gate` (which is a *per-head learnable scalar gain*
  `o_h *= (1 + g_h)`, a ReZero-style lever, no input conditioning — different
  site-of-leverage from the input-conditional sigmoid gate this idea adds).

## Change

### Mechanism (pinned verbatim from `idea.md`)
After the per-head attention output `o_h = A_h V_h` (post-AV, pre-merge) we
multiply it element-wise by an input-conditional per-head **scalar** sigmoid
gate:

    o_h ← o_h · 2 · σ(W_g · x_sub + b_g)

with

- `W_g : nn.Linear(d_model, n_heads)` — one scalar gate per head, **not** the
  per-head vector form. Identity-init: `W_g.weight = 0`, `W_g.bias = 0`.
  At init `2·σ(0) = 1.0` exactly → step-0 ≡ baseline to floating-point.
- `x_sub` — the **sublayer input residual** (pre-LN/attn). In pre-norm this
  is the raw `x` at the TransformerBlock level (NOT `norm1(x)`, NOT `o_h`
  itself — that would be circular). The reviewer pinned the pre-LN site.

### Files & function-level diff (prose)

1. **`configs/llm_config.py`** — add `use_gated_attn: bool = False` next to
   `use_sub_ln` (after line 325). Default OFF → bit-identical baseline.
2. **`models/layers.py — MultiHeadAttention.__init__`** — accept new kwarg
   `use_gated_attn: bool = False`. When on, create a single
   `self.gated_attn_proj = nn.Linear(d_model, n_heads, bias=True)` and
   `nn.init.zeros_(self.gated_attn_proj.weight)`,
   `nn.init.zeros_(self.gated_attn_proj.bias)` (zero-init → step-0 ≡ 1.0).
3. **`models/layers.py — MultiHeadAttention.forward`** — accept new optional
   kwarg `gate_x: torch.Tensor | None = None`. When `use_gated_attn=True` and
   `gate_x` is provided, compute
   `g = 2.0 * torch.sigmoid(self.gated_attn_proj(gate_x))` of shape
   `[B, T, H]`, then `attn_output = attn_output * g.view(B, 1, T, H)` (B,H,T,D
   layout, broadcast over D). Placed AFTER the
   `_apply_output_op(attn_output)` call (line 1737) and BEFORE the
   `[B,H,T,D] → [B,T,d_model]` merge (line 1740) so the gate is on the
   per-head output `o_h`, NOT on the merged tensor. Also placed AFTER the
   pre-existing `use_attn_output_gate` block (line 1732) so the two flags
   compose cleanly when both are on. When `gate_x is None`, fall back to
   using the MHA's primary input `x` (post-norm or call sites that don't
   plumb the raw residual — back-compat).
4. **`models/layers.py — TransformerBlock.__init__`** — accept
   `use_gated_attn: bool = False` and forward it to the `MultiHeadAttention`
   ctor.
5. **`models/layers.py — TransformerBlock.forward`** — at all three
   pre-norm / post-norm / parallel call sites, pass `gate_x=x` to
   `self.attention(...)` (the raw residual stream, not `norm1(x)`). When
   `use_gated_attn=False` the kwarg is ignored by MHA → no behaviour change.
6. **`models/llm.py — LLM.__init__`** — read
   `self.use_gated_attn = getattr(config, "use_gated_attn", False)` next to
   the FIRE/CoPE/Softpick reads (around line 228). Plumb to
   `TransformerBlock(..., use_gated_attn=self.use_gated_attn)` in the block
   construction loop (around line 343).
7. **`configs/llm_config.py — trt class`** — add
   `Tiny1M3MGatedAttnOnFireConfig` (FIRE-equip baseline + `use_gated_attn=True`).
8. **`configs/__init__.py`** — export the new trt class.

### Step-0 identity check
With `use_gated_attn=False` the MHA's primary path is **bit-identical** to
the current `MultiHeadAttention.forward` (the new kwarg is unused, no
`nn.Linear` is created, no `sigmoid` runs, no `*g` is applied). With
`use_gated_attn=True` and zero-init, `g = 2·σ(0) = 1.0` exactly → the
`o_h * g` multiply is bit-identical to the no-gate baseline (modulo the
matmul shape bookkeeping, which is just `view` + broadcast).

## Control
- **Ctrl** — `Tiny1M3MFIREConfig` (the 009 WIN signature, val 6.3234 at
  2026-06-09 per `closed.md:40`): `use_fire_pe=True` on `Tiny1M3MConfig`.
  The FIRE-equipped baseline is the established baseline for the active
  attention-side batch (same convention 020-FoX and 022-Softpick use), so
  the A/B partitions the **orthogonal** head-output axis (024) from 009's
  additive position bias. Without this, the 009 effect would be confounded
  with the gate effect.
- **Trt** — `Tiny1M3MGatedAttnOnFireConfig` (new class, same as ctrl +
  `use_gated_attn=True`).
- **Seed** — 42 (one-seed-only, per the seed-42 rule).
- **Tier** — tiny1m3m only (6L·8H·d_model=256, d_k=32, ~0.94M params).
  Scalar-gate math is budgeted at this tier; the d_k=128 vector form would
  blow the parameter budget (per the reviewer's per-tier scaling check).

## Cost
- **Params Δ** — 1 `nn.Linear(d_model, n_heads)` per layer: weights
  `[n_heads, d_model]` + bias `[n_heads]`. At tiny1m3m
  (d_model=64, n_heads=4, n_layers=12): `(4·64 + 4) = 260` params/layer.
  × 12 layers = **3,120 params** → **~0.33% of the 0.94M model**.
  (Reviewer estimated "12,288 params / 1.3% of model" assuming
  d_model=256, n_heads=8, n_layers=6; the actual tiny1m3m dimensions are
  smaller so the gate is even cheaper at this tier.)
- **FLOPs Δ** — one `(B, T, d_model) → (B, T, H)` matmul per layer
  (cheap, ~B·T·H·d_model = 2·B·T·1024 = 4096·B·T flops). Plus a sigmoid on
  (B, T, H) and a per-head broadcast multiply on (B, H, T, D). All
  negligible vs the attention O(B·H·T²·d_k) ≈ 268M·B·T at T=2048.
- **Memory Δ** — 12,336 params (~24 KB in fp32) + 1 activation per token
  (the (B, T, H) gate). Trivial.
- **Step-0 forward** — bit-identical to ctrl (g = 1.0 exactly, no extra
  ops materialize at step 0).

## Run
- **Tier** — `tiny1m3m`
- **Seed** — 42 (pinned; one seed only per the seed-42 rule)
- **Command** — the standard two-ctrl / one-trt runner invocation (see
  `prompts/runner.md`); spec the new
  `Tiny1M3MGatedAttnOnFireConfig` as the trt slot, the
  `Tiny1M3MFIREConfig` (or `use_fire_pe=True` on `Tiny1M3MConfig`) as the
  ctrl slot, run 2 ctrls + 1 trt at seed 42.
- **Expected wall-clock** — ~ the same as the FIRE baseline run on
  vast-34386 (the gate adds <1% FLOPs).
- **Pass bar** (copied from `idea.md:26`, reviewer-locked) — `Δ := trt_val
  − ctrl_val`; **pass iff `Δ ≤ −0.01`**. Box noise at tiny1m3m is
  ~±0.01 val loss; sub-noise deltas are **inconclusive, not real** — log
  null and close, do **not** "add seeds to confirm" (one-seed-only rule).

## Self-check (before release)
- `use_gated_attn=False` reproduces the existing control numerically (no
  `nn.Linear` allocated, no `sigmoid` runs, `gate_x` kwarg ignored) → no
  drift on existing configs.
- `use_gated_attn=True` with zero-init gives `g = 2·σ(0) = 1.0` exactly →
  the new code path *exists* in the forward graph but is numerically
  equivalent to the no-gate path at step 0. The new params will start
  receiving gradient from step 1 onward.
- The gate is applied to `o_h` (per-head output, [B,H,T,D]) not the
  merged tensor — matches the Qiu site exactly.
- Plan matches `idea.md:26` pass bar.
