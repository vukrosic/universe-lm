## r1 — 2026-06-10 — verdict: accept

**Mechanism check (idea.md ↔ code).** Idea pins `o_h ← o_h · 2·σ(W·x+b)` with
`W : nn.Linear(d_model, H)` (per-head scalar, identity-init `W=0, b=0`), gate
input = sublayer input residual (pre-LN). Implemented at
`models/layers.py:1870-1873` (post-AV, pre-merge) with the gate broadcast as
`g.view(B, T, H, 1).transpose(1, 2)` against `attn_output` of shape
`[B, H, T, D]` — correct shape plumbing, site is right.

**Wiring (LLM → TransformerBlock → MHA).** Plumbed in three places:
- `models/llm.py:233` reads `config.use_gated_attn` (default-off, no silent HP).
- `models/llm.py:367` forwards to `TransformerBlock(..., use_gated_attn=...)`.
- `models/llm.py:536-540` re-zeros `gated_attn_proj.{weight,bias}` after
  `_init_weights` runs (which would otherwise overwrite the zero-init with
  `normal_(std=0.02)` — defensive belt-and-suspenders, good).
- `models/layers.py:480` (MHA ctor) and `models/layers.py:2005` /
  `:2073` (TransformerBlock ctor) accept and forward the kwarg.

**Pre-LN gate input.** All three TransformerBlock.forward call sites pass
`gate_x=x` (raw residual, NOT `norm1(x)`):
- parallel: `models/layers.py:2335` — `self.attention(n, ve, gate_x=x, ...)` ✓
- post-norm: `models/layers.py:2353` — `self.attention(x, ve, gate_x=x, ...)` ✓
- pre-norm: `models/layers.py:2378` — `self.attention(self.norm1(x), ve, gate_x=x, ...)` ✓

Matches idea.md:16 ("NOT `o_h` itself — circular").

**Identity safety.**
- `use_gated_attn=False` ⇒ no `nn.Linear` allocated
  (`models/layers.py:739-742` is guarded), no `sigmoid` runs, `gate_x` kwarg
  ignored, MHA primary path is bit-identical to current.
- `use_gated_attn=True` with zero-init ⇒ `2·σ(0) = 1.0` exactly (fp32-safe
  constant) ⇒ `o_h * 1.0` is bit-identical to baseline at step 0.

**Composition with `use_attn_output_gate`.** Code places the new gate AFTER
the existing `use_attn_output_gate` block (`models/layers.py:1857-1859`) and
BEFORE `_apply_output_op` (line 1876). Both multiply on the same
`[B, H, T, D]` tensor; the two compose cleanly. Distinct site from
`use_attn_output_gate` (ReZero-style learnable scalar gain, NOT input-
conditional) — categorically distinct lever, as idea.md:11 claims.

**Plan prose discrepancy (informational, NOT a revise).** Plan said
"Placed AFTER the `_apply_output_op(attn_output)` call (line 1737)" but the
implementation is BEFORE `_apply_output_op`. With the default
`out_op=""`, `_apply_output_op` is a no-op (per its own docstring at
`models/layers.py:1874-1875`), so the gate result is bit-identical
regardless of ordering. The idea-level mechanism (gate on `o_h`, between
attention output and the O projection merge) is satisfied in both
placements. The actual placement is also more consistent with the existing
`use_attn_output_gate` lever (same site). No code change needed; plan
prose is the side that should be updated. Captured for plan hygiene only.

**HP-drift / seed / control.**
- `use_gated_attn: bool = False` default ⇒ baseline path untouched.
- Seed 42 only, ctrl = `Tiny1M3MFIREConfig`, trt =
  `Tiny1M3MGatedAttnOnFireConfig`. Plan ↔ idea match. Pass bar
  `Δ ≤ −0.01` copied correctly.
- LoC: idea ~18, plan 134 (well under 200).
- New trt class `Tiny1M3MGatedAttnOnFireConfig` is exported in
  `configs/__init__.py:15, :108`.

**Coordination.** `git status` is clean on `models/` and `configs/`
(no parallel-AI uncommitted edits to stomp), and the implementer's full
diff is committed at `c6bb6a3`. No push, no rebase.

**Verdict: accept.** Set status to `needs-run`.
