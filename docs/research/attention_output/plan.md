# Attention-output (W_O) ablations — research plan

**For the implementing AI.** Self-contained. The one point we're poking is the
**post-softmax output of attention**, just before the W_O projection mixes it
back into the residual stream. This is the §5 C1 candidate from
[../README.md](../README.md) — no plan existed before, this is it.

---

## The one point we're poking

```text
(inside MultiHeadAttention.forward, manual-attention branch):
  attn_w   = softmax(scores)            # [B, H, T, T]
  attn_out = attn_w @ V                  # [B, H, T, D]   <-- post-softmax output
  attn_out = op(attn_out)                # <-- the ONLY thing we change (NEW)
  attn_out = attn_out.transpose(...).reshape(...)  # merge heads
  x        = self.out_proj(attn_out)     # back into residual
```

Baseline `op = identity`. Every lever is a cheap op on `attn_out` (post-softmax,
pre-merge). Distinct from the **input-side** talking-heads (`talking_heads_q`
mixes `scores` pre-softmax — see [../query/plan.md](../query/plan.md) Batch 2,
Q5). This folder pokes the **output** side: once V has been aggregated by
softmax, before the heads are merged.

Wiring contrast:
- `use_talking_heads_q` (Q5): mixes `[B, H, T, T]` *scores* pre-softmax. M = I init.
- `use_talking_heads_out` (O1, this folder): mixes `[B, H, T, D]` *output* post-softmax. M = I init.

They are independent levers at different positions in the attention block.
Both are H×H = ~144 params for the 12-head case.

---

## Critical wiring note

The manual-attention branch is required for O1 because (a) it's the only branch
where the post-softmax `attn_out` is materialized as a [B, H, T, D] tensor
without an immediate reshape, and (b) all `talking_heads`-family levers are
defined to live in that branch. The O1 flag must therefore extend the
if-condition that triggers the manual path (the same one that routes
`talking_heads_q` etc.) — see Implementation contract below.

The same `M` could theoretically be applied to the SDPA branches too (their
output is also [B, H, T, D] before reshape), but the plan keeps O1 in the
manual branch to (a) match the talking_heads_q wiring and (b) keep the
per-branch behavior clean and easy to A/B against the input-side Q5.

---

## Implementation contract

- Edit `models/layers.py` — only the `MultiHeadAttention` class. One new flag
  in `__init__`, one new `nn.Parameter`, one new einsum in the manual
  forward, one extension of the manual-branch routing condition.
- One `class Tiny1M3M<Name>Config(Tiny1M3MConfig)` per lever, in
  `configs/attention_output_ablations.py`. Flip exactly one flag.
- **Identity-init:** every lever is `step-0 == base`. M=I for O1, gate=1 for O2,
  τ=1 for O3, α=1 for O4, b=0 for O6. The A/B isolates the mechanism.
- **Optimizer routing:** the H×H M (O1) is 2D → Muon. Per-head scalars/gates
  (O2, O4, O6) are 1D → AdamW. Per-head×per-channel gain (O3) is 2D but
  small → AdamW (low param count).
- **Wire a flag, never a fork.** Reuse the existing forward; guard new code
  with `if self.use_<x>:` so the baseline path is untouched when the flag is
  off.

---

## Batches (3 batches, 6 levers — v1 budget)

### Batch 1 — cross-head mixing on the output (the headline C1 lever)

| # | Name | Op on `attn_out` | Spec (step-0 == base) | Params |
|---|---|---|---|---|
| O1 | `TalkingHeadsOut` | `attn_out = einsum("bhtd,hH->bHtd", attn_out, M)` | post-softmax cross-head mix, **M init I** (no-op) | H×H |
| O2 | `OutputHeadGate` | `attn_out *= g_h` (per head, init 1) | per-head scalar gain on the output | H |

### Batch 2 — post-softmax nonlinearity / norm

| O3 | `OutputRMS` | rms-normalize `attn_out` per head, then per-channel gain |
| O4 | `OutputTanh` | `attn_out = tanh(α·attn_out)`, α=1 init (saturating smooth-clip on the output) |

### Batch 3 — value-side variants

| O5 | `OutputSoftplus` | `attn_out = softplus(attn_out)` (output is forced ≥ 0) |
| O6 | `OutputBias` | `attn_out += b_h`, b=0 init (per-head constant prior on the merged output) |

---

## Protocol (what counts as a result)

| Tier | Config | Purpose | Claim? |
|---|---|---|---|
| tiny | `Tiny1M3M*` | rank ideas fast (~2 min/run) | no — kill bad ideas only |
| screen | `Screen10M20M*` | confirm sign survives 20M tokens | **3-seed mean (42/43/44)** before any claim |
| full | `Full10M200M*` | the real target | a winner here only |

A screen winner re-runs on the full ladder before it's a claim. Screen tier
**does not transfer-promote**. Record val_loss vs the clean
`Screen10M20MConfig` control (currently **4.7984**, run dir `s_ctrl_full`).
A lever is "live" only if the 3-seed mean beats control by **≥0.01** and the
seeds don't straddle zero.

Wins here are especially interesting because the **input-side** Q5
(`talking_heads_q` — pre-softmax logit mix) is the natural mirror; the A/B
records are:
- O1 (post-softmax mix) vs Q5 (pre-softmax logit mix): which side of softmax
  is the cross-head lever more useful on?
- O2 (per-head output gain) vs the shipped `use_attn_output_gate` (per-head
  output gain, but `1 + gate` with gate=0 init): same shape, different
  reparam. Scalar vs additive-shifted scalar.

---

## Run guidance

Batch 1 first (the headline C1 lever + the cheap O2). O3–O6 only if O1/O2
show the axis is alive (a wash here means the output side doesn't reward
cheap parameterization at this scale). When a batch finishes, see below.

---

## When a batch finishes

1. Numbers → [tutorial/results.md](tutorial/results.md) (control + each lever,
   3-seed mean + std, run dir, commit).
2. Update status in [tutorial/experiments.md](tutorial/experiments.md).
3. Once a batch has a clear story (win or instructive null), draft
   [tutorial/README.md](tutorial/README.md) in the house style of
   [../../tutorials/qk_gain/README.md](../../tutorials/qk_gain/README.md):
   problem → one mechanism → fair baseline → number → what it means.
4. Add the run's `metrics.json` to `runs/` and re-run
   `runs/make_evidence_index.py`.
