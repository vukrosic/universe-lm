---
id: 196-ffn-glu-mish
status: repitching
round: 1
updated: 2026-06-15T08:31:31Z
transfer-risk: med
plain: Swap the FFN gating activation from SiLU (in SwiGLU) to Mish (MishGLU) — a smoother, non-monotonic alternative that may help the FFN's gradient flow.
---

# 196 — MishGLU FFN (Mish-Gated Linear Unit, SiLU → Mish in SwiGLU)

## Source
- Shazeer, "GLU Variants Improve Transformer" (2020, arXiv:2002.05202) — the original SwiGLU paper. The paper also tests "GEGLU" (GELU-gated) and finds SwiGLU slightly better at 1.1B-3.9B on T5-style pretraining. MishGLU is not in the original paper but is a natural variant (Mish instead of SiLU as the gating activation).
- Misra, "Mish: A Self Regularized Non-Monotonic Activation Function" (2019, arXiv:1908.08681) — the Mish activation: `Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))`. Mish is *smoother* than SiLU (`x * sigmoid(x)`) in the sense that it has a non-monotonic region for `x < 0` (small negative inputs get *amplified* slightly before the gate kicks in).
- In-repo context: 170-swiglu-ffn (null at tiny1m3m) — closed the *SwiGLU* form (SiLU-gated GLU) at 0.94M. 196 is the *MishGLU* form (Mish-gated GLU), a different gating activation. The closed SwiGLU suggests that the *GLU-axis* (gated FFN) doesn't bind at 0.94M, but the *gating-activation* axis (SiLU vs Mish) has not been directly compared. Mish has a *non-monotonic* region (for `x < 0`, Mish first grows, then decreases), which can provide a stronger gradient signal on the negative-gate axis.
- 153-relu2-ffn (null) — ReLU² (x · ReLU(x)) in FFN. Different activation (Mish is smoother, non-monotonic; ReLU² is non-smooth, monotonic).
- 157-conv-ffn (null), 158-gau (null), 156-moa (null) — other FFN-side levers, all closed at 0.94M. 196 is in the *FFN-gating* family, which is partially closed by 170.

## Mechanism
Standard SwiGLU FFN:
```
def ffn(x):
    gate = silu(W_gate @ x)   # W_gate: [d_model, d_ff]
    val = W_val @ x            # W_val: [d_model, d_ff]
    out = (gate * val) @ W_out.T  # W_out: [d_ff, d_model]
    return out
```
With MishGLU:
```
def ffn(x):
    gate = mish(W_gate @ x)
    val = W_val @ x
    out = (gate * val) @ W_out.T
    return out
```
`Mish(x) = x * tanh(softplus(x))`. The key property: for `x < 0`, Mish first *grows* (small negative x → tanh(softplus(x)) ≈ 0.6 → Mish(x) ≈ 0.6x, which is *less* negative), then *decays* (large negative x → tanh(softplus(x)) → 0 → Mish(x) → 0). This non-monotonic region is the key difference from SiLU (which is monotonic for `x < 0`).

**Step-0 byte-identity**: with `W_gate` initialized by Kaiming, the gate input is `W_gate @ x ≈ N(0, 1)` per component. `Mish(0) = 0 * tanh(softplus(0)) = 0 * tanh(ln(2)) ≈ 0`. `SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0`. Both activations are 0 at the origin. The *derivatives* differ: `Mish'(0) ≈ 0.6`, `SiLU'(0) ≈ 0.5`. So at step 0, the activations have *different* derivatives, but the *function values* are both 0 (since the gate is centered at 0). The product `gate * val = 0 * val = 0` at step 0, so the FFN output is 0 at step 0. **Step-0 byte-identity is exact in the FFN output (both forms give 0)**. The derivatives differ, which means the gradient is different at step 0, but the forward is the same.

For the **2/3-trick** (gate is `silu(0) = 0` ⇒ step-0 silent ⇒ bit-identical baseline at step 0), MishGLU also gives `mish(0) = 0` ⇒ step-0 silent ⇒ bit-identical to the *un-gated* baseline at step 0. With the gated init, the optimizer grows the gate from 0 to engage MishGLU.

**The lever is step-0 byte-identical to the gated baseline** (the standard 2/3-trick gating, with `mish(0) = 0` instead of `silu(0) = 0`). The forward at step 0 is bit-identical; the gradient at step 0 differs (Mish'(0) ≈ 0.6 vs SiLU'(0) ≈ 0.5, both small but different).

## Design sketch
- **Files**:
  - `models/components.py` (or `models/layers.py`) — add `MishGLUFeedForward` module. The module is structurally identical to `SwiGLUFeedForward` (already imported in `models/layers.py:5`) but uses `mish` instead of `silu`.
  - `models/llm.py` — add `use_mish_glu: bool = False` to `MinimalLLM.__init__`. When `True`, replace the `ffn` module with `MishGLUFeedForward`.
  - `configs/llm_config.py` — add `use_mish_glu: bool = False` to `LLMConfig`. Add `Tiny1M3MMishGLUConfig(Tiny1M3MConfig)` with `use_mish_glu: bool = True`.
- **Config flag**: `use_mish_glu: bool = False`.
- **Param count**: same as SwiGLU (3 matrices × d_model × d_ff = 3 × 64 × 256 = 49152 params). No new params.
- **Intuition (why it might lower val loss)**: Mish's non-monotonic region provides a *stronger gradient signal* on the negative-gate axis than SiLU. For a gating unit, the negative-gate axis is where the FFN *suppresses* features; a stronger gradient on this axis means the model can more quickly learn to *suppress* irrelevant features. The closed SwiGLU suggests the *GLU-axis* doesn't bind at 0.94M, but the *gating-activation* axis (Mish vs SiLU) is a different choice within the GLU family. MishGLU is a published variant (used in some 2024+ architectures) that has not been directly tested at our tier.
- **Why it might bind at 0.94M where SwiGLU didn't**: the closed SwiGLU used SiLU as the gating activation. The 2/3-trick gating with SiLU gives a *silu(0) = 0* step-0 silent path; the gate grows over training to engage SwiGLU. With Mish, the gate grows over training to engage MishGLU; the difference is the *shape* of the gating function for non-zero gate values. At 0.94M, the FFN has only 49152 params and 92 update steps; the *gating shape* is a subtle lever that the model may or may not be able to exploit. Mish's non-monotonic region is a *structural* difference from SiLU that the model can in principle exploit.

## Scale evidence
- SwiGLU (Shazeer 2020) — 1.1B-3.9B on T5-style pretraining. The original paper tested SwiGLU, GEGLU, and ReGLU, but not MishGLU.
- Mish (Misra 2019) — 30M-200M on image classification. The Mish activation is well-validated.
- MishGLU is a *compositional* lever (Mish × GLU) that has not been directly tested at scale, but each component is validated.
- **Transfer-risk: med** — the lever is a composition of two validated components (Mish, GLU), but the composition is novel. The closed in-repo analog (170 SwiGLU) is the *compositional* parent; if 170 didn't bind, 196 may also not bind.

## Why it's worth a slot
The bet, in one sharp sentence: **SwiGLU (170) closed null at 0.94M, suggesting the *GLU-axis* doesn't bind at this tier, but the *gating-activation* axis (SiLU vs Mish) is a separate choice that has not been directly tested** — Mish's non-monotonic region is a structural difference from SiLU that the model can in principle exploit for the *suppress-irrelevant-features* axis; a null at 0.94M would close the *gating-activation* axis (SiLU, Mish, and likely all smooth activations are equivalent at this tier), and a win would give a smoother-derivative FFN lever.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- 170-swiglu-ffn (null) — *SwiGLU* form. 196 is *MishGLU* form. Different gating activation.
- 153-relu2-ffn (null) — ReLU² in FFN. Different activation.
- 157-conv-ffn (null) — conv in FFN. Different mechanism.
- 156-moa, 158-gau (null) — different FFN-side levers.
- FFN-activation family is closed at 0.94M (153 + 170). 196 extends the family with Mish.
