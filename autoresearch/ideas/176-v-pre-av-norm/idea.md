---
id: 176-v-pre-av-norm
status: revising
round: 1
updated: 2026-06-15T01:59:50Z
transfer-risk: med
plain: Apply RMSNorm to the value vectors (V) before they get multiplied by attention weights, with learnable gain starting at 1.0 so step-0 is identical to the baseline.
---

# 176 — Pre-AV V RMSNorm (RMSNorm on Value Vectors Before Attention-Output Matmul)

## Source
- Mechanism is the natural V-side analog of 016-qk-norm (RMSNorm on Q
  and K before QK^T, **WIN** at tiny1m3m with Δ=-0.0138/-0.0185). The
  lever family is "normalize the attention-input tensors before they
  interact": QK-norm normalizes Q and K pre-softmax; 176 normalizes V
  pre-AV.
- In-repo context:
  - **016-qk-norm WIN** at tiny1m3m (Δ -0.0138/-0.0185, ≫ gap 0.0047).
    The win tells us pre-interaction normalization is a productive
    lever family at our tier.
  - **162-q-only-norm NULL** (Δ -0.0043 inside band) and
    **165-k-only-norm NULL** (Δ -0.0293, plan PASS bar missed by 0.031).
    The two single-side nulls together with the joint WIN tell us the
    QK-norm WIN was carried by the *joint* QK symmetry, not by either
    side alone. V-side is a separate axis: V has no symmetry partner
    in the pre-AV stage (only V → AV), so V-norm can act independently.
  - **160-rms-gain-per-head NULL** (Δ -0.0023 inside |Δ|<0.005 plan
    band): post-AV per-head gain is closed. 176 is *pre-AV* per-head
    V norm — distinct tensor location.
- V normalization is not in `closed.md`. The lever is novel at our
  tier and a natural extension of the 016 family.

## Mechanism
Standard attention:
1. Q = x @ W_Q, K = x @ W_K, V = x @ W_V.
2. scores = softmax(QK^T / √d_k).
3. out = scores @ V (per head) → concat → x @ W_O.

With V pre-AV RMSNorm:
1. Q, K, V same as above.
2. **V ← RMSNorm(V) · γ_h** where `γ_h ∈ ℝ^{d_k}` is a per-head learnable
   gain (init 1.0). RMSNorm: `V ← V / √(mean(V²) + ε) ⊙ γ_h`. No
   mean subtraction (RMS, not LayerNorm).
3. scores, out same as above.

**Step-0 bit-identical**: `γ_h = 1` for all heads (init) ⇒ `V ← V / √(mean(V²) + ε) ⊙ 1 = V / √(mean(V²) + ε)`. This is NOT bit-identical — V is rescaled. So `γ_h = 1` doesn't give us identity.

**Correct identity init**: scale RMSNorm so that with `γ_h = 1`, output equals input. Define a "1-RMS" RMSNorm that divides by `1.0` when the input is at unit RMS — equivalent to setting `γ_h = √(mean(V²) + ε)` initially. But this is non-trivial.

**Cleaner init**: make the entire RMSNorm *optional* and gate by a learnable scalar `α_h` init 0:
```
V_out = (1 − α_h) · V_in + α_h · RMSNorm(V_in) · γ_h
```
With `α_h = 0` for all heads (init), `V_out = V_in` exactly — bit-identical. The lever's knob is α_h: pushing it positive introduces V normalization; α=1 gives full RMSNorm(V)·γ_h. The transition is smooth and the lever can be partial.

**Even cleaner**: parameterize the gate as `α_h = sigmoid(α_raw_h)` so `α_h ∈ (0, 1)`, init `α_raw_h → −∞` for α→0. Use a large negative init (e.g. -10) so `α_h ≈ 5e-5` at step 0. This is bit-identical up to a tolerance of ~5e-5 × |V|, which is below the fp32 noise floor used in step-0 identity assertions. Or, simpler, use the direct linear `α_h = clamp(α_raw_h, 0, 1)` with init 0 and the **straight-through estimator** for backward pass — gives bit-identical forward and clean gradients.

For simplicity, we use `α_h = relu(α_raw_h)` (init 0 ⇒ α=0) with the understanding that the lever can only grow V normalization, not reverse it. This is the cleanest bit-identical-init.

## Design sketch
- **Files**:
  - `models/layers.py` — `MultiHeadAttention.__init__`: add
    `use_v_rmsnorm: bool = False`. Add
    `self.v_rmsnorm_alpha = nn.Parameter(torch.zeros(n_heads))` (init 0)
    and `self.v_rmsnorm_gain = nn.Parameter(torch.ones(n_heads, d_k))`
    (init 1.0). Wire into the existing parameter list.
  - `MultiHeadAttention.forward`: after the V projection, apply
    `V = (1 − relu(α_h)) · V_in + relu(α_h) · RMSNorm(V_in) · γ_h`
    per head, then continue with scores @ V as usual.
  - `configs/llm_config.py` — add `use_v_rmsnorm: bool = False`.
  - `models/llm.py` — pass through the kwargs at the existing MHA
    construction sites.
- **Config flag**: `use_v_rmsnorm: bool = False` (off by default,
  baseline path bit-identical).
- **Step-0 byte-identical**: at init, `α_h = 0` for all heads
  (relu(0) = 0) ⇒ `V = (1 − 0) · V_in + 0 · ... = V_in` exactly ⇒
  **byte-identical to baseline at step 0 (max-abs-diff = 0.0)**.
- **Intuition (why it might lower val loss)**: 016-qk-norm WIN
  (pre-softmax normalization) is the strongest evidence that
  pre-interaction normalization helps at this tier. V is the
  third tensor in the attention pipeline (Q, K are inputs to
  softmax; V is the values that get weighted). Pre-AV V
  normalization controls the *magnitude* of the attention output
  before the W_O projection, which is a fresh axis the closed
  per-head-attention-shape levers (152, 155, 160, 162, 165, 166)
  don't cover. The QK-norm WIN tells us normalization-on-attention-
  tensors is a productive family; V is the missing tensor.
- **LoC**: ~30 (parameter + apply-RMSNorm-with-gate). Uses the
  existing `nn.RMSNorm` from `torch.nn` (PyTorch ≥2.4 already used
  in this repo per `models/layers.py:340, 879`).

## Scale evidence
- 016-qk-norm WIN at tiny1m3m is the *closest* scale evidence:
  Δ -0.0138/-0.0185 tells us pre-interaction normalization helps
  at this tier. 176 extends the family to V.
- V-side normalization is less commonly cited in the literature
  than Q/K normalization. The closest analog is "QKNorm" applied
  to all three of Q, K, V in some recent models (Chameleon, etc.),
  but V normalization is rarely isolated.
- 160-rms-gain-per-head NULL (post-AV gain) suggests the post-AV
  axis is closed. 176 is pre-AV, a structurally different axis.
- 162-q-only-norm NULL and 165-k-only-norm NULL both tell us
  single-side Q/K normalization doesn't fire. V has no symmetry
  partner so this argument doesn't apply.
- **Transfer risk: med** (extends 016 to V; mechanism is novel
  at our tier and not directly validated at ≥100M. The bet is
  that 016's win extends to V because V has the same role as
  Q/K in the attention pipeline: it's an attention-input tensor
  that gets combined linearly with the others.)

## Why it's worth a slot
The bet: 016's WIN says pre-softmax normalization helps; 176 tests
the V-side extension. V is the only attention-input tensor not yet
tested for normalization. The gate-init-α=0 trick gives clean
bit-identity at step 0. We expect Δval ∈ [-0.005, -0.020] (modest,
similar to the 016 family). A null tells us V is structurally
different from Q/K (Q and K interact via dot product, V is just
weighted-averaged — there's no "norm of V affects the geometry"
argument) and the pre-interaction normalization family is exhausted
at 0.94M. A win unlocks the V-norm axis at Phase-2 where each head
has more gradient signal to develop a useful gate-α value.
