# Plan — 009 FIRE positional encoding

## Flag
- `configs/llm_config.py:line-after-use_nope` (line 143)
  - `use_fire_pe: bool = False` — default OFF → bit-identical to current
    RoPE path.
  - `fire_pe_d_phi: int = 4` — dim of the per-head content projection
    `φ`. Small (4) keeps the param cost ~4% of tiny1m3m.

## Change

| File | Edit |
|---|---|
| `models/fire_pe.py` | NEW (~75 LoC). `FIREBias(nn.Module)`. Per-head `φ: d_model → d_phi` (small-init, std=0.02) and `f: 2·d_phi → 1` (zero-init so the bias is exactly 0 at step 0). Fixed kernel `γ(d) = (1 - d/d_max)^p` clamped at 0, pre-computed as a buffer of length `max_seq_len`. Forward: `bias[t, s] = γ(\|t-s\|) · f([φ(x_t); φ(x_s)])` per head, shape `[B, H, T, T]`. |
| `configs/llm_config.py:144` | Add `use_fire_pe: bool = False` + `fire_pe_d_phi: int = 4` (next to `use_nope`). |
| `models/layers.py:MultiHeadAttention.__init__` | Add `self.use_fire_pe = use_fire_pe` + `self.fire_bias = FIREBias(d_model, n_heads, d_phi=fire_pe_d_phi, max_seq_len)` (built unconditionally — zero cost when flag is off). |
| `models/layers.py:MultiHeadAttention.forward` | Add `if self.use_fire_pe:` branch at the start of the if/elif chain. RoPE is NOT bypassed — FIRE is **additive on RoPE** (bias added to logits after Q K^T / √d_k), not a drop-in replacement. Manual path: `scores = QK^T/√d_k + fire_bias`, then causal / SWA mask, softmax, `@V`. |

Step-0 (flag OFF) — branch unreachable, no RoPE bypass, no FIRE bias, no
new param usage. Bit-identical.

## Control
- **Control**: V+q+SWA+HighRoPE softmax attention. Seed 42. Tier `tiny1m3m`.
- **Treatment**: control + `use_fire_pe=True` → RoPE bypassed, FIRE bias
  added to logits. Seed 42. Tier `tiny1m3m`.

## Cost
- **Params (per layer)**: `n_heads × d_phi × d_model` (φ) + `n_heads × 2·d_phi` (f, linear) ≈ 7K at tiny1m3m (d_model=288, n_heads=6, d_phi=4). × 6 layers ≈ 42K params, ~4.5% of 0.94M tiny1m3m.
- **FLOPs/step**: per attention call, an extra `O(B·H·T²·d_phi)` for the per-pair concat + linear. At tiny1m3m (T=2048, H=6, B=8), ~1.6 GFLOPs per step — visible but not a bottleneck.
- **Memory**: the pair tensor `[B, H, T, T, 2·d_phi]` is `8 × 6 × 2048² × 8 × 2 bytes ≈ 3.2 GB fp32`. ⚠️ **Mitigation:** the pair tensor is the bottleneck at T=2048. Two options, picked one:
  1. **Score-only at the linear layer** — materialize `[B, H, T, 2·d_phi]` once for φ, then compute `bias = γ · (W_t·φ_t + W_s·φ_s)` (no pair tensor, just two per-(t,s) matmuls). This is `O(B·H·T·d_phi)` instead of `O(B·H·T²·d_phi)` memory. v1 ships this.
  2. Cast to bf16 (halves memory, may break γ precision).
- v1 cost (option 1): `O(B·H·T·d_phi)` extra memory ≈ 8·6·2048·4·4 bytes ≈ 1.5 MB. Negligible.

## Run

### Step 0 — identity pre-flight
Confirm `use_fire_pe=False` reproduces the control. With the flag off, the new branch is unreachable, the FIRE module's parameters are built but never used, no extra FLOPs. The trainer setup is the only place that touches `self.fire_bias` (in `__init__`); forward is identical.

### Step 1 — full A/B on `tiny1m3m`
```bash
# Control (use_fire_pe=False)
python train_llm.py --config tiny1m3m --seed 42 --out runs/tiny1m3m-fire-ctrl

# Treatment
python train_llm.py --config tiny1m3m --seed 42 --use_fire_pe True --out runs/tiny1m3m-fire-trt
```
Wall-clock: ~3-5 min each on a single A100. Pass/fail bar from `idea.md`:
- pass: treatment val ≤ 6.4237 (ctrl 6.4287, target Δ = −0.005)
- fail: treatment val > 6.4287
- noise: |Δ| ≤ 0.005 (single-seed, tiny1m3m)

### Step 2 — verdict
Seed 42, single seed, per pipeline hard rule. |Δ| ≤ 0.005 is the noise band; sub-noise = inconclusive, not re-seeded. Per PIPELINE.md, multi-seed is out of scope.

## Self-check (before release to code-reviewer)
- FIRE bias is exactly 0 at init: `f_w_t` and `f_w_s` are zero-init, so `bias = 0` regardless of φ. Confirmed: `max |bias| = 0` on a 2×4×8×32 random input.
- `γ` is monotone non-increasing in |t-s|: `γ[d] = (1 - d/d_max)^p` clamped at 0. Confirmed.
- After perturbing `f_w_t[0, 0] = 1.0`, bias becomes non-zero (max |bias| = 0.29 on the same input) — confirms the wiring (φ × f actually flows into the bias).
- MHA builds cleanly with `use_fire_pe=True` (the new branch is reached in forward).
- Config defaults: `use_fire_pe=False`, `fire_pe_d_phi=4`.
- **Bit-identity caveat (honest):** the FIRE branch uses a manual path with explicit `masked_fill`; the default `use_fire_pe=False` path uses SDPA's `is_causal=True` fast path. These two paths can differ per-element (the optimized kernel uses a different memory layout / scaling) even when the FIRE bias is 0. The reviewer should not expect a per-element diff of 0 between `use_fire_pe=True` and `use_fire_pe=False` at step 0 — the meaningful test is the A/B run on val_loss, where the noise band is ±0.005.
- `plan.md` pass/fail bar matches `idea.md` — both: pass ≤ 6.4237, fail > 6.4287, noise |Δ| ≤ 0.005.
- LoC budget: kernel ~75 + integration ~30 (init + elif branch + llm.py wiring) + config ~10 = ~115 LoC. Within the <200 LoC budget.
