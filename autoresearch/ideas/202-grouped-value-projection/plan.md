# Plan — 202 V-Only Soft-Blend Probe (Isolate V-Sharing From K-Sharing)

## Flag
- `use_grouped_v: bool = False` — master switch (default OFF ⇒ baseline path bit-identical).
- `v_group_size: int = 2` — heads per group; G = `n_heads // v_group_size`.

Files / lines:
- `configs/llm_config.py` — `use_grouped_v: bool = False`, `v_group_size: int = 2` (in the residual / MHA lever cluster near the existing 178-mqa-gated and 207-wo-lowrank comments). New `Tiny1M3MGroupedVConfig(use_grouped_v: bool = True)`.
- `models/layers.py` — `MultiHeadAttention.__init__` adds the two kwargs; allocate `W_V_group` (`nn.ParameterList` of G `nn.Linear(d_model, d_k, bias=False)` initialized to the in-group mean of per-head W_V weights) and `v_group_alpha ∈ R^H` (init -25.0, so `σ(α) ≈ 1.4e-11` and the blend is numerically 0 at step 0 → bit-identical to baseline within fp32 noise). New forward branch blends V per head right after the GQA repeat_interleave (so V is in `[B, T, H, d_k]` layout).
- `models/llm.py` — `getattr(config, "use_grouped_v", False)` + `v_group_size` pickup at construction; pass-through into both `TransformerBlock` sites (the standard block and the YOCO upper-half block, mirroring the 178 plumbing).

## Change

Per head h, blend the per-head V projection with a per-group shared V projection via a per-head sigmoid(α_h) gate:

```
V_h_eff = (1 − σ(α_h)) · V_h_local + σ(α_h) · V_group_g(x)
```

where g = h // v_group_size is the head's group and `V_group_g(x) ∈ R^{d_k}` is the output of a fresh group-shared projection `W_V_group_g ∈ R^{d_k × d_model}`. K is **never touched** — every head keeps its own W_K_h, so the K-axis is the held-out implicit control.

Init (`step-0 byte-identical to baseline`):
- `α_h = -25.0` ⇒ `σ(-25) ≈ 1.4e-11` ⇒ `V_h_eff ≈ V_h_local` (well below fp32 precision; max-abs-diff = 0.0 vs the no-flag baseline).
- `W_V_group_g = mean(W_V_h for h in group g)` — for tiny1m3m with `n_kv_heads = 2, n_heads = 4`, this collapses to the per-KV-head W_V slice of `qkvo_proj` (each group has one KV head in the standard MHA path). The init uses the in-place slice view, not a copy, to keep construction deterministic and RNG-free (same alignment pattern as 178's zero-init shared K/V — keeps the `qkvo_proj` random init aligned with the no-flag baseline).
- K projection is untouched: every head keeps its own W_K_h via the merged qkvo_proj's K slice.

Forward site (`models/layers.py`, post the GQA repeat_interleave at the current line ~3436, pre-transpose):
```python
if self.use_grouped_v:
    G = self.n_heads // self.v_group_size
    # Stack G Linear(d_model, d_k) into one matmul to avoid Python loop
    W_stack = torch.cat([W for W in self.W_V_group], dim=0)  # [G*d_k, d_model]
    V_group = F.linear(x, W_stack).reshape(  # [B, T, G*d_k]
        batch_size, seq_len, G, self.d_k
    )
    # [B, T, G, d_k] → [B, T, H, d_k] via per-group repeat
    V_group_per_head = V_group.repeat_interleave(self.v_group_size, dim=2)
    alpha = torch.sigmoid(self.v_group_alpha).view(1, 1, self.n_heads, 1)
    V = (1.0 - alpha) * V + alpha * V_group_per_head
```

**LoC budget (well under 200).** ~6 LoC config flags + comments + ~30 LoC for the MHA init (W_V_group allocation + in-group-mean init) + ~10 LoC for the forward blend + ~6 LoC for the TransformerBlock / MinimalLLM pass-through + ~10 LoC for the `Tiny1M3MGroupedVConfig` subclass + ~20 LoC for the `_arq_*.py` stub = ~80 LoC. Well under the 200 LoC budget.

**Why α_init = -25, not -10 (reviewer finding 1).** σ(-10) ≈ 4.5e-5. At fp32, `(4.5e-5) · (V_group − V_h_local)` is non-zero in the V_h_eff calculation, so the step-0 forward has a non-trivial deviation from baseline. σ(-25) ≈ 1.4e-11 is well below fp32 precision — `V_h_eff ≈ V_h_local` exactly in fp32 arithmetic, satisfying the structural step-0 byte-identity test. The structural intent ("start at no-sharing, let α learn") is identical; only the gate init constant changes from -10 to -25.

## Control
- **Control**: `Tiny1M3MConfig` (the plain baseline), seed 42, no flags. The daemon owns the ctrl per `RUN-CONTRACT.md`.
- **Treatment**: `Tiny1M3MGroupedVConfig` (subclasses `Tiny1M3MConfig`, sets `use_grouped_v=True`, `v_group_size=2`).
- **Tier**: `tiny1m3m` (0.94M params · 3M tokens). Single tier per `PIPELINE.md`.
- **Seed**: 42, one seed only. A sub-noise effect is logged inconclusive and the run moves on — no "add seeds to confirm" (per the one-seed-only rule).
- **Champion baseline**: `Tiny1M3MAlibiConfig` (val 6.2403 ± 0.04) is the active champion. The 202 idea explicitly does **not** stack on it — the lever subclasses the plain `Tiny1M3MConfig` so the read-out isolates the V-axis from a clean tiny1m3m baseline (no compounding).

## Cost
- **Params**: −4·d_model·d_k (per-head V projs replaced by G·d_model·d_k group V projs) + G·d_model·d_k (the group V projs themselves) + H (the α scalars) = −4·64·16 + 2·64·16 + 4 = −4096 + 2048 + 4 = −2044 per block × 12 blocks = **−24,528 params (−2.6% of 0.94M)**.

  Wait — that's wrong. Let me recount.

  At tiny1m3m: `n_heads = 4`, `n_kv_heads = 2`, `d_k = 16`, `d_model = 64`. Per-block V projection under standard MHA = `n_heads * d_k * d_model = 4 * 16 * 64 = 4096` params (assuming head-decoupled V). The merged `qkvo_proj` shares the V slice across KV heads, so the per-block V cost is `n_kv_heads * d_k * d_model = 2 * 16 * 64 = 2048` params (current implementation).

  Under `use_grouped_v`: the standard V slice in `qkvo_proj` is **kept** (K is unchanged), so V projection stays at 2048 params. The new group V projections add G·d_model·d_k = 2·64·16 = 2048 params (new). Plus H α scalars = 4 params. **Net: +2052 per block × 12 = +24,624 extra params (+2.6% of 0.94M)**.

  (The original idea sketch said −24,528 because it assumed replacing the per-head V slice entirely with G group slices; with the merged-proj + GQA structure actually in the code, the per-head V slice is the per-KV-head slice (only `n_kv_heads * d_k` wide), so we keep that AND add the new G group projections. Same magnitude, opposite sign — +2.6% instead of −2.6%. The sign doesn't affect the probe: the per-head V is still bit-identical at step 0, the optimizer can grow the group-V branch, and the gate trajectory is the same signal.)

- **FLOPs**: per forward per block, extra 1 matmul `[B*T, d_model] @ [d_model, G*d_k]` ≈ `B*T*G*d_model*d_k` mul-adds. Tiny1m3m B*T ≈ 4096 (B=2, T=2048), so 4096·2·64·16 ≈ 8.4M extra mul-adds per block, ~100M total per training step. Within the ±5% FLOPs noise band.
- **Memory**: extra G·d_k·d_model = 2 KB per block, ~24 KB total. Negligible.
- **Wall-clock**: ~12 min for the treatment on the V100 (the daemon's default `job_timeout` covers this; the extra G·d_k·d_model matmul is in the noise).

## Run
- **Command**: `python _arq_202-grouped-value-projection.py` (seed 42, dataset `processed_data/pretrain_1B`, `--warmup false`).
- **Tier / seed**: `tiny1m3m` / 42.
- **Wall-clock expected**: ~12 min (default `job_timeout=12m`).
- **Primary signal**: per-block, per-head final `α_h` values (H=4 per block × 12 blocks = 48 scalars). The runner must dump `v_group_alpha_final ∈ R^{12 × 4}` at end of training (single tensor dump, cost negligible).
- **Secondary signal**: val loss Δ vs baseline (informative but not deciding — the σ(α) trajectory is the deciding metric, not val noise).
- **Pass/fail bar (from idea.md)**:
  - (a) All α_h stay near 0 (`σ(α) < 0.05` throughout): V-axis closed mechanistically at 0.94M; family dead — append to closed.md.
  - (b) At least one α_h moves off 0 (`σ(α) > 0.05`) but val Δ inside band: V-axis has a real per-head gradient that val landscape didn't reward. High-info-value null — separates V from K (distinct from 178-mqa-gated which couldn't separate).
  - (c) At least one α_h moves AND val Δ < band: real win → promote to lever idea with sharper scope; split the closed.md GQA entry (closed at fixed-group, gate-able at variable group).
  - (d) Mixed (some blocks move, some don't): layer-dependent V-redundancy map; read out and log; revisit at 12L+ tier.
- **Self-check (per `code-implementer.md` §5)**:
  1. `MinimalLLM(C())` constructs on CPU (build-smoke). `_arq_202-grouped-value-projection.py` defines `C` at module top level.
  2. Flag OFF (default): `MinimalLLM(Tiny1M3MConfig())` and `MinimalLLM(C(use_grouped_v=False))` produce identical forward outputs within fp32 noise. (`σ(-25) ≈ 1.4e-11` is below fp32 precision; the blend coefficient is bit-exact zero in fp32 arithmetic.)
  3. Flag ON: `MinimalLLM(C())` constructs, registers G·d_model·d_k + H extra params per block (verified via `param_count(Tiny1M3MConfig) + 24,624 == param_count(C)`).
  4. `plan.md` pass/fail bar matches `idea.md` (4 outcomes (a-d), primary = α_h trajectory).
  5. `run.json` + `_arq_202-grouped-value-projection.py` written, daemon-importable, defines top-level `C`.
