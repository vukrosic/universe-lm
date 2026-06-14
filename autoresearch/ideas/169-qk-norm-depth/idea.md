---
id: 169-qk-norm-depth
status: needs-run
round: 1
updated: 2026-06-14T10:00:14Z
transfer-risk: low
plain: Keep the QK RMS-norm from 016 (which won) but give each block its own learnable scale on top, starting at one — tests whether different blocks want different normalization strengths.
---

# 169 — Depth-Conditional QK-Norm (Per-Block Learnable Scale on the 016 Winner)

## Source
- 016-qk-norm (WIN, tiny1m3m) — applied RMSNorm to *both* Q and K with a single shared per-head learnable scale. Δ -0.014 vs both ctrls; pass-bar -0.005 cleared by ~3× (closed entry at `autoresearch/closed.md:60`: `trt=6.3906 vs ctrls 6.4044/6.4091`, Δ -0.0138/-0.0185).
- 162-q-only-norm (closed null at 0.94M, `closed.md:117`: trt=6.4303 vs cached baseline 6.4346±0.0458, Δ=-0.0043 inside band) — applies RMSNorm to Q *only*; tests whether the Q-side is the binding axis.
- 165-k-only-norm (just filed) — applies RMSNorm to K *only*; the K-mirror of 162.
- 169 is a different axis: the *scale* parameter of the RMSNorm is *per-block* (one per layer) rather than per-head-shared. The bet: different blocks may want different normalization strengths (e.g. shallow blocks want less, deep blocks want more, or vice versa).
- "NormFormer" (Shleifer et al. 2021) introduced per-layer learnable gains on attention output; the depth-conditional norm family is well-validated in the FFN / attention output space.
- Per-block QK-norm gain is the same idea applied to the 016 winner.

Distinct from 016 (per-head-shared scale), from 152/155/160 (per-head per-layer scalars on different tensors), and from 161 (per-layer temperature — closed null because per-layer temperature fights the canonical QK scale prior).

## Mechanism (Option A — locked)
Keep 016's per-head RMSNorm structure intact; add a **per-block scalar gain** applied **post-RMSNorm** and **pre-QK-matmul**, shared between Q and K within a block:
```
for each block l:
    Q_l = self.q_norm(Q_l) * alpha_l          # q_norm = per-head nn.RMSNorm(d_k) [016's lever]
    K_l = self.k_norm(K_l) * alpha_l          # k_norm = per-head nn.RMSNorm(d_k) [016's lever]
    logits_l = Q_l @ K_l^T / sqrt(d_head)
```
At init `alpha_l = 1.0` for all 12 blocks ⇒ the multiplicative gain is exactly the identity ⇒ the forward graph is **byte-identical** to 016's step-0 forward (max-abs-diff = 0.0 vs 016, no tolerance needed for this comparison). The optimizer can then adjust per-block scales.

**Why Option A, not Option B**: Option B (replace per-head RMSNorm with per-block RMSNorm) restructures 016's parameterization — different mechanism, different math, +192 params instead of +12, and loses the per-head normalization shape that 016 won with. Option A is a strict extension on top of 016's WIN: same per-head RMSNorm, an additional multiplicative gain with one learnable scalar per block. Mirrors NormFormer's per-layer attention-output gains (Shleifer et al. 2021) applied to the QK-norm output.

## Pass / fail bar
- **Control**: **016-qk-norm WIN config** (`Tiny1M3MQKNormConfig` subclass with `use_qk_norm` enabled — the closed config that produced `trt=6.3906 vs ctrls 6.4044/6.4091`, Δ ≈ -0.014/-0.019, per `autoresearch/closed.md:60`). NOT unmodded-nanogpt — the Δ is depth-conditional-on-016, not "QK-norm yes/no".
- **PASS (depth-conditional binds on top of 016)**: 169 treatment val < 016 control val by ≥ **0.005** (mirrors 016's own plan bar; clears the tiny1m3m ±0.04 box noise band by ≥2×).
- **NULL (shared per-head scale is sufficient)**: |169 val − 016 val| < 0.005 ⇒ 016's per-head-shared scale is sufficient at 0.94M and per-block scaling is not the binding axis.
- **DRIFT (lever harmful)**: 169 val ≥ 016 val + 0.005 ⇒ the additional per-block DOF adds noise rather than signal.
- **CRASH / NaN / OOM** → `needs-recode` (round 1, inside budget).

## Design sketch
- **File**: `models/layers.py` — `MultiHeadAttention.__init__` adds `use_qk_norm_depth: bool = False` kwarg (declared alongside the existing 016 / 162 / 165 knobs at lines ~664-863). When on, in addition to 016's existing per-head `self.q_norm` / `self.k_norm` modules, register **one** scalar `self.qk_norm_scale = nn.Parameter(torch.ones(1))` per MHA (so 12 scalars across 12 blocks — one per block, shared between Q and K within that block). In `forward`, after the existing per-head RMSNorm call and before the QK matmul, apply `q = self.q_norm(q) * self.qk_norm_scale` and `k = self.k_norm(k) * self.qk_norm_scale`. The per-head RMSNorm is preserved (Option A).
- **Config flag**: `use_qk_norm_depth: bool = False` (default off on `LLMConfig`).
- **Step-0 identity**:
  - 169 vs 016 (the chosen control): `qk_norm_scale = 1.0` ⇒ multiplicative gain is exactly the identity ⇒ **byte-identical forward, max-abs-diff = 0.0** (no tolerance needed).
  - 169 vs unmodded-nanogpt: the per-head RMSNorm rescaling is the accepted trade-off; spec-allowed `fp32 max-abs-diff < 1e-3` tolerance (same as 016, 162, 165).
- **Mutual exclusion**: at the top of `MultiHeadAttention.forward`, assert (mirroring the existing `assert not (self.use_cope and self.use_qk_norm_post_rope)` at `models/layers.py:1948`):
  ```
  assert not (self.use_qk_norm_depth and self.use_q_only_norm), \
      "use_qk_norm_depth=True is mutually exclusive with use_q_only_norm=True"
  assert not (self.use_qk_norm_depth and self.use_k_only_norm), \
      "use_qk_norm_depth=True is mutually exclusive with use_k_only_norm=True"
  assert not (self.use_qk_norm_depth and self.use_qk_norm_post_rope), \
      "use_qk_norm_depth=True is mutually exclusive with use_qk_norm_post_rope=True"
  ```
  (Q-only and K-only are the 162/165 orthogonal ablations; `qk_norm_post_rope` is 016's pre-existing symmetric path. Combining any of them with per-block scaling restructures the lever and must fail loud.)
- **Intuition**: 016's WIN was on a *shared* per-head scale (= 1.0 init, single weight per head). The hypothesis: different blocks have different attention statistics (shallow blocks have broader attention, deep blocks have sharper attention), so a *single shared* scale may not be optimal. Per-block learnable scales let the model adjust the normalization strength per block. If 169-WIN > 016-WIN, the depth-conditional axis is binding. If 169 ≈ 016, the per-head-shared scale is sufficient.
- **Why now**: 016 is the strongest QK-side win. 162/165 are the *which side* tests (Q vs K vs both). 169 is the *depth-conditional* test: does the per-block scale matter once we have a shared scale? The data point we don't have is whether *block-specific* normalization strength compounds with depth at 0.94M.

## Scale evidence
RMSNorm family is well-validated at 1B+ (LLaMA 3, Qwen 2.5, Mistral). Per-block learnable scales on attention-internal tensors are a sub-claim but the primitive is well-tested. NormFormer's per-layer gains (Shleifer et al. 2021) is the closest direct analog, validated at 100M+ on long-document tasks. Transfer risk is **low** (well-validated primitive, narrow extension of 016's WIN).

## Why it's worth a slot
A win (or marginal Δ on top of 016) would tell us *depth-conditional normalization strength* is a binding axis at 0.94M, suggesting future QK-norm variants should consider per-block scales. A null would tell us 016's shared scale is sufficient at this tier and the depth-conditional axis is not the binding one. The lever is cheap (~15 LoC, +12 scalar params = +0.001% of 0.94M) and provides a clean attribution test for the 016 win.

## Plan

**Files changed**
- `configs/llm_config.py` — add `use_qk_norm_depth: bool = False` field on `LLMConfig` (default off; sits next to the closed `use_q_only_norm` / `use_k_only_norm` / `use_qk_norm_post_rope` siblings at lines ~288-587). Also add a `@dataclass`-decorated `Tiny1M3MQKNormDepthConfig(Tiny1M3MConfig)` subclass with `use_qk_norm_depth: bool = True` at the end of the file (mirroring the 162/165 dataclass pattern at lines ~5315-5353; bare inheritance is broken by the dataclass pitfall that hit 155/159/161).
- `models/layers.py` — add `use_qk_norm_depth: bool = False` kwarg to `MultiHeadAttention.__init__` (declared alongside the 016 / 162 / 165 knobs at lines ~664-863). When on, register `self.qk_norm_scale = nn.Parameter(torch.ones(1))` next to the existing 016 `self.q_norm` / `self.k_norm` modules at lines ~1039-1062. Thread the kwarg through `TransformerBlock.__init__` (the existing site for `use_qk_norm_post_rope` / `use_q_only_norm` near line ~1357). In `MultiHeadAttention.forward`, the existing 016 QK-norm branches (no-RoPE / post-RoPE / default pre-RoPE) each gain a `use_qk_norm_depth` arm that multiplies `q` and `k` by `self.qk_norm_scale` after the per-head RMSNorm and before the QK matmul. The MoA `extra_K` branch (lines ~2437-2450) mirrors the same multiplicative scale on the extra K. Three new `assert not (self.use_qk_norm_depth and self.X)` lines at the top of `forward` (mirroring the `use_cope` / `use_qk_norm_post_rope` assert at line 1948).
- `models/llm.py` — read `use_qk_norm_depth` from `config` (`self.use_qk_norm_depth = getattr(config, "use_qk_norm_depth", False)` next to the existing `use_q_only_norm` read at line ~440) and thread it into the four `TransformerBlock(...)` / `MultiHeadAttention(...)` construction sites (lines ~607, ~685, ~850, ~941 — the sites that currently thread `use_qk_norm_post_rope` / `use_q_only_norm`).

**Flag name**: `use_qk_norm_depth` (off by default).

**Step-0 identity**: flag OFF → no `qk_norm_scale` parameter, no branch taken, baseline path bit-identical. Flag ON at step 0 ⇒ `qk_norm_scale = 1.0` exactly ⇒ **byte-identical to 016's step-0 forward (max-abs-diff = 0.0)** — the reference comparison is 169 vs 016 (the chosen control), not 169 vs unmodded. Per-head RMSNorm rescaling (flag ON vs unmodded) is the accepted trade-off (spec allows `fp32 max-abs-diff < 1e-3`, same as 016/162/165).

**Run command** (per `prompts/runner.md` / `PIPELINE.md`, standard tiny1m3m seed 42):

```bash
cd /root/universe-lm && \
LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
/venv/main/bin/python -m training.trainer \
  --config_class autoresearch.configs.tiny1m3m.Tiny1M3MQKNormDepthConfig \
  --activations "use_qk_norm_depth=True" \
  --seed 42 --steps 3000 --batch_size 32
```

**Reading final val loss**: standard runner prints `val_loss` at the end of training and writes `runs/<run_id>/metrics.json` with the `val_loss` field.

**LoC budget**: ~40 lines total (well under the 200 LoC cap).
