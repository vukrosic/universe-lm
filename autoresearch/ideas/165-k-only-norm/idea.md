---
id: 165-k-only-norm
status: reviewing
round: 1
updated: 2026-06-14T06:24:38Z
transfer-risk: low
plain: Apply RMS normalization to the key vectors only (not the queries) before the attention score is computed — start with the standard scale so step-0 is byte-identical to the baseline.
---

# 165 — K-Only RMSNorm (Asymmetric QK Pre-Softmax Normalization, K-Side)

## Source
- 016-qk-norm (WIN, tiny1m3m) — applied RMSNorm to *both* Q and K. The WIN was Δ -0.014 vs both ctrls; pass-bar -0.005 cleared by ~3×.
- 162-q-only-norm (round-1, in pipeline) — applied RMSNorm to Q *only*; tests whether the Q-side is the binding axis.
- 165 is the missing mirror: K-side only. Together with 162, this is a clean 3-way orthogonal axis test: Q-only (162) / K-only (165) / QK (016).
- Cohere Command-R / R+ (2024) and Gemma 2 ablation reports discuss asymmetric QK normalization tradeoffs (Q-normalized vs K-raw vs both-raw vs both-normalized).
- Henry et al. "QKNorm: Mitigating Transformer Attention Sink" (arXiv:2002.12928) — the original symmetric variant; recent ablations show asymmetric can match at half the parameter cost.

Distinct from 016 (symmetric), from 162 (Q-only), and from the closed per-head / per-layer axes (152, 155, 160, 161).

## Mechanism
Apply `RMSNorm(K)` pre-softmax while leaving Q untouched:
```
Q = Q                          # unchanged
K = RMSNorm(K)                 # K-only normalization
logits = Q @ K^T / sqrt(d_head)
```
With RMSNorm's `weight=1, bias=0` init (the standard `nn.RMSNorm` init), `RMSNorm(x) = x / sqrt(mean(x^2) + eps)` — *not* byte-identical at step 0 (a rescaling). The spec accepts `fp32 max-abs-diff < 1e-3` for rescaling levers (same trade-off as 159-emb-layernorm, 162-q-only-norm). For strict byte-identity, the implementer may multiply by `sqrt(mean(k^2))` post-norm (preserve the per-token RMS). ~6 LoC; +`d_k = 16` params per block (negligible).

## Design sketch
- **File**: `models/layers.py` — `MultiHeadAttention.__init__` adds `use_k_only_norm: bool = False` kwarg; when on, registers `self.k_only_norm = nn.RMSNorm(d_head, eps=1e-6)`. In `forward`, *after* `k = self.W_K(x)` is projected and *before* the QK matmul, apply `k = self.k_only_norm(k)`. Leave `q = self.W_Q(x)` untouched. K-norm applies *post-RoPE* to be consistent with 162's post-RoPE placement.
- **Config flag**: `use_k_only_norm: bool = False` (default off on `LLMConfig`).
- **Step-0 identity**: `nn.RMSNorm(d_head, eps=1e-6)` init has `weight=1, bias=0` ⇒ at step 0, K is rescaled to unit RMS per head-dim. Spec-allowed `fp32 max-abs-diff < 1e-3` tolerance (same as 162). For strict byte-identity, multiply by `sqrt(mean(k^2))` post-norm.
- **Intuition**: 016 won by normalizing *both* Q and K. The lever K-only tests whether the binding axis is the K-side specifically (because K controls what each token "offers" — its "what's available" identity), or whether Q-side is what matters. A K-only win would tell us 016's gain came from K; a null would tell us 016's WIN was carried by Q-side (or the symmetry of joint normalization).
- **Why now**: 162 is currently in the pipeline (Q-only). 165 is the K-mirror. The two together with 016 form a complete 3-way attribution test. The data point we don't have is: does *K alone* carry the gain, or does the *joint* normalization matter? This is the cleanest possible axis test for the 016 win.

## Scale evidence
RMSNorm family is well-validated at 1B+ (LLaMA 3, Qwen 2.5, Mistral). Asymmetric QK normalization is used in Cohere Command-R (35B+) and discussed in Gemma 2 ablation reports. Transfer risk is **low** (≥100M source scale, multiple production validations of the QK-norm family; the *K-only* axis is a sub-claim but the normalization primitive is well-tested).

## Why it's worth a slot
A win would tell us the *K-side* normalization is the binding axis (orthogonal to 016's combined QK gain). A null would tell us 016's WIN was carried by Q-side or the joint symmetry. Either result closes the QK-norm-attribution axis at 0.94M and tells future per-Q-shape levers whether to invest in Q-side (likely) or K-side. This is the cleanest possible null-or-win test in the current closed set — the K-mirror of 162 is the missing data point.

## Plan

- **Files**
  - `configs/llm_config.py` — add `use_k_only_norm: bool = False` on `LLMConfig` (next to `use_q_only_norm` at line 577); add `@dataclass Tiny1M3MKOnlyNormConfig(Tiny1M3MConfig)` with `use_k_only_norm: bool = True` (sibling of `Tiny1M3MQOnlyNormConfig`).
  - `models/layers.py` — `MultiHeadAttention.__init__` adds `use_k_only_norm: bool = False` kwarg (declared next to `use_q_only_norm` at line 853); when on, registers `self.k_only_norm = nn.RMSNorm(self.d_k, eps=1e-6)` at the construction site next to `q_only_norm` (lines 1039-1041). In `forward`, override the symmetric QK-norm path in the three branches (lines 2117-2144):
    - `use_nope or use_cope`: `K = self.k_only_norm(K)`, Q stays raw (Q-side is the symmetric partner of 162 with sides swapped).
    - `use_qk_norm_post_rope`: K goes through `k_norm(self.rotary(...))` iff `use_k_only_norm`, else symmetric.
    - default (pre-RoPE): `K = self.rotary(self.k_only_norm(K))`; Q is untouched (no `q_norm` call). Mutually exclusive with `use_q_only_norm` (asserted in forward, matching the 162/016 mutual exclusion). Also thread through the MoA `extra_K` branch at lines 2437-2450.
  - `models/layers.py` — `TransformerBlock.__init__` adds `use_k_only_norm: bool = False` kwarg (next to `use_q_only_norm` at line 3080) and passes it through to MHA at line 3490.
  - `models/llm.py` — `MinimalLLM.__init__` reads `self.use_k_only_norm = getattr(config, "use_k_only_norm", False)` at line 441 (next to `use_q_only_norm`) and pass-throughs it into both `TransformerBlock(...)` constructors at lines 685 and 941.

- **Config flag**: `use_k_only_norm: bool` (default off).

- **Step-0 identity**:
  - Flag OFF ⇒ no `k_only_norm` module is registered, no branch is taken, baseline path bit-identical (max-abs-diff 0.0 on a 16-token forward at seed 42, the same proof structure as 162's plan).
  - Flag ON ⇒ `nn.RMSNorm(d_k=16, eps=1e-6)` weight=1, bias=0 init → at step 0 K is rescaled to unit RMS per head-dim (spec-allowed fp32 max-abs-diff < 1e-3 tolerance, same trade-off as 159/162).

- **CPU build-smoke** (the daemon's `MinimalLLM(C())` check):
  - `MinimalLLM(Tiny1M3MConfig())` → 949,056 params (baseline).
  - `MinimalLLM(Tiny1M3MKOnlyNormConfig())` → +192 params (one `nn.RMSNorm(d_k=16)` weight × 12 blocks; bias is None by default).
  - The 165 build-smoke must call `MinimalLLM(C())` cleanly before any GPU time is spent (the daemon's `_box_smoke.py` wraps this).

- **Run command**:
  - Artifact: `_arq_165-k-only-norm.py` at repo root, imports `Tiny1M3MKOnlyNormConfig as C` from `configs.llm_config`, dispatches `train_llm.main()` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`.
  - Job: `python _arq_165-k-only-norm.py` with `JOB_TIMEOUT=12m` (tiny1m3m runs in ~2-6 min; the cap keeps a hung treatment from burning the box for 40 min).
  - Descriptor: `autoresearch/ideas/165-k-only-norm/run.json` — `{"name": "165-k-only-norm", "arq_file": "_arq_165-k-only-norm.py", "job_timeout": "12m"}`.
  - Val loss is read from the run's log via `grep "val_loss" ~/arq/logs/165-k-only-norm.log` (or whatever name the daemon assigns; the per-idea run name follows the `NNN-<slug>` convention).

- **Pass/fail bar** (mirror of 162's framing, since the two experiments are the clean 3-way orthogonal axis test):
  - **PASS (K-side carries 016):** treatment val ≤ 016-qk-norm's recorded val by ≥ 0.005. Win message: "K-only matches or beats the symmetric QK-norm win ⇒ K-side normalization is the binding axis."
  - **NULL (Q-side / symmetry carries 016):** |treatment val − ctrl val| < 0.005 against the bare no-norm ctrl (not 016). Null message: "K-only ≡ no-norm at 0.94M ⇒ 016's WIN came from the Q-side normalization or the symmetry, not from K."
  - **DRIFT (lever harmful):** treatment val ≥ ctrl + 0.005. Drift message: "the K-side rescaling disturbs a useful prior."
  - Crash / NaN / OOM → `needs-recode` (round 1, inside budget).
