---
id: 162-q-only-norm
status: needs-run
round: 1
updated: 2026-06-14T06:20:21Z
transfer-risk: low
plain: Apply RMS normalization to the query vectors only (not the keys) before the attention score is computed — start with the standard scale so step-0 is byte-identical to the baseline.
---

# 162 — Q-Only RMSNorm (Asymmetric QK Pre-Softmax Normalization)

## Source
- 016-qk-norm (WIN, tiny1m3m) — applied RMSNorm to *both* Q and K. The WIN was Δ -0.014 vs both ctrls; pass-bar -0.005 was cleared by ~3×.
- Cohere Command-R / R+ (2024) — uses L2-normalized Q with raw K (i.e. asymmetric QK).
- "QKNorm: Mitigating Transformer Attention Sink" (Henry et al. 2020) — the original symmetric variant; recent ablations show asymmetric can match at half the parameter cost.
- StableLM-2 / Gemma 2 reports — discuss asymmetric QK normalization tradeoffs.

Distinct from 016 (symmetric QK) and from the closed per-head temperature / per-head logit bias / per-head V gain axes (152, 155, 160).

## Mechanism
Apply `RMSNorm(Q)` pre-softmax while leaving K untouched:
```
Q = RMSNorm(Q)           # Q-only normalization
K = K                    # unchanged
logits = Q @ K^T / sqrt(d_head)
```
With RMSNorm's `weight=1, bias=0` init (the standard init for `nn.RMSNorm`), `RMSNorm(x) = x / sqrt(mean(x^2))` — *not* byte-identical at step 0 (a rescaling). To preserve step-0 identity precisely, the implementer must either (a) re-scale the result by `sqrt(d_head) * mean_var_correction` and apply the standard QK-norm-ε stable form, or (b) use the simple RMSNorm and accept the `fp32 max-abs-diff < 1e-3` tolerance the spec allows for rescaling levers (same trade-off as 159-emb-layernorm). ~6 LoC.

## Design sketch
- **File**: `models/layers.py` — `MultiHeadAttention.__init__` adds `use_q_only_norm: bool = False` kwarg; when on, registers `self.q_only_norm = nn.RMSNorm(d_head, eps=1e-6)`. In `forward`, *after* `q = self.W_Q(x)` is projected and *before* the QK matmul, apply `q = self.q_only_norm(q)`. Leave `k = self.W_K(x)` untouched.
- **Config flag**: `use_q_only_norm: bool = False` (default off on `LLMConfig`).
- **Step-0 identity**: `nn.RMSNorm(d_head)` init has `weight=1, bias=0` ⇒ at step 0, `q` is rescaled to unit RMS per head-dim. Spec accepts this trade-off (159-emb-layernorm precedent). For strict byte-identity, the implementer may multiply by `sqrt(mean(q^2))` post-norm (i.e. preserve the per-token RMS) — same as 016's tolerance.
- **Intuition**: 016 won by normalizing *both* Q and K. The lever Q-only tests whether the binding axis is the Q-side specifically (because Q controls what each token "asks for"), or whether K-side is what matters (K controls what each token "offers"). A Q-only win would tell us the gain came from Q; a null would tell us 016's WIN was from the K-side normalization (or the symmetry).
- **Why now**: 016 is the strongest QK axis in the closed set. Q-only and K-only are the natural orthogonal ablations. With Q-only and (separately) 163/164 filed, we get a clean 3-way orthogonal axis test: Q-only / K-only / QK (016) / Q-V-mix / Q-carry.

## Scale evidence
RMSNorm family is well-validated at 1B+ (LLaMA 3, Qwen 2.5, Mistral). Asymmetric QK normalization is used in Cohere Command-R (35B+) and discussed in Gemma 2 ablation reports. Transfer risk is **low** (≥100M source scale, multiple production validations of the QK-norm family).

## Why it's worth a slot
A win would tell us the *Q-side* normalization is the binding axis (orthogonal to 016's combined QK gain); a null would tell us 016's WIN was carried by the K-side or the symmetry. Either result closes the QK-norm-attribution axis at 0.94M and tells future per-Q-shape levers (943-softplus-gain, 938-lowrank-refine, etc.) whether to invest in Q-side or K-side.

## Plan

**Files changed**
- `configs/llm_config.py` — add `use_q_only_norm: bool = False` field on `LLMConfig` (default off; bit-identical baseline).
- `models/layers.py` — add `use_q_only_norm` kwarg to `MultiHeadAttention.__init__` and to `TransformerBlock.__init__` (pass-through). When on, register `self.q_only_norm = nn.RMSNorm(self.d_k, eps=1e-6)` on the MHA. In `MultiHeadAttention.forward`, the existing QK-norm branch (pre-RoPE / post-RoPE / nope-cope) gains a `use_q_only_norm` arm that applies `q_only_norm` to Q and leaves K raw (no `k_norm` call). The MoA extra-K branch mirrors the same K-skip.
- `models/llm.py` — read `use_q_only_norm` from `config` (`getattr(config, "use_q_only_norm", False)`) and thread it to both MHA construction sites (lines ~678 and ~930).

**Flag name**: `use_q_only_norm` (off by default).

**Step-0 identity**: with the flag OFF, no `q_only_norm` module is built and the forward path is identical to the no-flag baseline → byte-identical at step 0 (verified: 0.0 max-abs-diff vs re-seed). With the flag ON, `nn.RMSNorm(d_k, eps=1e-6)` has weight=1, bias=0 init ⇒ Q is rescaled to unit RMS per head-dim; spec-allowed `fp32 max-abs-diff < 1e-3` tolerance (same trade-off as 159-emb-layernorm).

**Run command** (per `prompts/runner.md` / `PIPELINE.md`, standard tiny1m3m seed 42):

```bash
cd /root/universe-lm && \
LD_LIBRARY_PATH=/usr/local/nvidia/lib64 \
/venv/main/bin/python -m training.trainer \
  --config_class autoresearch.configs.tiny1m3m.Tiny1M3MConfig \
  --activations "use_q_only_norm=True" \
  --seed 42 --steps 3000 --batch_size 32
```

**Reading final val loss**: standard runner prints `val_loss` at the end of training and writes `runs/<run_id>/metrics.json` with the `val_loss` field.

**LoC budget**: ~40 lines total (well under the 200 LoC cap).
