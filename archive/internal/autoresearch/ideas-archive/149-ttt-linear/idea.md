---
id: 149-ttt-linear
status: done
round: 1
updated: 2026-06-13T21:59:30Z
transfer-risk: med
plain: Replace the FFN with a fast-weight linear that updates its own weights from the input on the fly, like a tiny per-sequence fine-tune during the forward pass.
---

# 149 — TTT-Linear (Test-Time Training Linear)

## Source
Sun, Yang, et al. 2024, "Learning to (Learn at Test Time): RNNs with Expressive Hidden States", arXiv:2407.04620. https://arxiv.org/abs/2407.04620. The TTT-Linear layer in §3.2: a fast-weight linear where the fast weights W_f are computed via a single gradient step on a self-supervised reconstruction loss applied to the input. Validated at 1B–7B LM scale in the paper (Tables 2-3).

## Mechanism
A standard FFN block does `y = W2 · σ(W1 · x)`. TTT-Linear replaces the static `W1`/`W2` with *input-conditioned* weights:

1. Pick a self-supervised target: in the TTT paper, `x_target` is a learned projection of `x` (or the next token in sequence mode). For our use, take `x_target = x` (auto-encoding / denoising) — the fast weight is forced to compress-and-reconstruct `x`.
2. Compute a closed-form one-step gradient of `||W_f @ x − x_target||^2` starting from the slow weight `W_slow`:
   - `W_f = W_slow + lr · (x_target − W_slow @ x) ⊗ x / (||x||^2 + ε)`  (Batched-Vectorized-Newton style, closed-form)
3. The output is `W_f @ x` (or wrapped with a non-linearity to be a drop-in FFN replacement).

Identity at step 0: when `lr=0`, the gradient term vanishes and `W_f = W_slow`, so TTT-Linear collapses to a regular linear. Bit-identical to a vanilla linear at `lr=0`. ✓

## Design sketch (how it works + how to build it)
- New file `models/ttt_linear.py` (~120 LoC): `TTTLinear` Module that holds `slow_weight` (the baseline `nn.Linear` weight) and a scalar `lr` parameter (or a fixed 0-init). On forward, computes the closed-form one-step update and applies it. ~120 LoC.
- In `models/layers.py`, add `use_ttt_ffn: bool` to `FeedForward.__init__`. When on, replace the two `nn.Linear` layers with `TTTLinear` (use the same hidden dim; TTT can be 1-layer or 2-layer — start with 1-layer to keep param count parity with a 1x-expansion baseline). The slow-weight is initialized as a normal `kaiming_uniform_` linear so `lr=0` matches baseline.
- `configs/llm_config.py`: add `use_ttt_ffn: bool = False`, `ttt_target: str = "self"` (selects `x_target = x` vs `x_target = shift(x,1)` for next-token prediction), `ttt_lr_init: float = 0.0` (zero-init LR → bit-identical at step 0; ramp to a small positive via warmup).
- The flag-off path is bit-identical: `use_ttt_ffn=False` ⇒ the `if` branch in `FeedForward.forward` falls through to the standard `nn.Linear` path; the TTT module is never constructed (no RNG consumption).
- **Why it should lower val loss at tiny1m3m specifically**: the 0.94M model is severely capacity-bottlenecked. A static FFN must encode "what to do with token t" in fixed weights; TTT-Linear gives the model a *per-input* weight update that costs ~one extra matmul per step (the reconstruction gradient is closed-form, not an extra backward). At short context (T=128) and small d_model (64), the inner matmul is cheap, and the per-input adaptation could compensate for the lack of capacity — a 0.94M model with per-input fast weights behaves like a much larger static model in expectation. The bet is that this is a "free" sample-efficiency boost when the model is undertrained at 92 update steps.
- **Closest neighbor** in our closed list: 110-weight-ema / 134-mega-ema null at 0.94M — but those are weight-EMA regularizers (parameter-averaging), not per-input weight adaptation. TTT's failure mode is different: if `lr` is too large the fast weights thrash; if `lr` is too small it's a no-op. We start with `lr=0` and let the model learn the right LR via the lr parameter or via a schedule.

## Scale evidence
Paper validates TTT-Linear at 1B-7B LM scale, with perplexity matching or beating Mamba/Transformer baselines on Pile data. Smaller-scale behavior is not published. Transfer risk: med — paper is at the right scale (≥100M ✓) and TTT's design point is long-context, but the per-input fast-weight mechanism is independent of context length, so the sample-efficiency bet should still fire at T=128. Risk is that at 0.94M the inner matmul overhead (~3-5% extra compute) eats the benefit.

## Why it's worth a slot
Genuinely new mechanism (input-conditioned fast weights) that we have not filed. Distinct from closed axes: not weight-EMA (110/134), not per-token HP (closed norm zoo / soft-moe 117), not multi-token-prediction (070), not linear-attention (008). The bet: per-input adaptation is a *capacity-multiplier* for a 0.94M model, sample-efficiency is the binding constraint at 92 update steps, and TTT's closed-form inner step means it adds compute but not params. A win would give us a new family (fast-weight layers) for the lab; a null would close the input-conditioned-weight axis at our tier and tell us static FFN is not the binding constraint.

## Plan

**Files touched**

- **NEW** `models/ttt_linear.py` (~115 LoC): `TTTLinear` (drop-in `nn.Linear`
  replacement with per-input closed-form fast-weight update) and
  `TTTFeedForward` (squared_relu FFN whose up-proj is a `TTTLinear`).
- **EDIT** `models/layers.py` (~10 LoC): add `use_ttt_ffn: bool = False`,
  `ttt_lr_init: float = 0.0` to `TransformerBlock.__init__`; add a branch
  in the FFN block to construct `TTTFeedForward` when `use_ttt_ffn=True`.
- **EDIT** `models/llm.py` (~6 LoC): read the two flags from `config`,
  pass-through to both standard `TransformerBlock` and the YOCO upper
  blocks (mirror the existing `use_expert_choice_moe` plumbing).
- **EDIT** `configs/llm_config.py` (~50 LoC): add `use_ttt_ffn` and
  `ttt_lr_init` to the base config (next to `use_expert_choice_moe`),
  plus a new `Tiny1M3MTTTLinearConfig` dataclass at the bottom (mirrors
  `Tiny1M3MExpertChoiceConfig`).
- **NEW** `_arq_149-ttt-linear.py`: trt runner that subclasses
  `Tiny1M3MTTTLinearConfig` and calls `train_llm.main()`.

**Mechanism**

`TTTLinear` holds a standard `nn.Linear` weight (`weight`,
`kaiming_uniform_` init — bit-identical to `nn.Linear` at init) plus a
scalar `ttt_lr` parameter (default `0.0`). On forward:

```
y_slow = F.linear(x, weight, bias)                     # standard linear
if lr == 0: return y_slow                              # step-0 fast path
diff = x_target - y_slow                               # (..., out)
outer = diff.unsqueeze(-1) * x.unsqueeze(-2)           # (..., out, in)
norm = (x*x).sum(-1, keepdim=True).unsqueeze(-2) + eps # (..., 1, 1)
W_f = weight.unsqueeze(0).unsqueeze(0) + lr*outer/norm # (..., out, in)
y = (W_f @ x.unsqueeze(-1)).squeeze(-1)                # (..., out)
```

`x_target = x` (auto-encoding target — the fast weight is forced to
compress-and-reconstruct the input). At `lr=0` the closed-form
gradient term is exactly 0 so `W_f = weight` and the fast path is
skipped via `if lr.item() == 0: return y_slow` — bit-identical to a
plain `nn.Linear` with the same weight. After the first optimizer step
the learnable `lr` may become nonzero and the fast path engages.

`TTTFeedForward` is a squared_relu FFN whose `up_proj` is a
`TTTLinear(d_model, d_ff)`; `down_proj` stays a plain `nn.Linear` so
the standard FFN-output projection is intact.

**Config flag and zero-init at step 0**

- `use_ttt_ffn: bool = False` (default) ⇒ `TTTLinear`/`TTTFeedForward`
  are never constructed; baseline FFN path is bit-identical (existing
  `if use_soft_moe / elif use_switch_ffn / elif use_expert_choice_moe
  / elif ffn_variant == ...` chain is untouched).
- `use_ttt_ffn: bool = True` with `ttt_lr_init=0.0` (default) ⇒
  the `TTTLinear.ttt_lr` parameter is zero at init ⇒ `lr.item() == 0`
  on the first forward ⇒ `return y_slow` ⇒ output is
  `F.linear(x, weight, None)` with the same `kaiming_uniform_` weight
  as the baseline `nn.Linear` would have ⇒ bit-identical at step 0.

**Run command** (matches the runner pattern in `_arq_145-expert-choice.py`):

```
python _arq_149-ttt-linear.py
```

The script sets `argv = ["train_llm.py", "--config_class", "__main__.C",
"--seed", "42", "--dataset_path", "processed_data/pretrain_1B",
"--warmup", "false"]` and calls `train_llm.main()`. Final val loss is
read from the training log printed by `train_llm.py` (the same path
all other `_arq_*` runners use).

**Bet**: at tiny1m3m (0.94M params, 92 update steps), per-input fast
weights act as a *capacity multiplier* — the static FFN must encode
"what to do with token t" in fixed weights, but TTTLinear gives the
model a per-input weight update at the cost of one extra matmul per
step. PASS ≤ ctrl − 0.01 (sample-efficiency win). NULL band |Δ| < 0.01
(static FFN is not the binding constraint). DRIFT > +0.01 (compute
overhead eats the benefit or fast weights thrash).
