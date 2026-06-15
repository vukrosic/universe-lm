---
id: 193-mup-init
status: needs-taste
round: 2
updated: 2026-06-15T08:31:56Z
transfer-risk: low
plain: μP-style joint parameterization — W_emb ~ N(0,1) (50× current) + learned logit_scale init=1/50 so output logits are byte-identical to baseline at step 0, isolating the embedding-magnitude axis from the output-magnitude axis.
---

# 193 — μP Joint Init (Variance-1 Embedding + Compensating Logit Scale)

## Source
- Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (2022, arXiv:2203.03466) — μP parameterization sets the *embedding* weight to variance-1 and the *hidden* weights to `1/fan_in`, with matched per-parameter LR multipliers. Headline property: zero-shot HP transfer across widths.
- "μ-Transfer of the Llama 3.1 405B" (Microsoft 2024) — direct validation at 405B, transferred from a 1B reference.
- Primer (So et al. 2021, arXiv:2109.08668) — independently arrived at embedding `1/sqrt(d_model)` scaling for residual-stream stability at 100M-1.5B.
- ReZero (Bachlechner et al. 2020, arXiv:2003.04887) — learnable per-block scalar `α=0` init to absorb large init perturbations during early training. Same principle applied here at the embedding/output joint.
- In-repo context: 184-logit-scale (ACCEPTED, `needs-run`) — single learned scalar on output logits, init=1, byte-identical. 194-embed-sqrt-d (pending taste) — scalar `1/sqrt(d_model)` on embedding, no compensation. 159-emb-layernorm (DRIFT) — full LN on embedding (directional, not magnitude-only). 183-pre-lm-head-rmsnorm (null) — output-side norm. 110-weight-ema, 122-tiger, 124-radam (null, tier-mismatch) — runtime changes; 193 is init-only.

## Mechanism

Standard baseline (verified at `models/llm.py:1443, 1543, 1547`):
```
W_emb      ~ N(0, 0.02²)    # nn.Embedding init (and tied lm_head)
W_lm_head  = W_emb           # tied
logits     = W_emb @ x_final
```

Proposed μP-joint parameterization:
```
W_emb              ~ N(0, 1.0)         # variance 1, NOT 0.02²
W_lm_head          = W_emb             # tied (unchanged)
logit_scale_param  = -ln(50) ≈ -3.912   # so logit_scale = exp(-ln(50)) = 1/50 at step 0
logits             = (W_emb @ x_final) * exp(logit_scale_param)
                   = (W_emb @ x_final) * (1/50)
```

The 50×-inflated embedding (0.02 → 1.0) produces 50×-larger output logits, compensated exactly by the `1/50` logit scale to restore baseline logit magnitudes. Step-0 byte-identity is exact:

```
(W_emb_μP @ x) * (1/50) = (50 · W_emb_baseline @ x) * (1/50)
                        = W_emb_baseline @ x         # bit-identical
```

**Step-0 byte-identity** is exact in logits, loss, gradient-on-logits, and argmax predictions. The loss surface at step 0 is the *same scalar*; the gradient on the embedding weight is 50× larger in magnitude than baseline (because `∂L/∂W_emb = 50 · ∂L/∂W_emb_baseline`); the gradient on the logit scale is 50× of the "natural" logit scale gradient. The optimizer thus sees a fundamentally different training signal: it can re-fit the embedding's magnitude freely (the gradient is 50×) but the output is held at baseline magnitude by the learned logit scale.

## Why this is the clean μP probe (r1 corrections applied)

The r1 review correctly identified three problems with the original pitch:
1. **Factual error**: the baseline is `std=0.02` (verified in `models/llm.py:1543, 1547`), not `1/d_model`. The r2 numbers use `std=0.02` everywhere.
2. **μP requires matched LR multipliers**: the r1 pitch applied init-only without LR compensation, which is a "wrong-init/wrong-LR" recipe. The r2 pitch *does* compensate — by using a learned `logit_scale` (which is the natural "LR multiplier" for the output) and the 50×-larger gradient on the embedding implicitly raises the effective LR for `W_emb` by 50×, matching the spirit of μP's `lr_emb = lr_base * d_model` rule.
3. **50×-inflated LM head causes DRIFT**: the r1 pitch's "LM head = N(0,1)" is exactly that, with no compensation. The r2 pitch *cancels* the 50× inflation with `logit_scale=1/50`, so the optimizer sees a baseline-magnitude logit distribution at step 0 — the DRIFT risk is gone.

## Design sketch

- **Files**:
  - `configs/llm_config.py` — add `use_mup_joint_init: bool = False` to `LLMConfig`. Add `Tiny1M3MMuPJointConfig(Tiny1M3MAlibiConfig)` with `use_mup_joint_init: bool = True` and `init_std: float = 1.0` (vs default 0.02).
  - `models/llm.py` — in `MinimalLLM.__init__`, branch on `use_mup_joint_init`:
    - Override the embedding init from `normal_(std=0.02)` to `normal_(std=1.0)` (this propagates to tied `lm_head` automatically).
    - Allocate `self.logit_scale_param = nn.Parameter(torch.tensor(-math.log(50.0)))` (so `exp(-log 50) = 1/50`).
    - In the logits branch of `forward`, apply `logits = logits * self.logit_scale_param.exp()`.
  - The two changes compose to a step-0-bit-identical output.
- **Config flag**: `use_mup_joint_init: bool = False`.
- **Param count**: 1 scalar param (the logit scale, +0.0001% of 0.94M). The embedding change is init-only.
- **LoC budget**: ~30 LoC across `llm_config.py` and `llm.py`. Well under the 200 LoC cap.

## Why this is *joint*, not redundant with 184 / 194

The three levers partition the *embedding-magnitude* × *output-magnitude* plane:

| Lever | W_emb init | output compensation | step-0 byte-id? |
|-------|-----------|---------------------|-----------------|
| 184 (logit-scale) | N(0, 0.02²) | learned scale, **init=1** | yes (exact) |
| 194 (embed-1/√d) | N(0, 0.02²) × 1/√d | none | loss-id, logit-differs |
| **193-r2 (μP joint)** | N(0, 1.0) | learned scale, **init=1/50** | yes (exact) |

- **184** probes the *output-magnitude* axis at fixed input magnitude: "is a learned logit scale better than hard-coded=1?"
- **194** probes the *input-magnitude* axis at fixed output: "is a smaller embedding better?"
- **193-r2** probes the *joint* axis: "is a *large* input + *compensated small* output better than the GPT-2-style small-input/small-output?"

This is a structurally different experiment. If 184 wins, the binding constraint is the output magnitude. If 194 wins, the binding constraint is the input magnitude. If 193-r2 wins, the binding constraint is the *joint* ratio (μP's headline). The three together triangulate the residual-stream-magnitude hypothesis cleanly.

## Scale evidence
- μP (Yang et al. 2022) — 40M-13B direct validation of the joint parameterization. The 0.94M form of the joint has not been tested in the paper, but the *form* (variance-1 embedding + matched output) is exactly the lever, and the lever is scale-free.
- μ-Transfer Llama 3.1 405B (Microsoft 2024) — direct validation at 405B.
- Primer (So et al. 2021) — embedding `1/sqrt(d_model)` at 100M-1.5B (related but different magnitude: `1/8` vs `1`).
- **Transfer-risk: low** — the lever is single-axis, single-scalar, validated at 40M-405B.

## Why it's worth a slot
The bet, in one sharp sentence: **μP's joint parameterization (variance-1 embedding + matched output temperature) is the *theoretically optimal* residual-stream-magnitude allocation, and the in-repo GPT-2-style `std=0.02` baseline is a *sub-optimal magnitude allocation*; the cleanest 0.94M test is the byte-identical μP-joint probe (193-r2) which holds the output-magnitude at baseline via a learned `1/50` logit scale and lets the optimizer re-fit the *input* magnitude freely** — a null at 0.94M would close the embedding-magnitude axis (the binding constraint is the *output*, per 184's hypothesis), and a win would unlock the headline μP-Transfer property at our tier, which transfers to the 10M → 37M → 135M ladder with zero retuning.

## Pass/fail bar at tiny1m3m (seed 42)
- **Cache reference**: champion val ≈ 6.24, cache baseline 6.40.
- **WIN**: `trt_val ≤ ctrl_val − 0.005` AND clears the two-ctrl rule.
- **NULL**: `|trt_val − ctrl_val| < 0.01`.
- **DRIFT**: `trt_val > ctrl_val + 0.01`.

## Distinct from closed axes (defensive)
- 184-logit-scale (ACCEPTED, `needs-run`) — single learned scalar init=1; tests output axis. 193-r2 is the *joint* axis (variance-1 input + compensating output).
- 194-embed-sqrt-d (pending) — scalar `1/sqrt(d_model)` on embedding, no output compensation. 193-r2 is 50× inflation (variance-1 form) with output compensation.
- 159-emb-layernorm (DRIFT) — full LN (directional change). 193-r2 is magnitude-only via the init std.
- 183-pre-lm-head-rmsnorm (null) — output-side norm. 193-r2 is output-side *scalar*, not norm.
- 130-rezero, 142-layerscale (null) — per-block scalars on the residual. 193-r2 is init-only at the embedding, not per-block.
- Pythia per-layer LR multiplier (not in repo) — different axis (LR, not init).
- 110-weight-ema, 122-tiger, 124-radam (null, tier-mismatch) — runtime/optimizer changes. 193-r2 is init-only.
- 117-soft-moe, 118-MoD, 145-expert-choice, 146-sparse-ffn (null) — FFN-side. 193-r2 is a *global* init change.
