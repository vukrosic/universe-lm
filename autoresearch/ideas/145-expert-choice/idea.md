---
id: 145-expert-choice
status: running
round: 1
updated: 2026-06-14T04:47:07Z
transfer-risk: med
plain: A different MoE routing scheme where each expert picks its top tokens (rather than each token picking its top expert), so the load is perfectly balanced by construction.
---

# 145 — Expert Choice MoE Routing

## Source
Zhou, Lei, et al. 2022, "Mixture-of-Experts with Expert Choice Routing", Google, arXiv:2202.09368. https://arxiv.org/abs/2202.09368

## Mechanism
In standard (token-choice) MoE, each token picks its top-k experts. Load balance is a *loss term* (added to training objective) that softly enforces uniform expert usage. In expert-choice MoE, each expert picks its top-k tokens. Load balance is *by construction*: every expert processes exactly k tokens.
- `router_logits: [n_experts, n_tokens]`  (one score per expert-token pair)
- For each expert `e`: `topk_tokens_e = topk(router_logits[e, :], k)` where `k = n_tokens / n_experts`
- Expert `e` processes `topk_tokens_e`
- Output: weighted sum of expert outputs, with weight = `softmax(router_logits[e, topk_tokens_e])` per expert.

Identity at step 0: router weights init to 0 → all expert-token scores are 0 → topk picks tokens arbitrarily (by index in our impl). All experts process the same set of k tokens. Output is the *average* of all expert FFNs applied to the same tokens — equivalent to a dense FFN with mean pooling across N experts. If experts are init the same, output ≈ single-FFN output → identity to baseline (one expert is essentially "as good as" the average of N identical ones).

## Design sketch (how it works + how to build it)
- Add an `ExpertChoiceMoE` module to `models/layers.py`: holds `n_experts` parallel FFNs plus a `nn.Linear(d_model, n_experts, bias=False)` router. Forward: `scores = router(x) → [B, T, n_experts]`, transpose to `[n_experts, B, T]`, topk per expert, gather tokens, run each expert on its tokens, scatter back. ~120 LoC.
- Add `use_expert_choice_moe: bool = False`, `n_moe_experts: int = 4` to `configs/llm_config.py`. `n_moe_experts * d_ff` extra params; in the typical `n_experts=4` config this is a 4× FFN param cost.
- Identity at step 0: as above. All experts are init as standard FFN (same seed) and process the same tokens → output is identical to a single FFN at step 0. (For step-0 strict identity, init all experts to the same weights via seeding.)
- Why a real lever, not a hyperparam: the *routing direction* (token→expert vs expert→token) is a structural choice, not a knob. Token-choice MoE needs an auxiliary load-balancing loss; expert-choice doesn't. The two are not reachable by tuning the same loss coefficient.
- Targets baseline failure: 117-soft-moe is null at 0.94M with `+0.139` wrong-sign. The closed reason is "soft-routing overhead at d_model=64". Expert-choice uses *hard* routing (topk is discrete) which avoids the slot-assignment overhead. Different cost profile.

## Scale evidence
Paper trains 50B+ MoE LMs (Google's GLaM, 137B-active / 1.2T-total). 0.94M is well below the validated range. Transfer risk: med — MoE at 0.94M has been shown to be a null axis twice now (117-soft-moe, 118-MoD), and the leverage is unlikely to change with routing direction. But the inductive bias (load-balanced by construction) is different and might be cheaper to train at small scale.

## Why it's worth a slot
If we don't file this, the "MoE" axis closure is based on soft-routing (117) and skip-routing (118) — two specific mechanisms. Expert-choice adds a third data point with hard-routing, and would close the entire MoE axis conclusively. If it wins, the load-balanced-by-construction is the missing ingredient; if it nulls, MoE is closed at 0.94M and we stop filing MoE ideas.

## Plan

### Files
- **NEW** `models/expert_choice_moe.py` — `ExpertChoiceMoE` class (~120 LoC).
- **EDIT** `models/layers.py` — import + add `use_expert_choice_moe: bool = False`,
  `n_moe_experts: int = 4` kwargs to `TransformerBlock.__init__`; wire it
  into the FFN-selection branch right after the `use_switch_ffn` branch.
- **EDIT** `configs/llm_config.py` — add the two flags with the same
  defaults.
- **EDIT** `models/llm.py` — read the two flags off config in
  `MinimalLLM.__init__` and pass them through to both TransformerBlock
  construction sites (lines 504-509 and 636-642).
- **EDIT** `train_llm.py` — add `--use_expert_choice_moe` /
  `--n_moe_experts` CLI flags (mirroring the `--use_soft_moe` block at
  ~line 314 / 523).

### Config flag
- `use_expert_choice_moe: bool = False` (off by default → baseline path
  bit-identical).
- `n_moe_experts: int = 4` (default 4, matching 117-soft-moe /
  146-switch-ffn).

### Identity at step 0
- Router `nn.Linear(d_model, n_experts, bias=False)` zero-init.
- All expert FFNs init via the same per-`ffn_variant` factory as the
  baseline (squared_relu etc.).
- For STRICT byte-identity at step 0 with the flag off: we never build
  the `ExpertChoiceMoE` module — `use_expert_choice_moe=False` keeps
  the standard `SquaredReLUFeedForward` path. With the flag ON, the
  experts see different RNG draws so the layer is NOT byte-identical
  to a single-FFN baseline at step 0 (a benign ~0.005 val-loss drift
  in the first 1-2 steps is the documented caveat, mirroring
  117-soft-moe).

### Run command
```bash
python train_llm.py --config tiny1m3m --seed 42 \
  --use_expert_choice_moe true --n_moe_experts 4
```
Final val loss is read from the JSONL/CSV line emitted at the end of
`train_llm.py` (same as every other experiment in this repo).

### Cost when on
`n_moe_experts * d_ff` extra params (4× the FFN param cost at default
4 experts — same budget impact as 117-soft-moe and 146-switch-ffn).

### Re-code round 1 → 2 (2026-06-13)
The original pass shipped the model code (`models/expert_choice_moe.py`),
the layer wiring (`models/layers.py`, `models/llm.py`), and the CLI flag
(`train_llm.py`), but **never added `Tiny1M3MExpertChoiceConfig` to
`configs/llm_config.py` and never wrote `_arq_145-expert-choice.py`**.
The runner's preflight caught both. Round 2 adds:
- `Tiny1M3MExpertChoiceConfig(Tiny1M3MConfig)` in `configs/llm_config.py`
  with `use_expert_choice_moe=True`, `n_moe_experts=4`.
- `_arq_145-expert-choice.py` at repo root, same shape as
  `_arq_146-sparse-ffn.py` / `_arq_140-sophia.py`: imports the new
  config, defines `class C(...)`, runs `train_llm.main()` with
  `--config_class __main__.C --seed 42 --dataset_path
  processed_data/pretrain_1B --warmup false`.

Verified locally: `Tiny1M3MExpertChoiceConfig` imports, `MinimalLLM`
built from it has `ExpertChoiceMoE` on all 12 transformer blocks
(0/12 for the plain `Tiny1M3MConfig`), and the param delta is
+1,182,720 (~4× the FFN params, matching the design sketch). The
launcher imports cleanly and exposes `C` as the config-class entry.
