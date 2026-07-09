---
id: 002-cautious-adamw
status: done
round: 3
updated: 2026-06-09T01:02:18Z
---

# 002 — Cautious AdamW

## Source
Liang et al. 2024, "Cautious Optimizers" (arXiv 2411.16085). Extension of [[001-cautious-muon]] to the AdamW path (1D params: gains, scalars, embeddings, head).

## Mechanism
Same sign-mask as Cautious Muon, applied to the AdamW update for 1D parameters. The mechanistic claim from the paper is that the mask helps when the *preconditioned* update direction disagrees with the current gradient sign — that disagreement is the stale-momentum / 2nd-moment-scaling artifact. On Muon this is common (orthogonalized update is sign-agnostic by construction); on AdamW it is rarer in steady state because 2nd-moment normalization already pulls the update toward the sign of the gradient, so the mask is mostly a no-op. The gain on AdamW is therefore expected to be smaller than on Muon (Liang et al. Table 1), but the *failure mode* it targets is different — and complementary, not redundant. A null on this idea does NOT imply Muon-cautious was useless; the two paths are independent and the paper reports both as additive in their small-scale ablations.

## AdamW routing — what's actually in the bucket
The AdamW path is `torch.optim.AdamW` instantiated separately at `training/trainer.py:142`, with a different parameter set than Muon. Per the routing at `trainer.py:79-122` (default flags R1/R2/R3 all off), the AdamW bucket is everything that is NOT a Muon candidate:
- **Embedding params (2D)**: `token_embedding.weight` (vocab × d_model — ~91% of AdamW grads) and `emb_proj.weight` (r × d_model, only when `emb_rank < vocab`)
- **Norm gains (1D)**: every `*.norm.weight` (RMSNorm γ)
- **Output proj (2D)**: `out_proj.weight` (when `muon_for_output=False`)
- **1D scalars**: any 1D learnable scalar (q_gain, k_gain, smear_gate, output_temp τ, vocab_bias b_v, etc.)

The mask is *very* meaningful for embedding rows (a wrong sign on a rarely-updated row is worse than a zero update) and *barely* meaningful for a constant-sign gain. A null on the full bucket is uninformative — the experiment must be split.

## Conditions (A, B; C dropped)
- **condition A — embedding-mask**: mask on `token_embedding.weight` + `emb_proj.weight` only. The high-leverage single bucket with the clearest mechanistic story (rare-row updates).
- **condition B — gain-mask**: mask on `norm.weight` + any 1D scalar (q_gain, k_gain, smear_gate, output_temp τ, vocab_bias b_v) only. Tests whether the cautious mask helps parameters whose sign-of-update is "obvious."
- **C is dropped.** C = A ∪ B is not literally recoverable from A and B alone (the mask interacts), but the per-bucket result is the actionable answer. If both A and B are null, the idea closes. If A hits, the gain-mask question is moot. If B hits, the embedding question is moot.

**Launch order (cost-controlled):** run A first (seed 42, single seed — ~20 min). If A is null or in noise (|Δ| ≤ 0.005, treat as inconclusive), run B. If A is a clear hit, skip B for now — the gain-mask is not the load-bearing story and a follow-up can re-test the combo later.

## Wiring
`use_cautious_adamw` is **not** a config field. `configs/llm_config.py:358-360` mentions it as a future flag; the AdamW path is a separate `torch.optim.AdamW` instantiated at `training/trainer.py:142`. (The `adamw_lr=0.006` referenced in `optimizers/muon.py:75` is a Muon hyperparameter used only by the `rms_match` scale math at `muon.py:147-150` — it is NOT a parameter group; the AdamW path is independent.) **Decision: subclass `torch.optim.AdamW` as `CautiousAdamW`.** Add a top-level `use_cautious_adamw: bool = False` to `LLMConfig` (next to `use_cautious_muon: bool = False` on line 360), create `optimizers/cautious_adamw.py` with a class that copies `Adam._single_tensor_adam`'s body and applies the mask just before `param.add_(update, alpha=-lr)`:

```python
mask = (update.sign() == grad.sign()).to(update.dtype)
update = update * mask
param.add_(update, alpha=-lr)
```

In `training/trainer.py:142`, gate the swap: `CautiousAdamW(adamw_params, ...) if config.use_cautious_adamw else torch.optim.AdamW(adamw_params, ...)`. ~40 LoC total (new class ~30 + 4-line gate in trainer.py + 1 line in config), bit-identical to baseline when `use_cautious_adamw=False`.

## Run notes
- Run only after [[001-cautious-muon]] passes Phase 1 (tiny1m3m val ≤ 6.4206). If 001 fails, close this idea too — same mechanism, different path, gated on first.
- **Tier:** screen20m is the only tier where this is resolvable. tiny1m3m at ~8M training tokens has noise ±0.06-0.16 (`LEADERBOARD.md` line 96-99); the expected Δ is below the noise floor there.
- **Conditions:** 2 (A first; B only if A is null/in noise). C dropped.
- **Seeds:** 1 seed (42) per condition — single seed, pipeline rule. Worst case 2 screen20m runs ≈ 40 min on the RTX 3050; happy case 1 run ≈ 20 min if A hits cleanly. A sub-noise Δ is **inconclusive**, not a confirmed effect — do not add seeds to chase it.
- **Control:** V+q+SWA+HighRoPE 4.6364 (`LEADERBOARD.md` row 18d) — same control as [[001-cautious-muon]]'s screen20m follow-up so the two A/Bs are directly comparable.
- **Expected Δ:** `−0.005 to −0.01` on screen20m (per-parameter, single seed), with `−0.02` as a stretch outcome. A null is informative, not a failure.
- **Fallback:** if 001's screen20m follow-up lands first, the cheaper move is to add `use_cautious_adamw=True` to the same config and run a 2-flag combo (`use_cautious_muon=True` + `use_cautious_adamw=True`) on the V+q+SWA+HighRoPE baseline — one run, additive answer, no fresh A/B. (The previous "001's run exercises the AdamW path with the same config" claim was wrong: 001 only flips the Muon mask; the AdamW path is untouched in 001's run.)

(Pipeline status lives in the frontmatter above.)
