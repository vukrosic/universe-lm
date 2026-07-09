# Plan — 010 PolyLoss

## Flag
- `use_poly_loss: bool = False` — read via `getattr(config, "use_poly_loss", False)` in `training/trainer.py` (consistent with the existing `use_z_loss` / `label_smooth` / `conf_penalty_beta` getattr pattern). Default OFF → baseline path byte-identical.
- `poly_eps1: float = 1.0` — read via `getattr(config, "poly_eps1", 1.0)`. Pinned at 1.0 (the principled next-Taylor-term value per Leng et al. 2022). No sweep.
- Config class: `Tiny1M3MPolyLossConfig(Tiny1M3MConfig)` in `configs/output_head_ablations.py` (with `use_poly_loss=True`, `poly_eps1=1.0`) — for discoverability / wiring parity with the existing loss-side aux family (ZLoss, LabelSmooth, ConfPenalty).

## Change

| File | Edit |
|---|---|
| `training/trainer.py` (AMP path, ~line 415) | Add `poly_loss = ...` (3-5 LoC) behind `if use_poly_loss:` guard. Build `(B·T)` mask, gather softmax probs at gold index (clamped to 0 for safety), compute `ε₁ · (1 - p_t) * mask`, normalize by valid-token count. Default OFF → `poly_loss = 0`. |
| `training/trainer.py` (non-AMP path, ~line 446) | Same guard, identical math. |
| `training/trainer.py` (loss sum) | Add `poly_loss` to the per-path loss sum (AMP: line 433, non-AMP: line 464). |
| `training/trainer.py` (pbar logging) | Optional: surface `pl:` tag in `set_postfix` (cosmetic; does not affect identity). |
| `configs/output_head_ablations.py` | Add `Tiny1M3MPolyLossConfig` (2 LoC) — wires the flag for the runner's `--config` arg path. |
| `training/evaluation.py:53` | **0 LoC** (plain CE stays). Train-only reporting rule per idea.md. |

Step-0 (flag OFF) — `use_poly_loss=False` (or absent). The guard short-circuits and `poly_loss = logits.new_zeros(())`. Sum is bit-identical to the existing baseline.

## Control
- **Control**: V+q+SWA+HighRoPE (current screen20m best recipe) + plain CE. Seed 42. Tier `tiny1m3m` (0.94M params, 3M tokens).
- **Treatment**: control + `use_poly_loss=True`, `poly_eps1=1.0` → train loss becomes `L_CE + 1.0 · (1 - p_t)` masked to non-`-100` positions. Eval stays plain CE. Seed 42. Tier `tiny1m3m`.

## Cost
- **Params**: 0 (no model change).
- **FLOPs/step**: one extra `softmax` over logits + one `gather` per training step on the train path. For `B·T = 2·2048 = 4096` tokens × `vocab_size=49152`, ~200M flops → ~negligible relative to the forward/backward.
- **Memory**: same as baseline (we already have `logits` in scope; the gather is O(B·T)).
- **Eval cost**: 0 (PolyLoss is train-only; `training/evaluation.py:53` is unchanged).

## Run

### Step 0 — smoke (gate)
CPU smoke test of the `use_poly_loss` integration on `Tiny1M3MConfig`:
- `use_poly_loss=False` (default) → trainer still produces the same `loss = ce_loss + ... + conf_penalty` sum; `poly_loss` is the zero scalar.
- `use_poly_loss=True`, `poly_eps1=1.0` → `poly_loss` is positive (≤ 1.0 per token since `p_t ∈ [0,1]`, so `(1 - p_t) ∈ [0,1]`); final loss > baseline CE by `poly_eps1 · mean(1 - p_t)` over non-`-100` positions.
- `-100` mask: at every position with `shift_labels == -100`, the contribution to `poly_loss` is exactly zero. Verify by constructing a synthetic batch with all `-100` labels and confirming `poly_loss == 0`.

### Step 1 — full A/B on `tiny1m3m`
```bash
# Control
python train_llm.py --config tiny1m3m --seed 42 --out runs/tiny1m3m-poly-ctrl

# Treatment (use the new config class)
python train_llm.py --config tiny1m3m_poly_loss --seed 42 --out runs/tiny1m3m-poly-trt
```
Wall-clock: ~10-15 min each on a single A100 / T4. Pass/fail bar from `idea.md`:
- **PASS**: treatment val_loss ≤ control val_loss − 0.005.
- **NULL**: |Δ| < 0.005. Log `evidence.md` with verdict NULL; append one line to `closed.md`. (Informative — "CE's `j=1` Taylor truncation term is negligible at tiny1m3m.")
- **DRIFT**: control val_loss > 6.4287 + 0.01 → box validation; rerun or kill the slot.
- 0.005 ≤ |Δ| ≤ 0.01 → inconclusive; log only.

### Step 2 — verdict (single seed, per pipeline hard rule)
Seed 42, single seed. |Δ| ≤ 0.005 is the noise band; a sub-noise result is logged inconclusive, not re-seeded.

## Self-check (before release to code-reviewer)
- `use_poly_loss=False` (default) reproduces the control (no numeric drift) — by construction: the guard short-circuits to `poly_loss = 0`, the loss sum is unchanged.
- The treatment path actually exercises PolyLoss — confirm by smoke: `poly_loss` is positive and finite when the flag is on; `poly_loss` is exactly zero when the flag is off.
- `plan.md` pass/fail bar matches `idea.md` — both: PASS ≤ ctrl − 0.005, NULL = |Δ| < 0.005, DRIFT > ctrl + 0.01.
