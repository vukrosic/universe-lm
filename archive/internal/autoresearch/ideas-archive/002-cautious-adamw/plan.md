# Plan — 002 cautious-adamw

## Flag
`use_cautious_adamw: str = "none"` added to `LLMConfig` at
`configs/llm_config.py` immediately after `use_cautious_muon: bool = False`
(line 360). Allowed values: `"none"` (default — bit-identical to baseline
AdamW), `"embedding"`, `"gain"`, `"all"`.

**Deviation from `idea.md` wiring (bool) — recorded here so reviewer / user
can revert if undesired.** The wiring section of `idea.md` specifies a
`bool = False` flag, but the **Conditions (A, B; C dropped)** section commits
to a per-bucket A/B experiment (A = mask on `token_embedding` + `emb_proj`,
B = mask on `norm.weight` + 1D scalars). A single bool flag cannot express
both runs. The smallest faithful design is a 4-value string selector with
`"none"` as the default — this preserves the bit-identical-when-off
invariant the implementer prompt requires (baseline AdamW is selected
unchanged when `"none"`). Switching to a bool is a 1-line revert if the
reviewer prefers.

## Change
1. **New file `optimizers/cautious_adamw.py`** — `CautiousAdamW(AdamW)`
   subclass that duplicates the AdamW single-tensor kernel (the per-tensor
   `m`, `v`, bias-correction, decoupled weight decay, and `param.add_`
   sequence) and applies the sign-mask just before `param.add_`. The
   `mask = (update.sign() == grad.sign())` line matches the formula
   written in `idea.md` line 36-38. ~45 LoC including docstring.
2. **`configs/llm_config.py`** — add `use_cautious_adamw: str = "none"`
   directly after `use_cautious_muon: bool = False` (after line 360), with
   a 6-line comment block matching the surrounding style (cf. lines
   352-359). +8 LoC.
3. **`training/trainer.py`** — gate the AdamW swap at the
   `setup_muon_optimizer` call site (lines 142-147): if
   `config.use_cautious_adamw != "none"`, instantiate `CautiousAdamW`
   with the `mask_buckets` derived from the selector; otherwise keep
   `torch.optim.AdamW` unchanged. +5 LoC + 1 import.

**Bit-identity invariant.** When `config.use_cautious_adamw == "none"`
(the default), `setup_muon_optimizer` returns the same
`torch.optim.AdamW(adamw_params, lr=..., weight_decay=..., fused=...)`
instance as today — no behavioral change, no state-shape change. The
existing `001-cautious-muon` Muon path is untouched.

## Control
- **Control:** `V+q+SWA+HighRoPE` recipe (LEADERBOARD row 18d, val 4.6364
  on screen20m). `use_cautious_muon=False`, `use_cautious_adamw="none"`.
  Seed 42 (pipeline rule — single seed). Tier: screen20m.
- **Treatment A (run first):** same recipe + `use_cautious_adamw="embedding"`
  (mask on `token_embedding.weight` and `emb_proj.weight` only).
- **Treatment B (run only if A is in noise, |Δ| ≤ 0.005):** same recipe +
  `use_cautious_adamw="gain"` (mask on `*.norm.weight` and any 1D
  scalar in the AdamW bucket).
- Treatment C (`"all"`) is wired but not run; matches the `idea.md`
  decision to drop C from this sweep.

## Cost
- Params Δ: 0 (no parameter additions, just one optimizer class swap).
- FLOPs Δ: ~0 per step (one extra `.sign() == .sign()` element-wise op
  per AdamW param tensor — negligible vs. the orthogonalization cost in
  the Muon path; AdamW runs only on ~1.6M AdamW params vs. the Muon
  path's larger 2D weight matrices).
- Memory Δ: 0 (no extra tensors; the mask is a same-shape view of an
  existing tensor).

## Run
- **Tier:** screen20m (the only tier where the expected Δ is above the
  noise floor — `LEADERBOARD.md` line 96-99 gives tiny1m3m noise
  ±0.06-0.16, larger than the expected Δ of −0.005 to −0.01).
- **Gate:** only after `001-cautious-muon` passes Phase 1 (tiny1m3m
  val ≤ 6.4206). If 001 fails, close this idea — same mechanism.
- **Launch order:** run A first (~20 min on RTX 3050). If A is null or
  in noise, run B (~20 min). Worst case ≈ 40 min total; happy case
  ~20 min if A hits cleanly.
- **Seeds:** 1 seed (42) per condition. Pipeline rule — sub-noise Δ is
  **inconclusive, not a result**; do not re-run on another seed.
- **Fallback:** if `001`'s screen20m follow-up lands first, the cheaper
  move is to run `use_cautious_muon=True` + `use_cautious_adamw="all"`
  on the V+q+SWA+HighRoPE baseline — one run, additive answer, no fresh
  A/B.

**Pass / fail bar** (copied from `idea.md`):
- pass: screen20m val ≤ 4.6314 (control 4.6364, target Δ = −0.005)
- fail: screen20m val > 4.6364 (worse than control — close the idea)
- noise: |Δ| ≤ 0.005 — inconclusive, run condition B
- expected Δ ≈ −0.005 to −0.01; stretch outcome −0.02
- a null is informative, not a failure

**Run command** (treatment A):
```bash
python scripts/run_research.py --config screen20m_v_q_swa_highrope_cautious_adamw_emb --seed 42
```

(where `screen20m_v_q_swa_highrope_cautious_adamw_emb` is the standard
screen20m control config with `use_cautious_adamw="embedding"` added;
no new top-level preset is required if the user prefers to pass
`--use_cautious_adamw embedding` as a CLI override — depends on
`scripts/run_research.py`'s existing override surface; if overrides
don't support a string field, add the preset.)
