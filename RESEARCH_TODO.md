# Research / infra TODO

Captured 2026-05-30. Ordered by priority. Status: ⬜ todo · 🟡 drafted-not-tested · ✅ done

## Bugs / validity (do first)
- ⬜ **Seed init is hardcoded.** `training/trainer.py:443` calls `set_seed(42)` for
  model init regardless of `config.seed`. The sweep's `seed` only varies dataloader
  order (sweep.py:167), so our "5-seed" runs shared ONE weight init. Init is usually
  the bigger variance source → our seed spread (+0.005→−0.040) understates true noise.
  Fix: init model from `config.seed`. Re-interpretation: embed_residual evidence is
  weaker than reported (data-order noise only).

## Speed / iteration (drafted locally, NOT synced to box yet)
- 🟡 **`fastscreen` preset** (configs/llm_config.py): same 25M shape (keeps param
  regime → still transfers), but 10M tokens, compile on, batch 4×accum 2, eval_steps 50.
  Goal: ~6 min → ~2 min/screen. Registered in sweep.py + needs train_llm.py preset_map.
  NOT timing-tested yet.
- 🟡 **TF32 enable** (training/trainer.py ~line 105): `allow_tf32` + `matmul_precision("high")`
  on cuda. Free ~1.3× on fp32 paths. Untested on the RTX 5060.
- ⬜ Do NOT shrink params below 25M — 49k vocab makes small models embedding-dominated
  (94% at 5M, 73% at 25M, 19% at 135M target). That's why 5M screens flipped sign.
  Faithful cheap proxy is DEEPER, not smaller.

## Methodology
- ✅ **Curve logging** (sweep.py resolve_config): every run now logs val-loss milestones,
  so one baseline-to-100M covers all smaller budgets. DEPLOYED to box.
- ⬜ **Determinism lockdown** — prereq for "train baseline once, reuse." Currently
  cross-sweep drift ~0.007 vs within-sweep 0.0005, so stored baselines aren't clean.
  Need: `torch.use_deterministic_algorithms(True)`, seeded dataloader workers, cudnn
  determinism. Then 1 baseline/seed covers everything.
- ⬜ **Token-fraction milestones** — `build_eval_milestones` emits fixed STEP numbers
  keyed on tokens; breaks if batch/grad_accum/seq change. Switch to fractions (0,2,5,
  10,25,50,75,100%) converted to steps at runtime. Robust + auto-aligns curves.
- ⬜ **Effect-size filter protocol** — stop multi-seed screening of small effects.
  1-seed screen → only chase >0.03 → confirm survivors with 2–3 seeds.

## Results backlog
- ⬜ **embed_residual**: bank as "real but sub-threshold" (25M mean −0.015, flat 25M→100M
  at −0.004, only 4/5 and that's data-order-only). Not worth more seeds. Revisit only if
  a curve shows a widening gap.
- ⬜ Next mechanism candidates (big-effect, few seeds): muon_lr sweep, depth↔width at
  fixed params, logit soft-cap / z-loss.
