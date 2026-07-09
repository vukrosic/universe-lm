## r1 — 2026-06-09 — verdict: accept

- **Mechanism fidelity** — `models/layers.py:555-557` builds `q_norm`/`k_norm`
  with `_qk_use_ln = self.use_layernorm or use_qk_layernorm` and
  `make_norm(self.d_k, qk_norm_type, _qk_use_ln)`. With the new flag on, both
  become `nn.LayerNorm(d_head)` (verified: `q_norm`/`k_norm` are
  `nn.LayerNorm` w/ `normalized_shape=(d_head,)`). The forward at
  `models/layers.py:1219-1230` calls `self.q_norm` and `self.k_norm` on the
  Q/K reshape `(B,H,T,d_head)` before the attention dot product — faithful
  to the idea and to ViT-22B/Chameleon.
- **Q-side override site** — the second `q_norm` assignment at
  `models/layers.py:777` (the Q-side tweaks branch) now also passes
  `_qk_use_ln`, so the override fires in both paths. The comment
  "REPLACES the line-556 q_norm (this attribute is what forward() uses)"
  is accurate and important — without this line, the Q-side branch would
  silently leave Q on RMSNorm.
- **Default OFF / bit-identical baseline** — `use_qk_layernorm: bool = False`
  at `configs/llm_config.py:458` and at both `__init__` signatures
  (`models/layers.py:439`, `models/layers.py:1665`). With the flag off and
  the global `use_layernorm=False` (default), `_qk_use_ln=False` and
  `make_norm(d_k, "rmsnorm", False)` returns `RMSNorm` (the existing
  baseline). Verified: two-seeded MHA forward diff = 0.0 under
  identical flags.
- **Locality** — the override is OR'd with `self.use_layernorm` so either
  flag flips Q/K to LN, but the residual-stream norms (`norm1`, `norm2`,
  final `norm`) are constructed via `make_norm(d_model, norm_type,
  use_layernorm)` and stay on RMSNorm. Verified: with
  `use_qk_layernorm=True, use_layernorm=False`, `MHA.self.use_layernorm`
  remains False. No collateral flip of the residual stream.
- **Recipe hygiene** — `Tiny1M3MQKNormConfig(Tiny1M3MConfig)` at
  `configs/llm_config.py:621-632` flips exactly one attribute
  (`use_qk_layernorm=True`). No LR / init / schedule / seed drift.
  Single-seed 42 confirmed in `plan.md`.
- **Plumbing** — `MinimalLLM.__init__` reads `use_qk_layernorm` via
  `getattr(config, "use_qk_layernorm", False)` (default-safe) at
  `models/llm.py:232-235` and forwards to `TransformerBlock` at
  `models/llm.py:344`, which forwards to `MultiHeadAttention` at
  `models/layers.py:1770`. Three sites, all guarded.
- **LoC budget** — ~30 LoC added across three files (flag, plumbing,
  recipe, two `make_norm` call-site updates). Well under 200.
- **Coordination** — the diff shares `models/layers.py` /
  `models/llm.py` / `configs/llm_config.py` with 015/017/011 work in
  the same batch, but the 016 surface is cleanly separated: one new
  `bool` field, one `_qk_use_ln` variable, two call-site `use_layernorm`
  → `_qk_use_ln` swaps, one recipe. No stomp on the parallel
  `use_sub_ln` (017) or `use_moonlight_muon`/`use_lion`/`use_cautious_lion`
  (015/011) work.
- **Minor note (non-blocking)** — `plan.md` says "nn.LayerNorm is identity
  at step 0 when weight=1, bias=0" and "verified with a seeded two-model
  forward diff = 0". The first is loose: LN(γ=1, β=0) normalizes
  (mean→0, std→1), it is not identity on the input. The second is true
  for the OFF path only (the lever's job is to differ from baseline when
  ON). The mechanism is correct; the wording is misleading. Not blocking
  — the spec calls for LN with γ=1, β=0 init, and that is what ships.

Verdict: **accept**. Identity-safe when off, mechanism-correct when on,
single boolean, single seed, recipe clean. Ready to run.
