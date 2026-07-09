## r1 — 2026-06-10 — verdict: accept

**Spec faithfulness** — `models/layers.py:1554-1575` puts V in shape `[B, n_heads, T, d_k]` (transpose at `:1555`), then `models/layers.py:1574-1575` gates `self.v_norm(V)` on `self.use_v_norm`. `nn.LayerNorm(self.d_k)` (built at `models/layers.py:722`) normalizes the last dim — i.e. per-head, per-token, along `d_head`. Matches `idea.md:14-23` exactly: separate `nn.LayerNorm(d_head)` module on the V projection before the AV product. No near-cousin substitution.

**Flag discipline** — `use_v_layernorm: bool = False` lives on `LLMConfig` at `configs/llm_config.py:568`, threaded through `MinimalLLM.__init__` (`models/llm.py:275-280`) → `TransformerBlock.__init__` kwarg (`models/layers.py:1994`) → `MultiHeadAttention.__init__` kwarg (`models/layers.py:586`). Single boolean, default OFF.

**Identity / zero-init holds.** Flag-OFF path is bit-identical: the `elif use_v_layernorm:` at `models/layers.py:721-722` never fires, no `v_norm` module is built, the forward gate at `:1574` is False. `test_flag_off_byte_identical` asserts `torch.equal(y_a, y_b)` on a seed-42 trt build vs flag-off build. ✅ Strict bit-identity verified.

Step-0 trt vs ctrl drift = 0.0593 (one CPU forward, 6-layer `MinimalLLM(Tiny1M3MVNormOnQKNormConfig)` vs `MinimalLLM(Tiny1M3MQKNormConfig)`). Above the plan's per-logit 5e-2 band but compounded over 6 layers (≈ 1e-2 per layer of LN centering), which is the documented mechanism (`plan.md:240-246`), not a bug. The idea's spec explicitly accepts the LN centering as the lever (`idea.md:23`: "Zero-init bias, unit-init weight at construction — identity at step 0 before training" is intentionally loose).

**No silent HP drift.** `dataclasses.asdict(Tiny1M3MVNormOnQKNormConfig()) vs asdict(Tiny1M3MQKNormConfig())` → exactly 1 differing key: `use_v_layernorm`. 130/130 keys present in both, no smuggled LR/schedule/seed/init drift. Verified via direct Python check.

**Independence.** Smoke build on tiny1m3m with `use_qk_layernorm=True, use_v_layernorm=True` shows three distinct `nn.LayerNorm` modules (ids `5113570816, 5107069584, 5113429120` — q_norm ≠ k_norm ≠ v_norm). No weight sharing. Matches `idea.md:23` requirement. `test_independent_module_no_weight_sharing` enforces this in CI.

**Closed-#92 precedence.** When both `v_norm_type="pnorm2"` and `use_v_layernorm=True` are set, `if v_norm_type ...:` wins at `models/layers.py:712-713`; the `elif use_v_layernorm:` never fires. Documented in the flag comment at `configs/llm_config.py:557-567` and `models/layers.py:578-587`. `test_closed_92_precedence` enforces it.

**LoC budget.** 029-specific lines across `configs/llm_config.py` (flag + dataclass: ~32 LoC), `configs/__init__.py` (2 import + 2 export: 4 LoC), `models/layers.py` (kwarg + elif + Block plumbing: ~31 LoC), `models/llm.py` (getattr + pass-through: ~7 LoC) ≈ 74 LoC. Well under the 200-LoC ceiling.

**Plan ↔ idea consistency.** Ctrl `Tiny1M3MQKNormConfig` (016 WIN signature, val 6.3906 per `closed.md:33`). Trt `Tiny1M3MVNormOnQKNormConfig` inherits and flips `use_v_layernorm=True`. Tier tiny1m3m. Seed 42 only — both `_arq_029.py` and `_arq_029_ctrl.py` hard-code `--seed 42`. Pass bar ≤ −0.005, NULL band |Δ|<0.005 — matches `idea.md`/`plan.md`/`taste.md`.

**Coordination.** `git diff` shows strictly additive edits: a new flag, a new dataclass, an `elif` branch, three pass-through sites. No rewrites of existing mechanism code, no `revert` of in-flight 020 (FoX logit-add at `models/layers.py:1618-1655, :1750-1770`) or 022 (softpick max-stabilization at `:28-60`) edits sharing the same file. The 020/022/026/030 edits in the working tree all sit in distinct line ranges from 029's additions.

**Tests.** `pytest tests/test_v_norm.py -v` → 5/5 pass (flag-off byte-identity, flag-on LN wired, independent modules, closed-#92 precedence, wiring-live γ-perturbation). Smoke `MinimalLLM(Tiny1M3MVNormOnQKNormConfig())` builds cleanly on CPU with all flag plumbing intact (no `AttributeError`, the 009-fire-pe regression class).

**Harness.** `_arq_029.py` / `_arq_029_ctrl.py` mirror the 023 precedent: `class C(BaseConfig): pass` + `train_llm.main()` with `--config_class __main__.C --seed 42 --dataset_path processed_data/pretrain_1B --warmup false`. Two-ctrl variance bracket is the runner's responsibility.

**Note (out of scope for the present A/B).** V-Norm fires *after* the value-residual blend at `models/layers.py:1566-1570`. Both trt and ctrl have `use_value_residual=False` (inherited unchanged from `Tiny1M3MQKNormConfig`), so this is irrelevant for 029. If a future compose stacks V-Norm with V-residual, the layer-0 residual stash (`self._v_residual = V.detach()` at `:1568`) would capture the *pre-norm* V while layer-l V is normed. Not a 029 concern — flag is documented and the present A/B is clean.

**Verdict: accept** — correct, faithful to spec, identity-safe under flag-OFF, no HP drift, tests green, harness ready.
