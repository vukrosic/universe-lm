# Review log — 176 v-pre-av-norm

## r1 — 2026-06-15 — verdict: revise
- **Mechanism is locked and structurally sound.** Single parameterization
  (no Option A/B ambiguity like 169's r1): per-head scalar gate
  `α_h = relu(α_raw_h)` init 0, per-head gain `γ_h ∈ ℝ^{d_k}` init 1,
  output `V_out = (1 − α_h) · V_in + α_h · RMSNorm(V_in) · γ_h`. With
  `α_h = 0` for all heads at init, the lever is the *identity* function
  (no V rescaling) ⇒ step-0 is bit-identical to baseline. The
  V-out is the actual `V` consumed by `scores @ V` (line ~2422 per the
  existing 162/165 forward-branch pattern). The `relu(0)=0` gate is the
  simplest bit-identity formulation; the alternatives (sigmoid -10
  init, straight-through) are noted but not needed. ✓
- **Source-citation gap — actionable but non-blocking.** `## Source`
  frames 176 as "the natural V-side analog of 016-qk-norm" without
  citing a primary source for V pre-AV normalization. The actual
  primary cite already lives in this repo: **Wortsman et al. 2023,
  arXiv:2309.14322** (Stability AI report — see
  `models/layers.py:1017` for the existing 029 reference). The
  abstract "extension of the 016 family" framing is fine for the
  taste claim, but `## Plan` (which is missing — see next finding)
  should tighten the citation set when the implementer writes it.
  One sentence suffices: "176 is the per-head α-gated RMSNorm+γ-gain
  parameterization of Wortsman 2023's V-norm primitive."
- **CRITICAL — `use_v_layernorm` flag already exists in working tree.**
  `configs/llm_config.py:1777` defines `use_v_layernorm: bool = False`
  (the 029-V-Norm override, LayerNorm flavor, mirrors `use_qk_layernorm`).
  `models/layers.py:1026, 1324-1326, 3717-3720, 4175-4177` wire it
  through `MultiHeadAttention.__init__`, the v_norm-construction site,
  `TransformerBlock.__init__`, and the LLM pass-through. The 176 lever
  is *not* a mathematical duplicate — RMSNorm+α-gate+per-head-γ is
  structurally distinct from closed-#92 `v_norm_type` / closed-029
  `use_v_layernorm` (LayerNorm, no gate, no per-head gain) — but the
  working tree already has a V-pre-AV lever for this exact tensor
  location. The plan MUST (a) explicitly diff against `use_v_layernorm`
  in `## Cross-checks` and confirm distinctness, (b) add an
  `assert not (self.use_v_rmsnorm and self.use_v_layernorm)` at the
  top of `MultiHeadAttention.forward` (mirror the
  `use_cope ∧ use_qk_norm_post_rope` assertion at line 1948), and
  (c) add a `use_v_norm_type`-related assertion for the closed-#92
  path (`assert not (self.use_v_rmsnorm and self.v_norm_type not in
  ("", "none", None))` — the closed-#92 lever is "explicit > implicit"
  per the code comment at line 1322). The 169-r1 finding
  precedent (mutual-exclusion with the 162/165/016 family) is the
  template.
- **Plan section missing — must add before code gate.** `idea.md` has
  no `## Plan` section. The 169-r1 and 165-r1 reviewers flagged the
  same thing (162/165 got their `## Plan` in r1 only after a `revise`
  round). Add a `## Plan` that locks:
  - **Field name**: `use_v_rmsnorm: bool = False` on `LLMConfig`
    (sibling of `use_v_layernorm` at `configs/llm_config.py:1777`).
  - **Config subclass**: `Tiny1M3MVPreAVNormConfig` next to
    `Tiny1M3M*LayerNorm` configs near `configs/llm_config.py:2581`
    (the `@dataclass`-decorated subclass, NOT a bare `class C(...):`
    annotation per the 162/165/155/161 precedent that bare-class
    annotation breaks dataclass field inheritance).
  - **MHA kwarg plumbing**: add `use_v_rmsnorm: bool = False` kwarg
    to `MultiHeadAttention.__init__` at line 1026 (sibling of the
    existing `use_v_layernorm` parameter at the same line); the
    `use_v_layernorm` site at line 1324 is the construction
    template — slot a parallel `elif use_v_rmsnorm:` arm that
    registers `self.v_rmsnorm_alpha = nn.Parameter(torch.zeros(H))`
    and `self.v_rmsnorm_gain = nn.Parameter(torch.ones(H, d_k))`.
    Apply the gate-α+γ after `v = self.W_V(x)` and before
    `scores @ v` (i.e. before any RoPE, since V is *not* rotary in
    the current MHA — verify vs. the existing v-residual site at
    line ~1390 if 021-value-residual is also on, and the
    `self.alpha_av` site at line 1418 if AV-output carry is on).
  - **Mutual exclusion asserts** (top of MHA.forward, mirror
    `use_cope ∧ use_qk_norm_post_rope` at line 1948):
    `assert not (self.use_v_rmsnorm and self.use_v_layernorm)`,
    `assert not (self.use_v_rmsnorm and self.use_v_norm)` (the
    closed-#92 path), and `assert not (self.use_v_rmsnorm and
    self.use_v_mix_conv)` if v_mix_conv is also on (clean
    independence assertion, free insurance).
  - **TransformerBlock pass-through**: add `use_v_rmsnorm: bool =
    False` to `TransformerBlock.__init__` at line 3720 (sibling of
    `use_v_layernorm`), read from kwargs, pass into the MHA
    constructor at line 4175-4177.
  - **llm.py read+thread**: add `self.use_v_rmsnorm =
    getattr(config, "use_v_rmsnorm", False)` at line ~440 (sibling
    of `self.use_v_layernorm`), thread into both
    `TransformerBlock(...)` constructor sites at lines ~685 and
    ~941.
  - **Param count**: H=4 × (1 α + 16 γ) = 68 params per block × 12
    blocks = 816 params (+0.087% of 0.94M), well under the per-lever
    budget. (Not 204 as the miner's taste-summary says — re-derive
    from H=4, d_k=16, n_layers=12, not H=12.)
  - **Runner stub**: `_arq_176-v-pre-av-norm.py` mirroring the
    162/165/169/170 pattern (`build Tiny1M3MVPreAVNormConfig`,
    `config_class`, `/venv/main/bin/python _arq_176-v-pre-av-norm.py`).
- **Pass/fail bar not numerically specified.** `## Why it's worth a
  slot` says "expected Δval ∈ [-0.005, -0.020]" but no explicit
  PASS/NULL/DRIFT thresholds. Lock: **control = unmodded
  `Tiny1M3MConfig`** (no V-norm baseline), **WIN = Δval ≤ -0.005 vs
  cached baseline ≈ 6.4394** (mirrors 016's bar; clears the ±0.04
  box noise at tiny1m3m by ≥2×), **NULL = |Δ| < 0.005**, **DRIFT =
  Δval ≥ +0.005**. Add the numbers in `## Pass / fail bar` (a new
  section — 162/165/169 all have this).
- **Identity-init tolerance is clean, not just "1e-3".** Because the
  gate-α=0 formulation gives `V_out = V_in` *exactly* (no rescaling
  at step 0), the lever is **byte-identical** to baseline at step 0
  (max-abs-diff = 0.0). The fp32-tolerance wording in `## Mechanism`
  ("below the fp32 noise floor") should be sharpened to "byte-
  identical (max-abs-diff = 0.0)" — the lever doesn't depend on the
  RMSNorm-rescaling-tolerance precedent at all because the gate
  short-circuits it. Verify in the runner with a fp32 max-abs-diff
  test (assert `trt_step0_logits == ctrl_step0_logits` byte-exact).
- **Transfer-risk: med justified.** The 016 family is the closest
  scale-validated analog (RMSNorm at 1B+ via LLaMA-3 / Qwen-2.5 /
  Mistral). The per-head α-gate + per-head γ-gain is novel and
  unvalidated at scale. V-pre-AV normalization is well-validated
  (Wortsman 2023 at 0.94M-class + Chameleon-class QKNorm applied
  to all three of Q/K/V). `med` is the correct hedge: primitive
  is scale-tested, parameterization is novel. Tag holds. ✓
- **Cross-check vs closed list — distinct.** Verified:
  - 016-qk-norm (WIN) — symmetric QK RMSNorm pre-softmax; 176 is
    V-pre-AV. Different tensor, different position.
  - 029-v-norm (closed) — V-side LayerNorm, no gate, no per-head
    gain; 176 is V-side RMSNorm+per-head α-gate+per-head γ-gain.
    Different parameterization, distinct lever.
  - closed-#92 `v_norm_type` (closed zoo, in working tree as
    `models/layers.py:1312-1317`) — V-norm zoo, not gated; 176
    adds a learnable per-head gate the closed axis lacks.
  - 160-rms-gain-per-head (NULL) — post-AV per-head gain; 176 is
    pre-AV. Different tensor location (post-aggregation vs
    pre-aggregation).
  - 154-rebased-attn (WIN) — rebases K and V before softmax; 176
    only normalizes V (no rebasing), different mechanism.
  - 152-attn-logit-bias (NULL), 155-per-head-temp (NULL),
    161-dyt-temp (NULL), 162-q-only-norm (NULL), 165-k-only-norm
    (NULL) — none of these touch the V tensor at this location.
  - 173-entmax-15 (in queue) — entropy-based softmax replacement;
    orthogonal to V-normalization.
  - Not in `_closed/`. Not in the closed-axes section of closed.md.
- **Coordination note (non-blocking).** The working tree's
  `configs/llm_config.py` and `models/layers.py` have 162/165/166/
  167/168/169/170/171/172 hunks landed (per `git log --oneline`).
  The 176 hunks are not yet present — code-impl gate's job to add
  them per the `## Plan` above. No conflict. Running `git diff` to
  verify before editing is the reviser's job, not the reviewer's
  (the review gate is plan-not-code).

**Verdict**: revise → `needs-revision`. Mechanism is sound and the
parameterization is genuinely novel (not a duplicate of the
closed-#92 / 029 V-norm axis), but the plan is missing, the pass
bar needs numerical thresholds, and the existing `use_v_layernorm`
flag in the working tree requires explicit mutual-exclusion
asserts. All findings are reviser-actionable without further
review. Round stays at 1 (no round reset on revise).
