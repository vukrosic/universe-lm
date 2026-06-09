# Code-review log — 009 fire-pe

## r1 — 2026-06-09 — verdict: accept

Read `git diff` of `configs/llm_config.py`, `models/layers.py`, `models/llm.py`,
and new `models/fire_pe.py`. Faithful, identity-safe, in budget. Ready to run.

**Checks passed:**
- **Flag OFF = bit-identical.** `use_fire_pe=False` ⇒ the new `if self.use_fire_pe:`
  branch (layers.py:1326) is unreachable; `FIREBias` is built in `__init__`
  (layers.py:634) but never called in forward. The only OFF-path text change is a
  duplicated comment (layers.py:1198-1200) — no executable change. Baseline path
  untouched.
- **Single boolean flag, default OFF.** `use_fire_pe: bool = False`
  (llm_config.py:150). `fire_pe_d_phi: int = 4` is a dim hyperparam, not a second
  on/off switch — acceptable.
- **Zero-init holds with flag ON.** `f_w_t`, `f_w_s` are `torch.zeros`
  (fire_pe.py:64-65) ⇒ `bias = γ·(W_t·φ_t + W_s·φ_s) = 0` at step 0. Treatment
  step-0 == control exactly (bias adds 0 on top of the retained RoPE path).
- **Faithful to mechanism.** `bias(t,s) = γ(|t-s|)·f([φ_t;φ_s])` with fixed Lp
  kernel `γ[d]=(1-d/d_max)^p` (fire_pe.py:66-70, monotone non-increasing) and
  per-head φ/f. The score-only reformulation `f = W_t·φ_t + W_s·φ_s`
  (fire_pe.py:78-85) is a *linear* f (no hidden layer) vs idea.md's "small MLP",
  but plan.md line 31 explicitly ships this as the v1 memory mitigation
  (O(B·H·T·d_phi) instead of the 3.2 GB pair tensor). Documented, faithful
  degenerate — not a finding.
- **Mask convention matches the existing manual path.** FIRE branch uses the same
  causal/SWA mask, `-1e9` fill, softmax, dropout, `@V`, `[B,H,T,D]` output layout
  as the alibi/cosine manual branch (layers.py:1357-1439). Output feeds the
  downstream projection identically.
- **No HP drift.** No LR/schedule/init-constant/seed change rides along.
- **Seed/tier.** plan.md pins seed 42, tiny1m3m only; matches idea.md bar
  (pass ≤ 6.4237, fail > 6.4287, noise |Δ| ≤ 0.005, ctrl 6.4287). No multi-seed,
  no larger tier referenced in the run code.
- **LoC.** fire_pe.py ~92 + integration ~25 + config ~10 ≈ 127 < 200.
- **Coordination.** Diff adds an isolated `if` branch + a new module; does not
  revert or stomp the parallel Claude's unstaged edits in layers.py / llm.py /
  llm_config.py. No rebase, no push.

**Noted (non-blocking, no recode):**
- idea.md and the code comment (layers.py:1327, llm_config.py comment) call FIRE a
  "drop-in for RoPE", but the implementation is **additive on RoPE** — RoPE is
  still applied at layers.py:1197-1208 before the FIRE branch, then the bias is
  added on top. plan.md line 17 deliberately and explicitly chose additive-on-RoPE
  and the code matches plan.md. The "drop-in" wording is loose terminology, not a
  behavioral contradiction; the run produces a valid, interpretable A/B (FIRE bias
  over the HighRoPE control). Recommend the implementer tighten the comment wording
  on a future touch, but it does not gate this run.
