---
id: 052-fixnorm
status: needs-plan
round: 2
updated: 2026-06-11T01:22:23Z
transfer-risk: med
---

# 052 — FixNorm (fixed-length token embeddings)

## Source
Nguyen & Salazar, "Transformers without Tears: Improving the Normalization of Self-Attention" (arXiv:1910.05895). The same paper also reaffirms FixNorm: normalizing word embeddings to a fixed length.

## Mechanism
Project token embeddings to a fixed L2 radius before they enter the stack, and optionally keep the tied output side on the same radius. This is an embedding-side norm lever: it clamps the magnitude of the earliest residual injection without touching attention or FFN math.

## Scale evidence
The paper reports that fixed-length embeddings help low-resource translation and remain competitive in the high-resource WMT14 EN-DE setting. transfer-risk: med — the mechanism is simple and portable, but the strongest evidence is still translation, not language-model pretraining.

## Why it's worth a slot
If input embedding norm is the real bottleneck, FixNorm should move loss even when the rest of the model is unchanged; if it does not, embedding magnitude is probably not the culprit.

## Spec (gate 2 — fixes the recipe so the plan-stage has nothing to invent)

### 1. Radius parameterization
One **global learnable scalar** `g = nn.Parameter(torch.empty(()))` (paper default, Nguyen & Salazar §3.2). Init `g_init = 0.02 · √d_model` so the post-FixNorm mean row magnitude matches the baseline pre-FixNorm mean row magnitude — at tiny1m3m `d_model=64` ⇒ `g_init = 0.02 · 8 = 0.16`. The model is free to learn a different radius from there. Per-row or per-token radii are out of scope for this A/B.

### 2. Interaction with `embedding_scale = √d_model`
**Replace, don't multiply.** When `use_fixnorm=True`, the spec sets `embedding_scale = -1` (the "use sqrt(d_model)" sentinel in `models/llm.py:563-565`), so the existing `tok * emb_scale` line becomes a no-op and `g` is the **only** magnitude knob on the input side. Reasoning: FixNorm is the explicit magnitude lever — keeping the `√d_model` scalar on top would compound two independent scaling conventions and make the lever's effect non-isolable. The wiring is: a single `if use_fixnorm: emb_scale = 1.0` override in `LLM.forward` (the gating check is one line; the existing `emb_scale < 0 → sqrt(d_model)` branch stays untouched for the off case).

### 3. Position w.r.t. low-rank embedding (`emb_rank`)
**Post-projection d_model** vector. The normalization is applied to the d_model-dim activation that enters the transformer stack (`x` after `emb_proj(tok) * emb_scale` in `models/llm.py:567-569`), NOT to the rank-r table itself. This means the lever applies uniformly across `emb_rank=None` (full vocab×d_model) and `emb_rank=8` (factorised) — the normalization is one site, one line. Tiny1m3m uses `emb_rank=8`; the post-projection site is the only place the lever can sit without entangling with the factorisation.

### 4. Output side
**Input-side only.** `lm_head.weight = token_embedding.weight` stays shared by reference at `models/llm.py:509-510` (full-rank case), and the tied factorised softmax path is unchanged. Logits are computed from the **raw** shared weight — no FixNorm at the output site. The spec is the paper's "untied" variant on the output side; the "tied" variant (FixNorm on both sites) is a separate follow-up A/B if the input-side result is interesting. Rationale for input-side only: the *parameter* under test is the embedding-radius lever; adding a second site doubles LoC and confounds the A/B. Keep the lever single-site at this gate.

### 5. The actual op (precise, for the code-implementer)
```python
# Right after models/llm.py:569 (post emb_proj, post emb_scale), in LLM.forward:
if self.use_fixnorm:
    x = F.normalize(x, p=2, dim=-1) * self.fixnorm_g   # g is a learnable scalar
# Then x enters the transformer stack exactly as before.
```
One `nn.Parameter`, one `if`, one op. ~5 LoC. `F.normalize` with `p=2, dim=-1` is the standard per-row L2 normalization; it returns unit vectors along `dim=-1`, then we scale by `g` to land them on the radius. No eps clamp needed for rows of nonzero mass (init `N(0, 0.02)` rows have nonzero norm with prob 1). `F.normalize` handles the zero-row edge case by returning zero, but vocabulary rows that are never seen in training would have stale gradient — acceptable for tiny1m3m's full vocab coverage (~49k of 49k touched in 3M tokens).

### 6. Step-0 is not bit-identical (justified, not hidden)
After step 0:
- `tok` rows have `‖row‖₂ ≈ 0.02 · √d_model = 0.16` (baseline init at `models/llm.py:556`).
- After FixNorm: `x = unit · g_init = unit · 0.16` ⇒ `‖x‖₂ = 0.16` per row, per the spec's init choice.
- Baseline path (no FixNorm): `x = tok · √d_model` ⇒ `‖x‖₂ ≈ 0.16 · 8 = 1.28` per row.

So **mean per-row magnitude does NOT match**: baseline ~1.28, FixNorm at init 0.16. This is a real step-0 magnitude shift. The spec owns it explicitly: **FixNorm is not bit-identical to baseline at step 0 even with `g_init = 0.02·√d_model`**, because the baseline multiplies by `√d_model` *after* the embedding table is read, while FixNorm re-projects to a fixed length and removes the per-row magnitude variation.

The reason the lever is interesting is the *equalization across rows*, not the *global magnitude*: previously some rows had high norm, others low; FixNorm says "set them all to `g`, let the model learn what `g` should be." The step-0 distribution is therefore the test condition — that is what the A/B measures. The reviewer's option (b): "magnitude equalization across rows is the mechanism and is what we want to test from step 0." That is the chosen justification; the global-magnitude mismatch is the cost, owned in writing.

The runner MUST run the step-0 smoke gate (per the 022-softpick precedent at `autoresearch/ideas/022-softpick-attention/idea.md`): build trt config, run one fwd+bwd, assert loss is finite, embedding grads are non-zero, and `g.grad` is non-zero (every row contributes a `g` term to its loss, so the gradient on `g` should fire at step 0).

## Definition (gate 2)

### Ctrl vs trt
- **Ctrl**: `Tiny1M3MConfig` (`configs/llm_config.py:665`; val **6.4306** per `LEADERBOARD.md` row 14 — the plain tiny1m3m baseline). This is the cleanest A/B for an embedding-magnitude lever: no other flags on, so the Δ isolates FixNorm. The FIRE-equipped baseline (`Tiny1M3MConfig + use_fire_pe=True`, val 6.3234 per `closed.md:40`) is an **orthogonal stacking A/B** for a future follow-up if the plain A/B is interesting; not this gate's primary ctrl.
- **Trt**: same config + `use_fixnorm=True, fixnorm_radius_init=0.16` (init `g = 0.02·√d_model = 0.02·8 = 0.16` for tiny1m3m). New config class `Tiny1M3MFixNormConfig(Tiny1M3MConfig)` with `use_fixnorm: bool = True`. The `fixnorm_radius_init` is plumbed as an init-time override on `g` after the `nn.Parameter` is created (mirrors the `zero_init_resid` post-init pattern in `models/llm.py:534-538`).

### Pass bar (tiny1m3m noise floor)
Run-to-run val-loss variance at this tier is ≈ ±0.01 (`closed.md:41-44` ctrl spread 6.3875–6.4050 = 0.0175 for the FIRE-equipped ctrl; the plain `Tiny1M3MConfig` baseline is 6.4306 with a similarly-bounded bracket). With a single seed the pass bar must clear the in-session ctrl bracket, not just sit inside it:
- **Win**: `trt_val < ctrl_val − 0.005` (low-to-moderate bar; FixNorm is the cheapest member of the normalization family and the taste review put its leverage at the low end of the hypothesis range — a "should move loss" bet, not a "big if true" bet).
- **Null**: `|trt_val − ctrl_val| < 0.01` (sub-noise = inconclusive, not real; never add seeds to confirm).
- **Drift**: `trt_val > ctrl_val + 0.01` (FixNorm hurts — the embedding-magnitude hypothesis is closed for tiny1m3m).

### Seed
**Seed 42 only.** Single fixed seed, no multi-seed sweep, no per-seed mean. A sub-noise delta is *inconclusive, not real*; never add "run more seeds to confirm" — log null and move on. The runner must run a fresh in-session ctrl bracketing the trt run (ctrl₀, trt, ctrl₁) to validate the box, per the standard protocol.

### Evidence to capture
- `g` (the FixNorm radius scalar) value at start and end of training — confirms the model learned to move the radius from `g_init = 0.16`. A `g → g_init` post-training is a *stronger* null than "inside variance": it means the model rejected the per-row equalization and kept the baseline magnitude. A `g ≪ g_init` (e.g. `g → 0.08`) means the model wants a tighter radius. A `g ≫ g_init` means looser.
- `g.grad` snapshot at step ~half — confirms the gradient is flowing through the normalization and the lever is trainable, not dead.
- The A/B val-loss and step-time — the standard A/B output.

## Failure modes (gate 2)

- **FixNorm + low-rank embedding emb_proj interaction**: the normalization runs *after* `emb_proj`, not before. If it ran before, it would normalize rank-r vectors (vocab×r), which are sparse in r and would lose the r-axis structure. Post-projection is the only sane site. Spec §3 owns this; the code-implementer is not free to move it.
- **Tied output side**: the spec does NOT apply FixNorm at the output side (see §4). If the runner sees a config with `use_fixnorm=True` and a non-tied `lm_head` (e.g. `use_untied_head=True`), the lever is still input-side only — `lm_head` is not normalized. A separate `use_fixnorm_output` flag is a different lever; not this A/B.
- **Dead-grad on `g`**: if `g.grad == 0` at every step (vanishingly rare, but possible if all rows happen to be exactly unit-norm at some batch and `g` becomes a constant multiplier), the A/B is uninformative. The runner must surface this in `evidence.md` and log it as a mechanism-execution failure, not a val-loss null.
- **g runaway**: `g` is unbounded (positive scalar, no clamp). If `g → 1000` mid-training, the residual stream explodes; the existing grad-clip (`grad_clip=1.0`) should catch the downstream effect, but a runaway `g` is a clear sign the model is exploiting the lever in a degenerate way. The runner should log `g`'s max value over training and flag if it exceeds 10× `g_init`.
