# Review log — 147 DropKey

## r1 — 2026-06-14 — verdict: revise
- **Missing pass/fail bar tied to a real control.** The plan lists a
  `train_llm.py --use_drop_key --drop_key_rate 0.1 --seed 42` run command but
  does not name a numeric WIN / NULL / DRIFT threshold against the
  tiny1m3m baseline. Add a `## Pass bar` section using the **current
  cached baseline** at `autoresearch/baseline-cache.json` (box 5b8a7fea8963,
  commit 7f7fe90): `val_mean=6.4394`, `val_std=0.01`, `noise_band=0.04`.
  Suggested: **WIN iff `trt < 6.4394 - 0.04` (= 6.3994); NULL if
  `|trt − 6.4394| ≤ 0.04`; DRIFT if `trt > 6.4394 + 0.04` (= 6.4794) or
  step-0 val diverges from the ctrl step-0 val by > 1e-3 (bit-identity
  claim verification, see closed 150-xlayer-feedback lesson).** Cite
  `autoresearch/BASELINE-CACHE-DESIGN.md` for the noise-band
  definition. This is the dominant missing piece — a 0.94M model running
  92 update steps has val variance ~±0.01 within a session and ~±0.04
  across boxes; without a pre-registered bar the runner can't classify
  the result.
- **Mechanism implementation diverges from the paper's exact
  formulation.** The plan (and the already-plumbed forward at
  `models/layers.py:2174-2182`) uses `K = K * M / (1-p)` (inverted-
  dropout rescale). The DropKey paper (Xu et al. 2022) instead adds an
  additive log-mask: `attn_scores = QK^T/√d + log(M)` so the dropped
  positions get −inf logit ⇒ exactly 0 in softmax (and excluded from
  the denominator). The proposed K-rescale is mathematically
  "K-dropout with inverted scaling" — it is a real lever and a real
  regularizer, but it is **not** the paper's mechanism. Two options:
  (a) keep the K-rescale implementation and re-name the lever in
  `idea.md` as "K-mask (K-dropout with rescale; inspired by DropKey)"
  so the citation is honest; or (b) switch the forward to
  `scores.masked_fill(mask==0, -inf)` / sparse-softmax so the
  denominator actually excludes the masked keys (paper-faithful).
  Either is fine, but pick one and document the choice. If (a),
  tighten the `## Source` paragraph — drop the "applied to Vision
  Transformers in the paper itself" claim unless you have read the
  paper and confirmed it (the paper's title frames ConvNeXt/CNN,
  not ViT, and the ViT extension is in a separate follow-up).
- **Closest neighbor is null at this scale; expected-Δ should be
  tight.** 111-DropPath (the only other "regularize the attention
  path" lever in the queue) was null at tiny1m3m (Δ=+0.0535/+0.0469
  wrong-sign, well past the 0.005 DRIFT band — see closed.md 2026-06-13
  entry). DropKey's bet is that *per-key* granularity fires where
  *per-residual-branch* didn't. State the prior explicitly: with
  this null prior and box noise ±0.04, the *detectable* effect at
  0.94M is ~±0.04 — anything inside that band is mathematically
  unresolvable. A "soft pass bar" (e.g. expected Δ ∈ [−0.04, +0.04])
  is not a pass bar; the bar in the previous finding is the only
  defensible one.
- **DropKey's other closed twins to mention.** The current "Closest
  neighbor" paragraph only names 111-DropPath. Worth adding (1–2
  sentences) that 152-attn-logit-bias and 155-per-head-temp are the
  other "per-head attention-shape" levers that closed null at
  tiny1m3m on 2026-06-14 — both inside the ±0.04 band. DropKey is
  categorically different (regularizer on K, not a per-head shape
  on the logits), but the family of "per-head attention tweak"
  levers is now a closed axis at this tier; the regularization
  family is still open. Helps the runner / tinker-place-the-result
  step route correctly if this comes back null.
- **Transfer-risk tag is correctly `med`.** Confirmed: the paper
  validates gains on ImageNet-scale CNNs and ViTs, not on LMs at
  any scale. At 0.94M the lever has 0 prior; `med` is the right
  setting. No change needed.
- **Implementable, not duplicate, sources real.** Source arXiv:2207.01058
  resolves, mechanism is genuinely distinct from prior closed
  levers, plumbed-forward LoC is ~10 (in the `if self.use_drop_key
  and self.training and self.drop_key_rate > 0.0:` branch at
  `models/layers.py:2174-2182` plus the 2 kwargs at
  `models/layers.py:860-861` / `:1358-1359` / `:2785-2786` /
  `:3174-3175` and the 2 fields at `configs/llm_config.py:614-615`).
  `_arq_147-dropkey.py` launcher exists in repo root. These are
  not blockers — the bar/citation/prior findings above are.
