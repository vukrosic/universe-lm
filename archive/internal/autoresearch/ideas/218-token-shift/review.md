# Review log — 218 token-shift

## r1 — 2026-06-16 — verdict: approve

**Source is real and current.** Hyena (Poli et al. 2023, arXiv:2302.10866), MEGABYTE (Yu et al. 2023, arXiv:2305.07185), StripedHyena (DeepMind 2023) — all resolvable, all use local convolutions before or interleaved with attention, all >100M source scale. The "transfer risk: low" tag matches the cited scale.

**Mechanism is a mechanism, not a hyperparameter.** Pre-softmax depthwise causal k=3 Conv1d on Q and K with additive residual and zero init. Step-0 ≡ baseline by construction (the conv output is multiplied by zero-init weights before the add, so `Q' = Q + 0 = Q` and `K' = K` bit-identical). Distinct lever, not an LR/init-constant tweak.

**Not closed (mathematically distinct position).** Cross-checked `autoresearch/closed.md`:
- 143-shortconv — pre-attn **V** depthwise k=3 causal, *borderline null*
- 157-conv-ffn — post-FFN-activation depthwise, *null*
- 163-v-mix-conv — post-attn **V** depthwise k=3, *clean null*
- 180-qk-logit-conv — on the **scores** (not vectors), *LEAK (rejected) — broken causal mask*
- 191-token-attn-gain — per-tensor 1-D kernel on the *head attention output*, *null*

218 sits on **Q and K vectors pre-softmax** — a position none of the closed levers occupy. 180's LEAK is the closest in spirit (it tried to mix QK together, on the scores) and 180's failure mode (causal-mask break) is exactly the construction pitfall 218 must avoid. The two are not equivalent: 180 mixed the *post-softmax* logits; 218 mixes the *pre-softmax* vectors. Different tensor, different operation.

**Tiny1m3m only.** The design sketch names the tier, the +72 scalar param count (+0.008% of 0.94M, well under the 1% envelope), and a 92-step compatible operation (two extra Conv1d applications per layer, k=3, depthwise → negligible FLOPs). No `screen20m` references, no full-ladder promotion language.

**Implementable in <200 LoC.** The apply site is one method (`MultiHeadAttention.forward` in `models/layers.py`, after `Q = self.q_proj(x)` / `K = self.k_proj(x)` and before the head reshape), plus one new `use_token_shift: bool = False` field in `configs/llm_config.py` and an inline `@dataclass` subclass to activate. No new file, no new optimizer, no new dataset plumbing.

**Falsifiable pass/fail bar is present and tight enough to resolve at tiny1m3m.** Plan bar (per taste): Δ ≤ −0.01 = WIN, |Δ| < 0.01 = NULL, Δ > +0.01 = DRIFT, against a two-ctrl bracket at tiny1m3m and the cached baseline 6.2403 ± 0.04. Δ expected −0.01 to −0.03 sits inside the noise band — a real effect (not a sub-noise screen) is the testable claim, and a NULL is itself informative (closes the third axis of the 143/163/218 trio at 0.94M). The taste reviewer's stated null hypothesis ("global attention pass already absorbs local structure at 0.94M") is what 143 and 163 also tested against — 218 may fail the same way, but on a different tensor where the failure mode might look different.

**Findings to pin in `plan.md` (not blocking — the implementer MUST carry these, since they are the difference between a clean run and a LEAK like 180):**

- **Causal padding must be manual left-pad of length `k−1 = 2` on the `T` axis (a `F.pad(x, (2, 0))` after the `(B, d_model, T)` reshape), NOT `nn.Conv1d(..., padding=k//2)` symmetric.** Symmetric padding would let the conv see the future token at the boundary and produce a non-bit-identical step-0, plus leak future info into the attention pattern — exactly 180's failure mode on a different tensor.
- **Construct the conv as a raw `nn.Parameter(torch.zeros(d_model, 1, 3))` (or equivalent) explicitly zeroed — NOT `nn.Conv1d(d_model, d_model, 3, groups=d_model, padding=0)` then `.data.zero_()`.** `nn.Conv1d`'s constructor runs `kaiming_uniform_` on the weight, consuming the RNG state for every block's `q_proj`/`k_proj` random init downstream. Bit-identical step-0 across all 12 blocks at the no-flag path vs the flag-on path requires the explicit-init path. Same fix 157/163 plans already enforce.
- **Q and K each get a separate conv (24 total at 12L × 2 per-tensor).** Do NOT share weights across Q and K — they sit on different gradients and tying them would create a phantom cross-coupling the optimizer would have to undo.
- **Apply site is `models/layers.py` `MultiHeadAttention.forward`, AFTER `Q = self.q_proj(x)` and `K = self.k_proj(x)`, BEFORE the `(B, H, T, d_k)` head reshape.** Cite the exact line in `plan.md`. Verify the ALiBi bias (175) and `qk_norm` (016) hooks are still applied to `Q'`/`K'` post-mix, not to raw `Q`/`K` — pin the ordering.
- **Bit-identical step-0 self-check is mandatory.** Forward the no-flag model and the flag-on model on a fixed input; assert `fp32 max-abs-diff < 1e-6` on the logits and on every block's `Q'`/`K'` after the mix. No identity check, no run — same bar 191, 175, 154 enforced.
- **Route the conv weights to AdamW (the head optimizer), not Muon.** Per-tensor 1-D kernels on a non-orthogonal-2-D parameter don't fit Muon's Newton-Schulz orthogonalization step; 191 set the precedent. Tiny param count (72 scalars) means this is a routing concern, not a memory concern.
- **Active treatment via inline `@dataclass` subclass on `LLMConfig`** (the established pattern, e.g. `LLMConfig175ALiBI` in `configs/llm_config.py`). The base `use_token_shift: bool = False` defaults to off → existing configs bit-identical. Cite the subclass name in `plan.md`.
- **No interaction with the active champion stack** (175-alibi + 154-rebased + 016-qk_norm + 021-value-residual): 218 mixes Q/K *before* the dot product and the post-mix `Q'`/`K'` still go through the existing `qk_norm` and ALiBi hooks unchanged. Verify the plan calls this out and the diff is a single, localized edit to `MultiHeadAttention.forward`.

**Verdict: approve.** The lever is sound, the position is novel, the bit-identical claim is plausible (with the explicit-init construction), the bar resolves at tiny1m3m, and the implementer has clear guardrails to avoid 180's LEAK. `round` reset to 1 for the code gate.
