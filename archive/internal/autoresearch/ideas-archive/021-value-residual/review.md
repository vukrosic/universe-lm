# Review log — 021 value-residual

## r3 — 2026-06-10 — verdict: approve

- **3-round cap reached** (frontmatter `round: 3`) — `revise` is forbidden, forcing the call. The r2 findings are addressed cleanly: blend site pinned to post-transpose / post-GQA right after `models/layers.py:1380` (HEAD-state lines; current working tree shows `transpose` at 1431 only because the parallel 022-softpick patch added ~51 lines above — verified by `git show HEAD:models/layers.py | grep -n "transpose(1, 2)"` → 1380), shape `[B, n_heads, T, d_k]` is what actually exists at that point; the option-A per-block scalar avoids `layer_idx` plumbing (none exists in MHA/TransformerBlock and this idea no longer requires it); the V_1 stash is pinned to MHA-side `self._v_residual = V.detach()` with model.forward orchestrating the positional-arg pass to blocks 1..N-1; closed.md pointers corrected to 44 / 41-44.
- **Source, dedup, and shape pass.** arXiv:2410.17897 ResFormer/SVA real (verified r1). Distinct from closed V-embed axis (input-side projection scaling) and the 015/016/017/020 batch sites (optimizer / per-head logit / per-sublayer / post-softmax mass). Structural change, identity at λ=0, ~36 LoC well under the 50/200 caps. tiny1m3m + seed 42 only, ctrl = FIRE-equipped 009 WIN config — correct control for "is cross-layer V mixing orthogonal to additive positional bias". Pass bar tied to box noise (WIN < ctrl−0.005, NULL band ±0.01, FAIL > ctrl+0.01).
- **One remaining inconsistency for the code-implementer to resolve at plan time (not a blocker).** LoC item (a) puts `self.lambda_v = nn.Parameter(torch.zeros(()))` on `TransformerBlock`; LoC item (c) writes the blend as `V = (1-self.lambda_v)·V + self.lambda_v·V_1` inside `MHA.forward` (where `self` = MHA, not the block). The evidence readout at line 56 reads `[block.lambda_v.item() for block in model.transformer_blocks]`, which assumes block-level storage. Two equivalent fixes — both work; the code-implementer picks one in `plan.md`: (i) keep λ on the block and pass `lambda_v=self.lambda_v` to MHA.forward (one extra arg, readout stays `block.lambda_v`), or (ii) move λ onto MHA and adjust the readout to `[block.attention.lambda_v.item() for ...]`. Option (ii) is slightly simpler (no signature change), option (i) keeps the block-scoped semantic the spec text leans on. This is a code-detail clarification, not a spec-correctness gap — the mechanism, blend site, identity-init property, and gradient path are unambiguous regardless of which is chosen.
- **Capture per-block λ at end of training** as specified (line 56) — a uniform `λ_l → 0` is a stronger null than "inside variance" and a non-monotonic λ profile across blocks is itself a result.

## r2 — 2026-06-10 — verdict: revise

- **Blend-site description contradicts the actual `models/layers.py` order of operations.** Spec (idea.md:16-27) says the blend happens "post-`qkv.split`, post-`reshape`, post-`transpose(1,2)` line, BEFORE GQA `repeat_interleave`" with shape `[B, n_kv_heads, T, d_head]`. The real code does it in the *opposite* order: V is reshaped to `[B, T, n_kv_heads, d_k]` at `models/layers.py:1257`, then GQA expansion (`repeat_interleave` on **dim=2**) happens at `models/layers.py:1293-1294` BEFORE the transpose at `models/layers.py:1380`. The shape `[B, n_kv_heads, T, d_head]` never exists at any single point in the current MHA path. Fix: pin the blend to the **post-transpose, post-GQA** site (right after `models/layers.py:1380`, before the optional `v_norm` at 1383 and well before the manual-attention branch at 1402). Stash V_1 at the *same* point in layer 0. The resulting shape is `[B, n_heads, T, d_k]` and is identical across all layers regardless of GQA settings — no broadcast worries. Rewrite the spec's "V_1 shape and blend site (precise)" paragraph to name `models/layers.py:1380` (or the line directly after) and the shape `[B, n_heads, T, d_k]`.
- **`layer_idx` plumbing is unspecified and the current code carries no such field.** `grep -n "layer_idx" models/layers.py models/llm.py` returns zero hits — neither `MHA` nor `TransformerBlock` knows its own depth. The LoC-budget item (a) (`nn.Parameter(torch.zeros(n_layers))` on the model) is unreachable without first plumbing `layer_idx` to every block. Pick one of two concrete shapes and pin it: **(option A — preferred, simpler)** drop the model-level vector and store `self.lambda_v = nn.Parameter(torch.zeros(()))` (a 0-dim scalar) **on each `TransformerBlock`**, init 0; the blend reads `self.lambda_v` locally. The per-block λ readout for `evidence.md` is then `[block.attention.lambda_v.item() for block in model.transformer_blocks]`. **(option B)** pass `layer_idx: int` through `TransformerBlock.__init__` and store it on `self`, then read `self.lambda_v[self.layer_idx]` from a model-level `nn.Parameter(torch.zeros(n_layers))` registered on the model. Spec must pick one; (A) is ~3 LoC less and avoids touching the model loop. Update the LoC budget (a) accordingly.
- **V_1 stash mechanism is internally inconsistent.** Idea.md:27 says "The model stashes it on `self` during the layer-0 forward (`self._v_residual = V.detach()`) and consumes it in the per-block forward" but the LoC budget (b) says "stash V_1 in layer-0 forward (`self._v_residual = V.detach()`) and **pass via positional arg through `TransformerBlock.forward` → `MHA.forward`**". These are two different mechanisms — a stash-on-self requires each block to know its parent model (no such backreference exists in `models/llm.py`), while a positional-arg pass is orchestrated by the **model's** outer forward loop. Pick one and pin it: **prefer the positional-arg path** — model.forward stashes V_1 as a local variable, passes `v_residual=None` to block 0 and `v_residual=V_1` to blocks 1..N-1. Block forward receives it and passes it through to MHA. This means `models/llm.py` (the model's forward loop) takes a small edit too — add that to LoC budget (b) explicitly. Also: clarify *how* layer-0's MHA returns V_1 to the model — either by setting `self._v_residual` on the MHA module after computing V (then model reads `block.attention._v_residual` after the block call), or by changing the block/MHA signature to return `(out, v_residual_or_None)`. Pin one; the `self._v_residual` on the MHA path is simplest (~2 extra LoC, no signature change).
- **Minor: closed.md line pointers drift.** Idea.md cites `closed.md:43` for the 009 FIRE WIN entry, but the WIN line is at `closed.md:44` (`009-fire-pe — WIN: trt=6.3234 …`); the ctrl-spread reference `closed.md:33-40` should be `closed.md:41-44` (the run lines for 001/004/005/009 carry the 6.3875/6.4050 ctrl bracket). Low-impact (numbers are right, pointers stale by ~4 lines after the recent batch of appends) but fix while you're in there.

## r1 — 2026-06-10 — verdict: revise

- **Source verified.** arXiv:2410.17897 resolves: "Value Residual Learning",
  Zhou/Wu/Jiang, 23 Oct 2024 v1. Mechanism (cross-layer V shortcut, per-layer
  scalar λ_l, identity at init) matches the paper. Not a fabrication. Not in
  `closed.md` (closed `V/Q/K/O embeds` axis is input-side projection scaling, a
  different site than this layer-0-V stream).
- **Definition section missing — blocks the code gate.** The idea has Source /
  Mechanism / Why, but no `## Definition` block with ctrl/trt/pass-bar/seed/LoC
  budget — compare 020-forgetting-attn/idea.md which has the full Definition
  required by gate 2. Add a `## Definition (gate 2)` section with all five
  subsections (### Ctrl vs trt, ### Pass bar, ### Seed, ### LoC budget, and the
  per-block λ readout in ### Evidence to capture).
- **Pin ctrl/trt to the FIRE-equipped baseline** (same shape as 020 in
  `020-forgetting-attn/idea.md:78-86`). Ctrl =
  `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True`
  (`configs/llm_config.py:773`, FIRE flag = the 009 WIN config from
  `closed.md:43`). Trt = ctrl + `use_value_residual=True`. Stacking on FIRE tests
  the orthogonality of cross-layer V mixing to additive positional bias — the
  same question 020 tests for multiplicative decay. (Do **not** test against the
  un-FIRE'd baseline; that's the wrong control given 009.)
- **Pin numerical pass bar with a number.** Idea says "small but real val-loss
  drop" — that fails the "falsifiable pass/fail bar tied to a real control"
  check. Use the same shape as 020 idea.md:88-96 calibrated to the FIRE-ctrl
  noise floor (closed.md ctrl spread 6.3875–6.4050 = 0.0175; the FIRE ctrl will
  re-bracket but assume the same ~±0.01 box noise): **WIN** `trt < ctrl − 0.005`
  (the taste reviewer's caveat #1 explicitly asks for this band, low-to-moderate
  bar because the bet is at the small end of the paper's effect), **NULL**
  `|trt − ctrl| < 0.01`, **FAIL** `trt > ctrl + 0.01`.
- **Pin λ shape to per-block (not global) and state it explicitly.** Taste r1
  caveat #3 flags this as a "decide and pin" item. Spec must say:
  `self.lambda_v = nn.Parameter(torch.zeros(n_layers))` on the model, with
  `lambda_l = self.lambda_v[layer_idx]` blended in each block — **not** a single
  global scalar, and **not** per-head. This is the paper's canonical lever.
- **Pin the blend site to remove ambiguity.** Idea says "blend before attention
  output" — ambiguous between (a) blending V before `attn @ V` and (b) blending
  the attention output before `o_proj`. The paper's mechanism (and taste r1
  caveat #4) is (a): `V_1` is the **projected V at layer 0**, stashed
  post-`W_V@x`, and the blend `V_l ← (1−λ_l)·V_l + λ_l·V_1` happens **before**
  the `attn_weights @ V` matmul in every later block. Spec must name the exact
  insertion point in `models/layers.py` (the MHA.forward path that 020 also
  touches, around the V-projection line) — not just "before attention output".
- **Pin where V_1 lives.** Add to spec: V_1 is a forward-pass-local stash on
  the model (not a persistent buffer, not a `nn.Parameter`); pass it via a
  positional arg through `TransformerBlock.forward` or a per-forward attribute
  on the model. Confirm shape `(B, H, T, d_head)` matches layer-l's V.
- **Add λ_l logging to the evidence requirement.** Taste r1 caveat #2: a uniform
  `λ_l → 0` post-training is a **stronger null** than "inside variance" (says
  the model rejected the shortcut). Spec must log post-train `λ_l` values per
  block to `evidence.md`. Add an "### Evidence to capture" subsection naming
  this artifact explicitly.
- **LoC budget breakdown.** Mirror 020's (a)/(b)/(c)/(d)/(e) decomposition in
  020-forgetting-attn/idea.md:103-115 so the code-implementer has a target
  shape: (a) `nn.Parameter` `lambda_v` on the model ~3 LoC; (b) stash V_1 in
  layer-0 forward, pass via kwarg ~6 LoC; (c) blend in `MHA.forward` of later
  layers ~4 LoC; (d) flag wiring (`use_value_residual` in `MHA` +
  `TransformerBlock` + `LLMConfig` + new config class
  `Tiny1M3MVResidualOnFireConfig`) ~12 LoC; (e) step-0 identity test
  (`use_value_residual=True` ≡ baseline because λ=0 → 1e-5 tolerance) ~10 LoC.
  Total ~35 LoC ≤ 50 cap.
