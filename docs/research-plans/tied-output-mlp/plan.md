# Tied output MLP vs. embedding residual (plan)

**Research question.** Two ways to force the final-token representation to stay coupled to the *initial* token embedding:

- **Strategy A — layer-by-layer injection** (already in repo as #20 embedding residual): rms-norm `x0` once and add it into every block. Continuous, additive, distributed across depth.
- **Strategy B — tied output MLP** (this plan): couple input and output token-space *structurally* at the two ends only, via a weight-tied (autoencoder-style) MLP. Nothing is injected mid-stack.

Both encode the prior "the token you emit should live near the token-embedding manifold." A pushes that prior at every layer; B makes it an architectural constraint at the boundary. Which inductive bias is the better use of the same idea — and do they stack or cancel?

## The mechanism (Strategy B)

Today the head is `logits = norm(x) @ token_embedding^T` — already weight-tied, but the residual stream `x` and the embedding space are only linearly related (`emb_proj`/identity).

Tied output MLP inserts a shared nonlinear map so the decode path is the *transpose* of an encode path:

```text
encode (input):   h0  = x_emb + Wd · φ(Wu · x_emb)        # standard SwiGLU/FFN, run once on the embedding
decode (output):  z   = x + Wu^T · φ(Wd^T · norm(x))      # SAME Wu, Wd, transposed — tied autoencoder
logits = z @ token_embedding^T
```

`Wu`, `Wd` are one FFN's up/down projections, reused on both ends (tied, like input↔output embeddings). The decode branch is zero-init (`Wu^T` path starts at 0) so **step 0 == tied-head baseline** and the MLP earns its keep. Net new params = one extra FFN (`2·d_model·d_ff`), shared across both ends — cheaper than it looks, and the comparison vs. #20 (which adds only one RMSNorm) is reported per-param, not just per-loss.

Rationale: a tied autoencoder forces `decode ∘ encode ≈ identity` on the embedding manifold, so the network *cannot* drift the output representation arbitrarily far from where tokens are embedded. That is the same goal as re-injecting `x0`, expressed as a constraint instead of a perpetual additive nudge.

## Variants

| Variant | What | Extra params vs base | Step-0 == base? | Conf | Why |
|---|---|---|---|---|---|
| B0 Tied output MLP | shared Wu/Wd encode+decode, decode zero-init | +1 FFN (`2·d·d_ff`) | yes | med | The clean form of the idea. Structural coupling at the boundary only. |
| B1 Untied output MLP | same shape, **separate** decode weights | +2 FFN | yes | low-med | Control: isolates whether the *tying* matters or it's just "more output capacity." If B1≈B0, tying is free regularization; if B1>B0, the constraint is a cost. |
| B2 Tied, linear (no φ) | `z = x + Wu^T Wd^T norm(x)`, no nonlinearity | +1 FFN | yes | low | Ablates the nonlinearity. Likely folds into the existing linear tied head → expect ≈ baseline. Sanity rung. |
| A (ref) | #20 embedding residual, already implemented | +1 RMSNorm | no¹ | high | The incumbent layer-by-layer strategy. Head-to-head reference. |
| A+B0 | embedding residual **and** tied output MLP | +1 RMSNorm +1 FFN | no¹ | med | Do they stack (complementary: distributed + boundary) or cancel (redundant prior)? |

¹ #20 adds `x0` from step 0, so A and A+B0 are not exact baselines at init by construction.

## Implementation notes

- New flag `use_tied_output_mlp: bool` (+ `untie_output_mlp`, `tied_output_mlp_linear` for B1/B2) on `LLMConfig`.
- Encode runs once in [models/llm.py](../../../models/llm.py) right after `x = tok * emb_scale` (≈ line 214), before the block loop. Decode runs in the output block after `x = self.norm(x)` (line 243), before the tied unembed (lines 245–251). Reuse the `SwiGLU`/`FeedForward` in [models/components.py](../../../models/components.py) for `φ`, `Wu`, `Wd`; hold one instance, call `.encode()`/`.decode()` or expose the raw projections.
- **Zero-init decode** so the additive output path is a no-op at step 0 — mirror the `output_adapter_out` zero-init already at [models/llm.py:183](../../../models/llm.py:183).
- **Tying under Muon:** `Wu`, `Wd` are 2D → routed to Muon. Tied transpose means one parameter receives gradient from both ends; confirm the orthogonalized update is sane (no double-counting, grad shapes match). This is the main implementation risk — verify before trusting numbers.
- Add `Screen10M20MTiedOutputMLPConfig` (+ untied/linear/combo) in [configs](../../../configs) following the `OutputAdapter`/`ValueEmbed` config pattern, register in `configs/__init__.py`.
- Interactions to gate: `emb_rank` (factorized embedding changes the decode target space — start with `emb_rank=None`), `output_adapter_rank` (don't stack — they're both additive output paths), `use_output_embed` (#33, also touches the output side).

## Eval protocol (don't skip — Screen ≠ transfer)

1. Baseline #0 number + seed count fixed first; reuse the standard Screen10M20M harness.
2. Run A, B0, B1, B2, A+B0 at the **same** screen budget, ≥2 seeds each (the tweaks here are subtle; one seed is noise).
3. Promotion gate: report Δnats **and** Δnats-per-extra-param vs baseline. B0 must beat A *per param* to be interesting, not just beat baseline.
4. Transfer check: only promote a winner to a 25M confirm run; never ship on Screen alone (per the config docstrings' own warning).
5. Kill conditions stated up front: B2 ≈ baseline (expected) → drop. B1 ≈ B0 → tying is free, keep tied form. A+B0 ≈ max(A,B0) → redundant prior, ship only the cheaper one.

## Open questions / risks

- **Redundancy with linear tying.** The head is already `x @ E^T`. B2 tests whether anything beyond the existing linear tie helps; if B0's win is small over B2, the nonlinearity isn't doing real work.
- **Param fairness.** B0 costs a full FFN; #20 costs one RMSNorm. A "win" that just spends more params isn't a mechanism win — hence the per-param gate.
- **Transfer prior.** Boundary couplings tend to matter *more* at small `d_model` (embedding is a bigger share) and wash at scale — same trap that bit embedding-factorization. Down-weight the Screen result accordingly.

## Status / results
(add per-variant branches, seeds, A/B numbers here)
