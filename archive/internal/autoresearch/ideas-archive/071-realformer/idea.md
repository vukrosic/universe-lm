---
id: 071-realformer
status: needs-plan
round: 2
updated: 2026-06-11T01:22:23Z
transfer-risk: med
---

# 071 — RealFormer (residual attention across layers)

## Source
RealFormer: Transformer Likes Residual Attention (He et al., arXiv:2012.11747, Dec 2020). The mechanism is the lever; the publication is just where it was first named.

## Mechanism
Cache the previous layer's pre-softmax (post-positional-bias, pre-mask) attention score tensor and add it into the current layer's scores before the softmax: `scores_l = scores_l + scores_{l-1}`, with the cache **detached** across layers so each layer's softmax is trained on its own path and the carry only learns the cross-layer routing shape. Layer 0 receives `prev_scores=None` and adds nothing — step 0 ≡ baseline by construction at layer 0 (no carry, no add ⇒ scores are bit-identical to the FIRE-ctrl). This makes the lever an identity-init at step 0, like 021's λ=0 V-blend.

## Scale evidence
RealFormer's central claim is **deeper-is-better via score carry**: the score residual lets later layers attend to a structure shaped by earlier layers' patterns, and the paper reports the effect surviving across BERT-base (12L), BERT-large (24L), and BERT-xlarge (36L) MLM pretraining, plus ETC and downstream GLUE / SQuAD / NMT / WikiHop / HotpotQA / NQ / OpenKP. The mechanism does not require deep stacks to fire at all — it imposes a soft cross-layer attention-pattern prior, which a 6L causal LM can still register — but the magnitude of the effect should scale with the number of carryover transitions (12L → 11 carries; 6L → 5 carries). transfer-risk: **med**: the lever is structural and identity-init (so it is a no-regret carry to a 135M recipe if it lands here), but the published wins are not specifically on a 0.94M causal LM and the expected magnitude at L=6 sits at the low end of the paper's reported band.

## Why it's worth a slot
Residual attention tests whether the model wants attention state to persist across depth instead of being recomputed fresh every layer; the 5 carryover transitions at L=6 are small but non-zero. A WIN says the tiny model wants depth-wise attention carry even on top of the FIRE-equipped baseline; a NULL says recompute-from-scratch is fine at this scale and the 135M recipe can drop the residual-attention family. **Expected magnitude band is low** (a real win here is `-0.005 ≤ Δ ≤ -0.015`, not `-0.02+`); the runner should not mistake a real-but-tiny `-0.003` move for a null.

## Definition (gate 2)

### Ctrl vs trt
- **Ctrl**: `Tiny1M3MVQGainSWAHighRoPE250KConfig + use_fire_pe=True` (`configs/llm_config.py:864-918` + the `use_fire_pe` flag — the 009 WIN config from `closed.md:44`; the de-facto control shared by 020/021/023/024/025). The FIRE-equipped baseline is the correct control: 071 stacks on FIRE to test the orthogonality of cross-layer score carry to additive positional bias. **Do not** test against the un-FIRE'd baseline — that re-litigates 009, not 071.
- **Trt**: same config + `use_realformer=True`. New config class `Tiny1M3MRealFormerOnFireConfig(Tiny1M3MConfig)` with `use_fire_pe: bool = True, use_realformer: bool = True` (mirroring the `Tiny1M3MVResidualOnFireConfig` shape at `configs/llm_config.py:1090`).

### Pass bar (tiny1m3m noise floor)
Run-to-run val-loss variance at this tier is ≈ ±0.01 (`closed.md:41-44` ctrl spread 6.3875–6.4050 = 0.0175). With a single seed and only 5 carry transitions at L=6, the expected effect is at the low end of the paper's band; the pass bar must clear the noise floor and not just sit inside it:
- **Win**: `trt_val < ctrl_val − 0.005` (low-to-moderate bar; 5 carryover transitions at L=6 is small but non-zero, matching the taste r1 reviewer's expectation and 021's bar).
- **Null**: `|trt_val − ctrl_val| < 0.01` (sub-noise; the lever does not fire on top of FIRE at this scale).
- **Fail**: `trt_val > ctrl_val + 0.01` (the score carry is hurting, not helping — the cross-layer prior is fighting the model's per-layer attention rather than reinforcing it).

### Seed
**Seed 42 only.** Single fixed seed, no multi-seed sweep, no per-seed mean. A sub-noise delta is *inconclusive, not real*; never add "run more seeds to confirm" — log null and move on.

### LoC budget (≤ 50 LoC, well under the 200 ceiling)

**Cache shape & site (precise).** The cache is the **pre-softmax, pre-mask, post-positional-bias** attention score tensor, shape `[B, n_heads, T, T]` (T = `seq_len`). Site: the FIRE branch in `models/layers.py:1591-1652` already forces the manual-attention path (same site 021's `_v_residual` piggybacks on at `models/layers.py:1380`), so 071 needs no extra manual-branch forcing. The carry-add and stash live **after** all optional pre-softmax additive/multiplicative biases (FIRE at L1598, optional CoPE at L1602, optional SSMax at L1617, optional FoX at L1633) and **before** the causal mask at L1639. This composes the cross-layer carry on the **full** pre-softmax tensor the paper specifies.

**Plumbing (mirrors 021's `_v_residual` shape).** (i) Forward-pass-local stash on the MHA module: `self._scores_cache = scores.detach()` after the carry-add. (ii) `models/llm.py:forward()` outer loop reads `block.attention._scores_cache` after each block call and passes it as a positional arg `prev_scores=` to the next block's MHA. (iii) Layer 0 receives `prev_scores=None`; the MHA guard `if self.use_realformer and prev_scores is not None: scores = scores + prev_scores` short-circuits the add, so layer 0 scores are bit-identical to baseline. The `.detach()` on the stash means the layer-l add's gradient does not flow back into the layer-(l−1) attention logit computation — each layer's own softmax is trained on its own path; the cross-layer carry learns only the routing shape. (Same `.detach()` discipline as 021's V stash.)

- (a) `use_realformer: bool = False` kwarg on `LLMConfig`, `MultiHeadAttention`, and `TransformerBlock`, plumbed through the constructor chain (`configs/llm_config.py` + `models/llm.py` + `models/layers.py` — small additions analogous to 021's flags): ~10 LoC
- (b) every-layer MHA stash of `scores.detach()` (after the carry-add) and outer-loop `prev_scores` plumbing in `models/llm.py:forward()` analogous to 021's `v_residual` plumbing; model reads `block.attention._scores_cache` after each block call and passes it positionally to the next block (`prev_scores=`); layer 0 receives `None`: ~10 LoC
- (c) in the MHA FIRE-branch (and, for forward-compat with future non-FIRE controls, the elif manual branch at L1653), add `if self.use_realformer and prev_scores is not None: scores = scores + prev_scores` **right after the last pre-softmax additive/multiplicative bias and before the mask** (after L1633 / before L1639 in the FIRE branch), then `self._scores_cache = scores.detach()` immediately after the same add: ~6 LoC
- (d) new config class `Tiny1M3MRealFormerOnFireConfig(Tiny1M3MConfig)` exported from `configs/__init__.py`, with `use_fire_pe: bool = True, use_realformer: bool = True`: ~5 LoC
- (e) step-0 identity test: `use_realformer=False` ≡ baseline at step 0 (trivially, since the flag short-circuits both the add and the stash); AND `use_realformer=True` with `prev_scores=None` at the layer-0 site produces bit-identical pre-softmax scores to `use_realformer=False` at layer 0, within `1e-5`. This is the actually-bit-identical assertion (see Reviser note (1) below for why the looser "step 0 ≡ baseline at all layers" version of the test is malformed once layer 1 reads layer 0's non-zero stash): ~10 LoC

Total ≈ 41 LoC, well under 50 and the 200 ceiling.

### Evidence to capture
- Per-block carryover magnitude at **end of training**: `||scores_l − scores_{l-1}||_F / (||scores_{l-1}||_F + 1e-6)` for each block transition (5 scalars at L=6), collected once via the existing `prev_scores` plumbing during a final eval batch and appended to `evidence.md`. A uniform "carry has near-zero magnitude" is a **stronger null** than "inside variance" — it means the model learned to leave the previous score un-touched. A non-monotonic profile (carry grows with depth, or is non-zero only in middle layers) is itself a finding. Mirror 021's per-block λ readout (`autoresearch/ideas/021-value-residual/idea.md:55-58`).
- The A/B val-loss and step-time — the standard A/B output.

## Reviser note (r1 → r2)
Three places I deviated from the r1 findings, flagged for the r2 reviewer to adjudicate:

1. **Stash site = post-add, not pre-add.** The r1 finding asked to pin the stash site but did not decide pre- vs post- the cross-layer carry-add itself. I chose **post-add**, matching the paper's recurrence `scores_l ← scores_l + scores_{l-1}` where the LHS *is* the next layer's `prev_scores` (cumulative carry). The alternative (stash the pre-carry layer-l scores) gives an adjacent-only carry where only neighbouring layers communicate. The paper is the cumulative form. If the r2 reviewer prefers the adjacent-only carry, this is a one-line move of the `_scores_cache =` assignment to before the add.

2. **Lever wired in both branches.** The r1 finding only required the FIRE branch (the current ctrl). I'm also wiring it into the elif manual branch (L1653 onward) for forward-compat with future non-FIRE controls — 2-3 extra LoC inside the 50 budget. Pure cleanliness, not a behavior change for this A/B.

3. **Identity-init scope.** The r1 finding sketched two ways to make step 0 ≡ baseline: skip-add at L0, or zero-tensor stash at L0. Both make **layer 0** bit-identical to baseline at step 0, but neither makes layers ≥1 bit-identical to baseline at step 0 once the carry from L0's *initialized-but-arbitrary* scores propagates (that's the lever doing its job at step 0; it's not a wiring bug). The honest identity assertion is therefore scoped to **layer 0 only** (or, equivalently, to flag-off ≡ flag-on-with-prev_scores=None at the layer-0 site). I encoded the latter in (e). The original Mechanism paragraph keeps the paper's "zero-init cache ⇒ step 0 baseline" phrasing for fidelity but I scoped the precise testable claim in the LoC (e) entry and in the Mechanism paragraph itself ("layer 0 only").
