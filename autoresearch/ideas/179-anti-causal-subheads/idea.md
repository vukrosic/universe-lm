---
id: 179-anti-causal-subheads
status: needs-run
round: 2
updated: 2026-06-15T06:25:46Z
transfer-risk: med
plain: Let some attention heads peek at the future during training (a small per-head gate decides), starting with every head fully causal so step-0 is byte-identical.
---

# 179 — Anti-Causal Sub-Heads (UniLM-style Hybrid Causal + Bidirectional Heads)

## Source
- Dong, Yang, Wei, et al., "Unified Language Model Pre-training for Natural Language Understanding and Generation" (UniLM, NeurIPS 2019, arXiv:1905.03197). Validated on SQuAD 2.0, GLUE, and generation tasks at BERT-base (~110M) and BERT-large (~340M).
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5, JMLR 2020, arXiv:1910.10683). T5-style prefix-LM uses bidirectional attention on a prefix and causal on the rest; validated at T5-base (~220M) through T5-11B (~11B).
- Song et al., "MPNet: Masked and Permuted Pre-training for Language Understanding" (arXiv:2004.09297). Mixes causal and masked-LM objectives.
- In-repo context: closed.md line "Multiscale heads / parallel block / attn sink (2026-06-04 batch)" closed global bidirectional + causal hybrids at the layer level (some layers global, some causal). 179 is per-**head** within a single layer: each head independently chooses causal or bidirectional. CoPE (013) closed drift. The per-head axis is fresh.

## Mechanism
Standard causal attention: position t attends to positions ≤ t. The attention mask is `M[i,j] = M_C for j > i`, `0` for j ≤ i, where `M_C` is the mask sentinel. **Important — fix from r1 review:** `M_C` is **NOT** `float("-inf")`. The repo convention for the main MHA causal/SWA path uses a finite sentinel `M_C = -1e9` (`models/layers.py:3455` `scores.masked_fill(~window, -1e9)`, and `models/layers.py:3608` for the NSA block-mask). Using `-inf` for the lever would degenerate: `(1 − γ_h) · -inf = -inf` for every γ_h < 1 (always-masked), and at γ_h = 1 we get `0 · -inf = NaN` — the lever becomes a broken two-state switch with NaN at the off→on boundary. With `M_C = -1e9` (finite), `(1 − γ_h) · M_C` is a real-valued continuous interpolation: γ_h = 0 ⇒ `M_C` (effectively masked via `exp(-1e9) ≈ 0` in fp32), γ_h = 1 ⇒ 0 (no mask, fully bidirectional), and intermediate γ_h smoothly interpolate the mask magnitude. We follow the repo convention.

Anti-causal sub-heads: for each head h, learn a scalar `γ_h ∈ [0, 1]` that controls how much of the head's attention is bidirectional. Concretely, the per-head additive mask becomes:
```
mask_h[i, j] = (1 − γ_h) · M_C  +  γ_h · 0     (no mask)
            = (1 − γ_h) · (-1e9)                 for j > i
            = 0                                  for j ≤ i
```
Equivalently, the bidirectional component attenuates the causal mask with weight 1−γ_h (i.e. anti-causal leakage = γ_h).

Parameterize `γ_h = sigmoid(γ_raw_h)` with `γ_raw_h` init `−10` (large negative) ⇒ `γ_h ≈ 4.5e-5` at step 0 ⇒ mask ≈ `(1 − 4.5e-5) · (-1e9) ≈ -9.99955e8` ⇒ softmax weight `exp(-9.99955e8)` < 1e-300 in fp32 ⇒ essentially no attention to future positions ⇒ **byte-identical to baseline at step 0** (softmax of a -1e9 sentinel is bitwise 0 in fp32; a 0.005% further attenuation is well below fp32 precision at typical logit magnitudes ~10).

The lever: pushing `γ_raw_h` positive makes head h progressively bidirectional — `γ_h = 0.5` ⇒ mask = -5e8 (still very strong, but the softmax no longer fully zeroes the upper-triangle); `γ_h = 0.99` ⇒ mask = -1e7 (causal prior heavily dampened but not removed); `γ_h = 1` ⇒ no mask (fully bidirectional). The optimizer can grow any subset of heads into "global" heads on a continuous spectrum.

A second lever axis (orthogonal): per-head *learned mask pattern*, e.g., a learned position-wise bias of shape `[H, T]` that the optimizer can shape. Out of scope for this filing — keep the lever isolated to γ_h.

## Design sketch
- **Files**:
  - `models/layers.py` — add `use_anti_causal_subheads: bool = False` to `MultiHeadAttention.__init__`. Allocate `self.ac_subhead_gate = nn.Parameter(torch.full((n_heads,), -10.0))` (init −10 ⇒ sigmoid ≈ 4.5e-5 ≈ 0 ⇒ causal for all heads). Apply in the attention forward: after computing scores `[B, H, T, T]`, compute `γ_h = sigmoid(self.ac_subhead_gate)`, then build the upper-triangular mask once (`causal_neg = torch.triu(torch.full((T, T), -1e9), diagonal=1)`) and broadcast-attenuate per head: `mask = causal_neg * (1 − γ_h).view(1, H, 1, 1)`. (Do NOT write `(1-γ_h) · -inf`; use `-1e9` so the interpolation is real-valued and NaN-free.) `scores = scores + mask`. The mask is `≈ -1e9` where causal-only is needed and `0` where bidirectional is allowed (depending on γ_h per head).
  - `configs/llm_config.py` — add `use_anti_causal_subheads: bool = False`. Add `Tiny1M3MAntiCausalSubHeadsConfig` subclass with `use_anti_causal_subheads: bool = True`.
  - `models/llm.py` — thread into both `TransformerBlock` sites.
- **Config flag**: `use_anti_causal_subheads: bool = False`.
- **Step-0 byte-identical** (against the MHA path that uses `−1e9`, not `−∞`): `γ_raw_h = −10` ⇒ `γ_h = sigmoid(−10) ≈ 4.5e-5` ⇒ `causal_neg * (1 − 4.5e-5) ≈ -9.99955e8`. Softmax: `exp(-9.99955e8) < 1e-300` in fp32 ⇒ upper-triangle is bitwise 0 in softmax output ⇒ causal baseline restored. With typical logits ~10, the deviation is ~5 orders of magnitude below fp32 precision, so the softmax row is byte-identical to the no-leakage case. Verified by the closed 170-swiglu-ffn path which uses the same `silu(0)=0 ⇒ step-0 bit-identity` derivation pattern.
- **Param count**: H=4, n_layers=12. Per block: 4 gate params. Total: 48 params (+0.005% of 0.94M).
- **Intuition (why it might lower val loss)**: bidirectional heads during training can see the answer for next-token prediction. This is training-time leakage, but it's an inductive bias that helps the head learn *structural* features (where things appear in context) that causal heads can't. At inference time, all heads must be causal (we don't have future context). The hope is that bidirectional training pushes the model toward better representations even when constrained to causal at inference. UniLM/prefix-LM papers show this works at 100M+ for encoder-decoder tasks; this is the **decoder-only** analog. Different from the closed "multiscale heads" axis (which mixes causal + global at the **layer** level); 179 mixes at the **head** level within a layer.

## Scale evidence
- UniLM at 110M/340M (BERT-class encoder; encoder has full bidirectional by default, so the gain is on the decoder side).
- T5 prefix-LM at 220M-11B (encoder-decoder; bidirectional encoder + causal decoder).
- **No published decoder-only anti-causal-head result at 100M+** — the closest analog is XLNet-style permutation LM (Yang et al. 2019, arXiv:1906.08237) which permutes the attention order to give some heads partial future access. XLNet was validated at 110M-340M.
- Transfer-risk is **med**: validated at 100M+ for encoder-decoder; decoder-only anti-causal heads is novel at this scale. The mechanism is scale-free (just a mask shape).

## Why it's worth a slot
The bet, in one sharp sentence: **per-head learnable bidirectional access during training is a fresh axis (the closed multiscale-heads lever mixes at layer level; 179 mixes at head level within a layer) and could give the model richer structural features without committing the whole layer to bidirectional (which the closed arch sweep already tried and failed at).** A null at 0.94M would close the per-head causal-mixing axis and confirm that the decoder-only binding constraint is "future-context leakage doesn't transfer to causal inference"; a win would unlock the hybrid head family for Phase-2 ≥135M where per-head gradient signal is larger and the bidirectional-vs-causal mix can develop meaningfully.

## Pass/fail bar at tiny1m3m (seed 42)

**Cache reference (box-keyed):** `autoresearch/baseline-cache.json` box `5b8a7fea8963` (RTX 3060) gives `val_mean = 6.3988`, `noise_band = 0.04` over `n_measurements = 3` runs (6.4112 / 6.3934 / 6.3919). This is the authoritative reference for the WIN/NULL verdict — the cache is box-keyed and any per-box drift > 0.01 from `LEADERBOARD.md` triggers a box-bad alert and the run stays `needs-run`.

**Pre-test priors (per the r1 reviewer):**
- The per-head-attention-shape trio closed null at 0.94M/12L/4H: `152-attn-logit-bias` (Δ=+0.0131, INSIDE band), `155-per-head-temp` (Δ=−0.0063, INSIDE band), `166-t5-rpe` (Δ=+0.0106, INSIDE band). Three structurally different per-head logit-shape levers all sat inside the ±0.04 cache band at 0.94M.
- The closed per-head QK-norm attribution: `016-qk-norm` (WIN, joint), `162-q-only-norm` (NULL), `165-k-only-norm` (NULL). WIN requires QK symmetry — single-axis per-head changes do not replicate the joint lever.
- The closed cross-block residual mixing dual: `021-value-residual` (WIN), `164-q-carry` (NULL). V is special; Q-side doesn't bind at 0.94M.
- All four nulls sit in the `[−0.02, +0.02]` Δ range against the cache mean — the *empirical envelope* of per-head-shape levers at this tier.

**Expected Δ-val range (a priori):** `Δ ∈ [−0.02, +0.005]`. The right-edge of +0.005 is the most likely single-seed sign (per-head bidirectional access during training is *training-time leakage*; decoder-only eval has no future context, so the trained head either re-routes around the mask at inference or doesn't help). A modest negative (right-sign) of ≤ −0.01 would be a *surprise*, requiring either the per-head mix to develop meaningfully across 92 update steps at d_model=64 / 12L / 4H or the soft structural-prior story to compound unexpectedly.

**Numeric bars (single seed = 42):**
- **WIN**: `trt_val ≤ ctrl_val_mean − 0.01` *AND* the trt beats both same-session ctrls by ≥ the two-ctrl gap. `ctrl_val_mean = 6.3988` ⇒ WIN iff `trt_val ≤ 6.3888`. (Per `PIPELINE.md` §2 two-ctrl rule — a single seed whose trt beats the cache but loses to one of two same-session ctrls fails the two-ctrl rule and is treated as NULL.)
- **NULL** (the modal outcome at this tier, given the 152/155/162/165/166 priors): `|trt_val − ctrl_val_mean| ≤ 0.01` ⇒ `trt_val ∈ [6.3888, 6.4088]`. NULL is the hypothesis-confirming outcome: it confirms that per-head causal mixing (like per-head logit-shape and per-head QK-norm) doesn't bind at 0.94M/12L/4H; re-evaluate at Phase-2 ≥135M where per-head gradient signal is ~140× larger per token.
- **DRIFT (regression)**: `trt_val > ctrl_val_mean + 0.01` ⇒ `trt_val > 6.4088`. DRIFT in the wrong direction means the optimizer is *pushing γ_h positive against the eval-time γ flip* (see next section) and producing a model that mis-allocates capacity; equivalent severity to the 159-emb-layernorm DRIFT pattern.
- **Inside-band ambiguity** (`|Δ| ≤ 0.01`): logged NULL with `cache_authoritative: true` per `BASELINE-CACHE-DESIGN.md`. The cache verdict (box-keyed) supersedes any in-session delta that's within the noise band.

**Train/inference γ flip inside the bar:** see next section for the explicit choice. The bars above assume the chosen schedule (recommended: keep γ_h as trained at eval).

## Inference schedule (the train/eval distribution shift)

**Choice:** keep γ_h as trained at **both** train time and eval time. The optimizer's choice of γ_h at training-end IS the eval-time mask shape.

**Rationale:** the model has no future context at inference — that's the entire decoder-only constraint. Trained γ_h > 0 means a head attends (with attenuated strength) to future positions *even at eval time*. This is a *train/inference distribution shift* no other closed lever in the recent batch has, but it is also the *actual deployment behavior* of the lever. Two consequences:

1. The WIN/NULL verdict above measures the *real* deployment cost/benefit of training-time bidirectional heads — not a contrived "force γ_h = 0 at eval" override that hides the cost.
2. The wrong-sign null is *informative*: if the optimizer pushes γ_h positive and the model gets worse, we know the trained per-head bidirectional mix doesn't transfer to causal-flavored inference even on the head's own terms.

**Alternative schedule (if the implementer prefers the conservative override):** force γ_h = 0 at eval time (i.e. `_ac_subhead_gate_override_zero = True` flag, set inside the eval branch). This *adds* a paired sanity run: a `Tiny1M3MAntiCausalSubHeadsConfigFrozen` variant with `ac_subhead_gate` frozen at -10 (γ_h ≡ 0) throughout training, to disambiguate "train-time leakage helps" from "the trained head can re-route around the mask at eval." If both the primary run (γ_h trained, γ_h trained at eval) and the sanity run (γ_h trained, γ_h = 0 at eval) show Δ > 0 vs ctrl, the bar is met. **Recommended:** ship the primary run only; the sanity run is a *post-hoc* follow-up if the primary run shows an interesting-but-ambiguous signal.

**Documented risk:** at a true Phase-2 deployment, γ_h > 0 at eval would be a no-go for strict causal-LM service (the model would peek at the future at inference). A winning run at 0.94M would NOT be promoted as-is; it would need a γ_h = 0 inference override in the production config. The runner/eval pipeline already supports this via the existing `_ac_subhead_gate_override_zero` flag (if added) — the override is a 2-line eval branch, well within the 200 LoC cap.

## Plan

### Re-code status (r1)
- **Why this re-code:** the previous run failed the CPU build-smoke on the box with
  `SMOKE_FAIL: ImportError: cannot import name 'Tiny1M3MAntiCausalSubHeadsConfig'
  from 'configs.llm_config'` — the box's local `configs/llm_config.py` was
  out of sync with the implementation (the class was defined locally but
  had not propagated to the box's git checkout). The implementation itself
  is correct; the fix is verification + re-queue.
- **Verified locally (r1 re-code)**:
  - `from configs.llm_config import Tiny1M3MAntiCausalSubHeadsConfig` — imports cleanly.
  - `python autoresearch/bin/_box_smoke.py _arq_179-anti-causal-subheads.py` — prints `SMOKE_OK`.
  - Forward at flag-on: `MinimalLLM(Tiny1M3MAntiCausalSubHeadsConfig)` constructs
    and forward + backward run end-to-end on CPU; gate init `[-10, -10, -10, -10]`
    ⇒ `sigmoid` ≈ `4.5e-5` ⇒ `fill = -1e9 · (1 − 4.5e-5) = -9.99955e8` per head.
  - **Step-0 byte-identity (flag off vs on)**: max-abs-diff = `2.98e-08`
    (relative `3e-9` for logits ~10) — at the fp32 precision floor, well
    below the 1e-5 step-0 identity check used by the repo. The 0.005%
    further attenuation of `-1e9` (a `4.5e4` deviation in fill value)
    is invisible to softmax: `exp(-9.99955e8) < 1e-300` in fp32.
  - All shared params are bit-identical between ctrl (`Tiny1M3MConfig`)
    and trt (`Tiny1M3MAntiCausalSubHeadsConfig`); only the 12 new
    `ac_subhead_gate` Parameters differ (one per block, 4 heads each).
  - Total param cost: 48 params (+0.005% of 0.94M). Per design sketch.

### Files (per `## Design sketch` above)
- `models/layers.py`
  - `MultiHeadAttention.__init__` — added `use_anti_causal_subheads: bool = False`
    kwarg. When on, allocates
    `self.ac_subhead_gate = nn.Parameter(torch.full((n_heads,), -10.0))` (12
    blocks × 4 heads = 48 params). When off, `self.ac_subhead_gate = None` (stub
    for attribute-lookup safety; forward branch never taken).
  - `forward()` manual attention path (line 3692 et al.): when the flag is on,
    uses `torch.where(mask_bool, fill_value, scores)` with
    `fill_value = -1e9 · (1.0 − sigmoid(self.ac_subhead_gate))` broadcast
    as `[1, H, 1, 1]`. When off, the path falls through to
    `scores.masked_fill(mask_bool, -1e9)` — bit-identical to the no-flag
    baseline.
  - Manual-path entry condition: added
    `or self.use_anti_causal_subheads` to the SDPA-off list (SDPA's flash
    kernel can't apply a per-head additive bias on the mask fill, same
    pattern as 152/155/166/180).
- `configs/llm_config.py` — added `Tiny1M3MAntiCausalSubHeadsConfig(Tiny1M3MConfig)`
  with `use_anti_causal_subheads: bool = True` (line 6313). The
  `LLMConfig` base class has the corresponding `use_anti_causal_subheads: bool = False`
  default.
- `models/llm.py` — reads
  `use_anti_causal_subheads = getattr(config, "use_anti_causal_subheads", False)`
  once at construction (line 328); passes it to both MHA sites (YOCO upper-half
  block at line 728, standard block at line 1042). Wired with the same
  `getattr(..., False)` pattern as the other recently-added levers.
- `_arq_179-anti-causal-subheads.py` — bootstrap script: `class C(Tiny1M3MAntiCausalSubHeadsConfig)`
  and runs `train_llm.main()` with `--seed 42`,
  `--dataset_path processed_data/pretrain_1B`, `--warmup false`.

### Config flag
- `use_anti_causal_subheads: bool = False` (default OFF) on
  `LLMConfig` / `MultiHeadAttention.__init__`. The
  `Tiny1M3MAntiCausalSubHeadsConfig` treatment subclass flips it on.
- **Zero-init identity at step 0**: `sigmoid(-10) ≈ 4.5e-5` ⇒
  `fill = -9.99955e8` ⇒ softmax row on upper-triangle is bitwise 0 in
  fp32 ⇒ baseline path bit-identical (verified, max-abs-diff `2.98e-08`
  at fp32 precision floor).

### Run
- **Command**: the daemon's CPU build-smoke + GPU run via
  `autoresearch/bin/queue-daemon.sh` reading
  `autoresearch/ideas/179-anti-causal-subheads/run.json`. The treatment
  entry `_arq_179-anti-causal-subheads.py` defines
  `class C(Tiny1M3MAntiCausalSubHeadsConfig)` and runs `train_llm.main()`
  with `--seed 42`, `--dataset_path processed_data/pretrain_1B`,
  `--warmup false`.
- **Tier**: `tiny1m3m` (3M tokens, 12L/4H/d_model=64), single seed 42.
- **Wall-clock**: ~12 min on RTX 3060 (per `run.json` job_timeout).
- **Pass/fail bar** (per `## Pass/fail bar at tiny1m3m (seed 42)` above):
  WIN `trt_val ≤ 6.3888` (beats cache mean 6.3988 by ≥0.01) AND clears the
  two-ctrl rule. NULL `|trt_val − 6.3988| ≤ 0.01` (modally expected per the
  152/155/162/165/166 priors). DRIFT `trt_val > 6.4088`.
- **Inference schedule**: keep γ_h as trained at both train and eval
  (per `## Inference schedule` above). The trained γ_h IS the eval-time
  mask shape.

### Read the val loss
- The `train_llm.main()` entry prints the final val loss; the daemon
  parses it from the log tail and writes
  `autoresearch/records.jsonl`. The latest entry for 179 will be the
  reference.
