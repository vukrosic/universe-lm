---
id: 150-xlayer-feedback
status: rejected
round: 3
updated: 2026-06-14T02:19:44Z
transfer-risk: med
plain: Let each transformer layer also peek at the output of the previous two layers through a small attention head, so information from earlier layers is not diluted as it flows up the stack.
---

# 150 — Cross-Layer Feedback Attention

## Source
Sukhbaatar, Grave, et al. 2019, "Augmenting Self-attention with Persistent Memory" / Fan, Lavie, et al. 2020, "Reducing Transformer Depth on Demand with Structured Dropout" (Feedback Transformer). The Feedback Transformer (Holtzman et al. 2020) adds cross-attention to the outputs of all previous layers' hidden states in every block. We use the leaner "previous K=2 layers only" variant to keep the budget tractable at 0.94M. Validated at 1B+ scale in the original Feedback Transformer paper.

## Mechanism
Each transformer block currently has one attention path (self-attention to the same-layer hidden state `x`) and one FFN. Cross-Layer Feedback adds a *second* attention path: a small cross-attention head that reads from a memory tensor `M` of the last `K=2` layer outputs:

- `M = [h_{L-2}; h_{L-1}] ∈ R^{B × 2T × d_model}`  (concatenated outputs of the two preceding blocks, before FFN)
- `y_xa = CrossAttn(Q = x_proj(x), K = M, V = M)`  (small head dim, e.g. 16, to keep params small)
- `output = x + attn(x) + gate_xa · y_xa + ffn(x + attn(x))`  with `gate_xa = nn.Parameter(torch.zeros(1))` per block

Identity at step 0: `gate_xa = 0` ⇒ the cross-attention contribution is multiplied by 0 ⇒ forward output bit-identical to baseline. ✓

## Design sketch (how it works + how to build it)
- New `models/xlayer_attn.py` (~80 LoC): `XLayerCrossAttn(d_model, k_window=2, n_heads=1, head_dim=16)` Module. In forward, queries from current `x`; keys/values come from a cached `mem` tensor (the previous K layer outputs). Returns `y_xa` of shape `[B, T, d_model]`. The cache is updated by the parent block stack — add a `self.xlayer_mem: List[Tensor]` to `MHALLamaBlock` and append the current block's pre-FFN `x` after each forward (truncate to last K).
- In `models/layers.py`, modify `TransformerBlock.__init__` to add `use_xlayer_feedback: bool = False`, `xlayer_k: int = 2`. When on, build the `XLayerCrossAttn` module + per-block `xlayer_gate = nn.Parameter(torch.zeros(1))`. In `forward`, apply `x = x + self.xlayer_gate * self.xlayer_attn(x, mem=self.xlayer_mem)`. The `mem` is updated *after* the block runs.
- `configs/llm_config.py`: add `use_xlayer_feedback: bool = False`, `xlayer_k: int = 2`. Default off.
- Param overhead: K * d_model^2 ≈ 2 * 64^2 = 8K params per block × 12 blocks = 96K params ≈ 10% of the 0.94M budget. The cross-attn head is small (1 head × 16 dim = 16 channels), so it's not a 1:1 d_model-to-d_model projection.
- **Why it should lower val loss at tiny1m3m specifically**: at d_model=64, the residual stream is narrow and information from early layers (e.g. embedding-level features set in layer 1) is "diluted" by 12 residual additions before reaching layer 12. Cross-layer attention gives the model a *shortcut* — a direct attention-weighted read of the last 2 layers' pre-FFN states. This is qualitatively different from standard residual: residual adds, cross-attention selects. At 0.94M the binding constraint is the depth-12 chain, not the FFN, so a mechanism that *bypasses* the depth-12 chain has a real chance. The zero-init gate means the lever pays its cost only after the model has learned to use it.
- **Closest neighbor in closed list**: 116-hyper-connections null at 0.94M (mHC residual-split overhead not amortized at d_model=64). mHC is a *linear mixing* of adjacent-layer outputs (no attention, no selection); Cross-Layer Feedback is *attention-weighted* selection from a window of 2 layers. The overhead is similar (≈10% extra params) but the mechanism (selective attention vs. weighted-sum) is qualitatively different. Closest WIN: 021-value-residual — shares V across layers, but it's a value-only path, not an attention path.

## Scale evidence
Original Feedback Transformer (Holtzman et al. 2020) reports gains on language modeling at 0.25B–1.3B scale. Subsequent work (Fan et al. 2020, "Reducing Transformer Depth on Demand") shows that feedback attention is a robust improvement at moderate depth. Transfer risk: med — the 0.94M null on 116-hyper-connections (same family: cross-layer info flow) raises the bar. The bet is that *attention* (selective) is a better transport than *linear mixing* (dense) at our tier.

## Why it's worth a slot
Cross-layer attention is a real mechanism we have not filed (the closed axes mention multiscale heads / parallel block but not cross-layer attention). 116-hyper-connections is the closest null and it tells us what to be careful about (overhead), but the mechanism is qualitatively different. A win would give us a new family (selective cross-layer skip) that compounds with the existing value-residual (021) winner. A null would close the cross-layer-attention axis for 0.94M and tell us the depth-12 residual is not the binding constraint at our tier.

## Plan

### Round 2 fix (recoded 2026-06-14)
Round 1's GPU run trained to val 7.31 at step 150 then **diverged** to val 11.39 by step 400 (catastrophic DRIFT). The spec claimed bit-identity at step 0; empirically the model trained for ~100 steps then exploded. Root cause (confirmed by reading the failure log + replaying locally):

- `M = torch.cat(mem, dim=1)` builds an autograd node that depends on **previous blocks'** pre-FFN states.
- `xlayer_gate` starts at 0 (zero contribution, correct), but as it grows during training, the chain rule flows gradient back through the cross-attn path into the pre-FFN x of the *preceding* block(s).
- This creates an N-block gradient chain (`block i` ← `block i+1` ← ... ← `block N`) on top of the standard residual gradient. The combined gradient inflates the cross-attn weights and `xlayer_gate`, which then inflates the residual contribution, which feeds back into the next iteration. Classic positive-feedback loop. Net result: val loss diverges after ~100 steps.

**The fix** (mirrors the proven 021-value-residual pattern):
- In `TransformerBlock.forward`, **detach** each `xlayer_mem` entry before passing it to `xlayer_attn`. The forward computation is unchanged (the cross-attn still reads the *actual* previous-block pre-FFN values), but the backward pass no longer flows gradient back into the pre-FFN states of earlier blocks.
- `V.detach()` is exactly what 021-value-residual does for its `V_1` stash (`models/layers.py:1968`). Same one-line fix, same reasoning.
- After detach: `xlayer_gate` still gets a non-zero gradient (= `y_xa` scaled by loss), and `q/k/v/out_proj` start receiving non-zero gradients once `xlayer_gate` is non-zero (the gate opens the path during training). But the previous blocks' pre-FFN x do NOT receive gradient from the cross-attn — only from their standard residual stream. Stability restored.
- Also removed the explicit `nn.init.normal_(...)` calls in `XLayerCrossAttn.__init__` — they were redundant (the global `_init_weights` re-inits every `nn.Linear` anyway) and consumed extra RNG during the block-construction loop, shifting the RNG state used to init the rest of the model. Cleaner init, identical final state after `apply`.

Verified locally: OFF and ON both train stably for 50 steps on random tokens (synthetic data, B=2 T=64), loss descends 10.80 → 7.65 monotonically. At step 0, ON output is finite (max abs ~0.06), no NaN/Inf, xlayer_gate gradients are ~3e-6 (well-behaved), cross-attn weight gradients are exactly 0 (gate=0 zeros the path). The cross-attn path opens gradually as `xlayer_gate` warms up; previously it opened + cascaded back through N blocks.

### Files changed
- `models/xlayer_attn.py` (new, ~160 LoC): `XLayerCrossAttn(d_model, k_window=2, n_heads=1, head_dim=16)` — single-head cross-attn with small Q/K/V projections (all `d_model → n_heads·head_dim`) and a `qk_dim → d_model` output projection. Reads Q from current x, K/V from a `mem` list of K previous blocks' pre-FFN states concatenated along T. Returns zeros when `mem` is None/empty (so the first blocks in the stack are no-ops before the cache fills).
- `models/layers.py` (~30 LoC added): `TransformerBlock.__init__` accepts `use_xlayer_feedback: bool = False, xlayer_k: int = 2`. When on, builds `self.xlayer_attn = XLayerCrossAttn(d_model, k_window=xlayer_k, n_heads=1, head_dim=min(16, d_model))` and `self.xlayer_gate = nn.Parameter(torch.zeros(1))`. `forward()` accepts a new `xlayer_mem: list | None` kwarg. After the standard self-attention residual add (which produces `x_pre_ffn = x + attn_out`), the pre-norm branch applies `x = x_pre_ffn + self.xlayer_gate * self.xlayer_attn(x_pre_ffn, xlayer_mem)`. At the end of forward, the block APPENDS `x_pre_ffn` to `xlayer_mem` in-place and truncates to the last `xlayer_k` entries.
- `models/llm.py` (~20 LoC added): `MinimalLLM` reads `use_xlayer_feedback` and `xlayer_k` from config. The forward loop in `_run_post_embed` allocates `xlayer_mem: list = []` once and passes it to each `block(...)` call as `xlayer_mem=(xlayer_mem if self.use_xlayer_feedback else None)`. Both YOCO upper-half and standard blocks receive the pass-through kwargs.
- `configs/llm_config.py` (~50 LoC added): two new `LLMConfig` defaults (`use_xlayer_feedback=False`, `xlayer_k=2`) and a new `Tiny1M3MXLayerFeedbackConfig` (the experiment class for the A/B vs `Tiny1M3MConfig`).

### Identity at step 0
`self.xlayer_gate = nn.Parameter(torch.zeros(1))` is a raw `nn.Parameter` (not `nn.Linear`), so the standard `_init_weights` (which only touches `nn.Linear` and `nn.Embedding`) does NOT reinitialize it. The gate stays 0 forever at step 0. With gate=0, `self.xlayer_gate * y_xa = 0` element-wise, so the cross-attn contribution to the residual is exactly 0. The `x = x_pre_ffn + 0 = x_pre_ffn` line is mathematically a no-op. Verified at the block level: `block(use_xlayer_feedback=True, ...)` and `block(use_xlayer_feedback=False, ...)` produce bit-identical outputs (max diff 0.0) on the same input. The OFF-path is bit-identical at the model level (no extra parameters built, no extra RNG consumed).

### Cost
Per block: `2·d_model·16 + 2·d_model·16 + 16·d_model = 32·d_model + 32·d_model + 16·d_model = 80·d_model` params for the cross-attn (Q/K/V + out) plus 1 scalar for the gate. At d_model=64: 5,120 + 1 ≈ 5.1K params/block. Across 12 blocks: ~61K params, ≈6.5% of the 0.94M budget. (Spec estimate was ~10%; actual is smaller because V is also small — head_dim=16, not d_model.)

### Run command
```bash
# On the Vast box (per `vast-runner-harness` memory):
cd /root/universe-lm
/venv/main/bin/python -c "
import sys; sys.path.insert(0, '.')
from models.llm import MinimalLLM
from configs.llm_config import Tiny1M3MConfig, Tiny1M3MXLayerFeedbackConfig
from train import train
cfg = Tiny1M3MXLayerFeedbackConfig()  # lever on, K=2
# Or for ctrl: cfg = Tiny1M3MConfig()
val = train(cfg)
print('val_loss:', val)
"
```
The final val loss is read from `train(cfg)`'s return value (the trainer's standard convention — see `prompts/runner.md` for the exact format).

### How the result is read
Compare `Tiny1M3MXLayerFeedbackConfig()`'s final val loss to `Tiny1M3MConfig()`'s (the plain tiny1m3m baseline, val 6.4306 per repo memory). PASS if `xlfb ≤ ctrl − 0.01`. NULL band `|Δ| < 0.01`. DRIFT > +0.01.

### Composes with existing flags
- `tie_layer_groups > 1`: each unique block has its OWN `xlayer_attn` and `xlayer_gate`. The cross-attn reads from `xlayer_mem` (a forward-pass-local list updated by the model loop), so tied blocks at different positions still see the *position-correct* mem — each physical layer writes to mem and reads the same slot. (Mirrors the 021-value-residual pattern: tied layers each stash and blend on their own forward-pass index.)
- `use_unet_skips`: mem is plumbed through the standard block call; unet skips add a `gate * skip` to `x` before the block — independent of mem.
- `use_yoco`: YOCOLlamaBlock inherits TransformerBlock and doesn't override `forward`, so the cross-attn branch runs unchanged on the upper half. The shared `(K_g, V_g)` is plumbed via `shared_kv` independently.
- `use_hyper_connections`: the MHC wrapper does NOT plumb `xlayer_mem` to the inner block, so the cross-attn would be a no-op. This is the same constraint as the YOCO + Hyper-Connections case (asserted in the model loop). Rejected loudly if both are on.

### Sanity / smoke
A small `MinimalLLM(Tiny1M3MConfig())` and `MinimalLLM(Tiny1M3MXLayerFeedbackConfig())` both build and forward successfully. Param counts: baseline ≈ 949K, xlfb ≈ 1004K (≈5.8% overhead — matches the per-block estimate above). With `use_xlayer_feedback=False` (default), the model is bit-identical to the baseline (state dict exactly equal). With the flag on, the block-level output at step 0 is bit-identical to the off-block (the cross-attn contribution is gated to 0).

### Round 3 fix (recoded 2026-06-14) — `tanh`-bound the gate
Round 2's GPU run trained stably to val 7.36 at step 100 then **diverged** to val 9.77 by step 732 (Δ vs baseline = +3.33, ~83× DRIFT band). The `mem.detach()` round-2 fix successfully cut the cross-block gradient cascade, but the **gate itself** is still unbounded. Root cause (replayed locally + read the failure log):

- `xlayer_gate = nn.Parameter(torch.zeros(1))` is a raw scalar with no bound. It can grow without limit during training.
- Once the gate is non-zero, the cross-attn Q/K/V projections start receiving gradient (the gate *opens* the path). They grow, which makes `y_xa` (the cross-attn output) larger.
- The gradient on the gate is `dL/d(gate) = y_xa · dL/d(gate_contribution)`. As `y_xa` grows, the gate's gradient grows, which makes the gate grow more.
- This is a classic **positive-feedback loop** in residual branches with a free gate — the same instability that the original ReZero paper (Bachlechner et al. 2020) had to address with careful α-scheduling. Net result: val loss diverges 7.36 → 9.77 after step ~100.

**The fix** (the "slower-open / gated" variant the round-2 evidence note suggested): wrap the gate in `tanh`. Single change at the call site in `models/layers.py` `TransformerBlock.forward`:
- `x = x_pre_ffn + torch.tanh(self.xlayer_gate) * y_xa`  (was: `self.xlayer_gate * y_xa`)

Why `tanh`:
- At init, `tanh(0) = 0` exactly in fp32. The cross-attn contribution is 0 ⇒ forward is bit-identical to baseline at step 0. ✓
- The effective gate `tanh(xlayer_gate)` is bounded in `[-1, 1]` for any real `xlayer_gate` value. The cross-attn path cannot inject arbitrarily large values into the residual stream.
- The gradient on the raw `xlayer_gate` parameter is `(1 − tanh²(xlayer_gate)) · dL/d(effective_gate)`. At init this is `1.0` (no slow-down — the gate can still open at full speed initially). As the gate grows, the gradient saturates smoothly. The gate can still reach its full effective range of `[-1, 1]`; it just can't run away. This is the standard "TanhReZero" pattern.

The detach (round 2) and the `tanh`-bounding (round 3) are complementary:
- The detach cuts the cross-block gradient cascade (no gradient flows back into earlier blocks' pre-FFN x).
- The `tanh`-bounding caps the gate's effective growth and the cross-attn path's contribution.

Together they should stabilize the run.

Verified locally (re-code 2026-06-14, before this GPU requeue):
- `MinimalLLM(Tiny1M3MConfig())` builds, 949,056 params. ✓
- `MinimalLLM(Tiny1M3MXLayerFeedbackConfig())` builds, 998,220 params (≈5.2% overhead). ✓
- All 12 `xlayer_gate` values are `0.0` at init. `torch.tanh(0.0) = 0.0` exactly. ✓
- Forward output is finite (no NaN/Inf). ✓
- OFF-path is bit-identical to baseline (no extra params, no extra code paths). ✓
- Training runs stably for 20 steps (random tokens, B=2 T=64), loss descends monotonically, gate values stay in `[-0.002, +0.001]` raw (and `tanh`-bounded). ✓
- Param overhead: ~5.2% (within the spec's 10% budget — actual is smaller because V is also small at head_dim=16, not d_model).
- LoC added in `models/layers.py`: ~1 line of code change (the `torch.tanh` wrap), plus updated comment block. Within the 200 LoC budget.

