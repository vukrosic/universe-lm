---
id: 158-gau
status: needs-plan
round: 1
updated: 2026-06-14T05:43:53Z
transfer-risk: low
plain: Fuse the attention block and the feed-forward block into a single shared gated unit so the model has fewer parameters and a single information-mixing operation per layer.
---

# 158 — Gated Attention Unit (GAU)

## Source
Hua et al. "Transformer Quality in Linear Time" (Google, 2022) — introduces the Gated Attention Unit (GAU) which combines attention and FFN into one block. arXiv:2202.10447.

## Mechanism
Replace the standard `Attention → Add → FFN → Add` block with a single GAU block:
```
y = x + U_g * x  (gating from input)
z = softmax(Q_g y · K_g y^T) V_g y  (attention)
out = U_o (z * V_o y)  (output proj with gating)
```
The GAU merges FFN gating and attention gating into one operation. With `U_g` init to zeros, the gating is identity at step 0. Compared to standard transformer: same FLOPs, fewer parameters (no separate FFN matmuls), tighter information flow.

## Design sketch
- **File**: `models/layers.py` — replace `TransformerBlock` (or add a new `GAUBlock`) that fuses `Attention + FFN` into one block with shared projections.
- **Config flag**: `use_gau: bool` (default False); when True, swap `TransformerBlock` for `GAUBlock`.
- **Step-0 identity**: the GAU paper carefully initializes all gating projections (`U_g`, `V_o`) to zero at init so the block is identity at step 0. This requires the implementer to *replace* the existing block, not add to it, to avoid double-counting. The byte-identity check at step 0 must verify that `x → x` for the full block (residual-only, all gates zero).
- **Intuition**: GAU's paper claim is that FFN-style gating and attention-style mixing can be fused without losing quality, saving ~30-50% of FFN parameters. At 0.94M, the parameter savings should be re-invested in attention dim (tested implicitly by the GAU paper at T5 scale).

## Scale evidence
Hua et al. tested at T5 scale (250M-13B); GAU is in Google's research codebase. Transfer risk is **low** (≥100M source scale, multiple production-grade ablations).

## Why it's worth a slot
A win would tell us the *separation* between attention and FFN is the binding bottleneck at 0.94M (vs all the residual/normalization variants that are closed); a null would close the fused-block axis at our tier.

## Plan

**Files touched:**
- `models/layers.py` — new `GAUBlock` class (~190 LoC) appended after `TransformerBlock`.
- `models/llm.py` — import `GAUBlock`; add `self.use_gau = getattr(config, "use_gau", False)`; conditionally build `self.gau_blocks` and skip the standard `self.transformer_blocks` build when `use_gau=True`; add a dispatch branch in the `_run_post_embed` forward loop (`block = self.gau_blocks[i // tie_layer_groups]`); add a `block.attention._v_residual` skip guard for the GAU path; add `use_hyper_connections + use_gau` mutual-exclusion assert.
- `configs/llm_config.py` — new `Tiny1M3MGAUConfig(Tiny1M3MConfig)` dataclass with `use_gau: bool = True`.
- `training/trainer.py` — guard the A10 entropy-reg `_collect_entropy_reg` helper so it iterates `m.transformer_blocks or ()` instead of `m.transformer_blocks` directly. With `use_gau=True`, `m.transformer_blocks` is `None` (hard swap), so the original `for ... in m.transformer_blocks:` raised `TypeError: 'NoneType' object is not iterable` on the very first train step. The `or ()` short-circuits to an empty tuple for the GAU path (correct: GAU has no MHA sub-block to stash `_entropy_reg_loss`); with the flag off, `m.transformer_blocks` is a truthy `nn.ModuleList` and the loop runs unchanged ⇒ baseline path bit-identical.

**Re-code round (rc=1, 2026-06-14):** the GPU run failed with `TypeError: 'NoneType' object is not iterable in _collect_entropy_reg(model)`. Reproduced locally (`/Users/vukrosic/miniconda3/bin/python`, `MinimalLLM(Tiny1M3MGAUConfig())` builds to 640,320 params; bare `for block in m.transformer_blocks` raises the exact reported error). Fix is the 1-line guard above; no other files needed changes. Baseline path verified bit-identical (`use_gau=False` ⇒ `m.transformer_blocks` is a 12-block `nn.ModuleList`, the loop runs as before).

**Config flag:** `use_gau: bool = False` (default). When `True`, the model stack is `GAUBlock × n_unique` instead of `TransformerBlock × n_unique`.

**Step-0 identity (the spec pin):** GAU's two gate projections are zero-init slices of the fused Q/K/V/U_g/V_o parameter, so `U_g = 0 ⇒ y = x` and `V_o = 0 ⇒ V_o · z = 0 ⇒ U_o(0) = 0 ⇒ block(x) = x`. Verified empirically: at step 0, `GAUModel(x)` ≡ `norm(emb(x)) @ emb_table.T` to fp32 zero.

**Param cost at tiny1m3m (12 layers, d_model=64, n_heads=4, n_kv_heads=2, d_ff=256):**
- Baseline `TransformerBlock` stack: **949,056** params (~0.94M).
- `GAUBlock` stack: **640,320** params (~0.64M, **-32%**).
- Per-block savings: ~29K (FFN's `2·d_model·d_ff = 32,768` is the main cut).

**Mutual-exclusions** (asserted at construction):
- `use_yoco=True` (YOCO needs a standard MHA + shared KV; GAU fuses attention+FFN).
- `use_hyper_connections=True` (mHC wrapper assumes standard block internals).

**Run command** (on the Vast V100 box):
```bash
# Stage 1 — sync code (local commit → box pull)
git add -A && git commit -m "158: implement Gated Attention Unit (Hua et al. 2022) — fused Attention+FFN block"
git push origin main
ssh BOX 'cd /root/universe-lm && git pull'

# Stage 2 — build smoke + run treatment
ssh BOX 'cd /root/universe-lm && /venv/main/bin/python -c "
import torch; torch.manual_seed(42)
from configs.llm_config import Tiny1M3MGAUConfig
from models.llm import MinimalLLM
cfg = Tiny1M3MGAUConfig()
m = MinimalLLM(cfg)
print(\"GAU build OK, params:\", sum(p.numel() for p in m.parameters()))
x = torch.randint(0, cfg.vocab_size, (2, 32))
m.eval()
with torch.no_grad(): y = m(x)
print(\"forward OK, out shape:\", y.shape)
"'

# Stage 3 — full training run (criterion: final val loss at step 700)
ssh BOX 'cd /root/universe-lm && TORCHDYNAMO_DISABLE=1 /venv/main/bin/python /root/universe-lm/_arq_158-gau.py'
# Then: baseline.sh verdict → write evidence.md → flip.sh 158-gau done/null/drift
```

**Local runner script:** `_arq_158-gau.py` should subclass `Tiny1M3MConfig` with `use_gau: bool = True` and pass `--config_class __main__.C --seed 42` per the standard A/B convention.

**Verdict read:** the runner records the milestone val losses; final loss at step 700 is the headline. Compare against the box baseline cache (`autoresearch/baseline-cache.json`) via `baseline.sh verdict` — same WIN/NULL/DRIFT band conventions as every other idea (|Δ| ≤ 0.01 ⇒ NULL; > +0.01 ⇒ DRIFT; ≤ −0.01 ⇒ PASS).

**Predictions:** most likely a **DRIFT/null** outcome at 0.94M. GAU's design point is parameter efficiency at low budgets — the per-block compute reduction (-32% params, ~-25% flops) means the GAU model is undertrained relative to baseline at the same token budget, and the parameter savings can't be re-spent on attention dim in this 1-idea A/B. A PASS would be a real signal that the attention/FFN separation is the binding bottleneck at 0.94M (a stronger claim than all the closed levers support).

**Why the architecture-level test is still worth running:** every FFN-side lever at this tier closed null (MoE 117/118/145/146, FFN-internal conv 157, FFN-side mods 130/142). The closed set doesn't test *whether the FFN should exist at all*; this idea does. A PASS / NULL / DRIFT here closes the "fused attention+FFN" axis alongside the "FFN-internal mods" axis.
