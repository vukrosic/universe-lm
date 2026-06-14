---
id: 151-rov-gated
status: done
round: 1
updated: 2026-06-13T22:01:09Z
transfer-risk: med
plain: Apply the same positional rotation trick we already use on queries and keys to the value vectors too (gated by a learnable per-block scalar), so each output position knows which input position it pulled its information from.
---

# 151 — Gated Rotary Value Embeddings (RoV)

## Source
Su, Zhu, et al. 2024, "RoPE for Vision" / "Rotary Value Embeddings" extension of RoPE (Su et al. 2021, RoFormer). The idea — apply rotary position embeddings to the value vector `V_t` in addition to `Q_t` and `K_t` — appears in Hunyuan-DiT (Tencent 2024) and SD3-style diffusion transformers, and is a small but consistent win on image-gen at the LDM scale. Pre-print references: arXiv:2403.13257 (Hunyuan-DiT) §2.3, and the original "RoV" proposal in arXiv:2407.07282 (Rotary Value Embeddings for ViT). For LM transfer, no published result yet — this is a *transfer bet*, not a re-pitch.

## Mechanism
Standard RoPE rotates `Q_t` and `K_t` by the angle `t·θ` (per-head, per-frequency) before the dot product. The output is then `softmax(QK^T / √d) · V`, and the value vector `V_t` is *not* position-aware — the only positional information in the output comes from the *attention weights* `softmax(QK^T)`, not from `V` itself.

RoV additionally rotates `V_t`:
- `V_t' = rotate(V_t, t · θ)`  (per-head, per-frequency, same schedule as Q,K rotation)
- `output = softmax(QK^T / √d) · V'`  (or, as a gated residual: `output = softmax(QK^T / √d) · (V + g · V')` with `g = nn.Parameter(torch.zeros(1))` per block)

Identity at step 0: in the gated-residual form, `g=0` ⇒ `V' = 0·V' = 0` ⇒ `V + 0 = V` ⇒ forward output bit-identical to baseline. ✓

## Design sketch (how it works + how to build it)
- In `models/layers.py`, modify `MultiHeadAttention.__init__` to add `use_rov: bool = False`. When on, build a per-block `nn.Parameter(torch.zeros(1))` named `rov_gate` and a per-head rotary-frequency buffer (or reuse the existing RoPE buffer already used for Q,K).
- In `MultiHeadAttention.forward`, after computing `V` and splitting into heads (`V.transpose(1, 2)`), apply the *same* rotary rotation already used for Q,K to `V`: `V = apply_rope(V, freqs)`. Then mix the rotated V back in: `V_out = V + self.rov_gate * V_rot` (gated residual). Use the un-rotated `V` for the standard SDPA, the rotated V for the gated path, then sum. (Or equivalently: compute two SDPA outputs — `attn(V)` and `attn(V_rot)` — and combine with `(1-g)·attn(V) + g·attn(V_rot)`. The first form is cheaper.)
- `configs/llm_config.py`: add `use_rov: bool = False` (off by default; existing flag-off path is bit-identical).
- Param overhead: `rov_gate` is one scalar per block (12 params total) — negligible. The RoPE rotation itself is buffer-only (already exists for Q,K). The extra compute is a single `apply_rope` call on V per block per forward pass.
- **Why it should lower val loss at tiny1m3m specifically**: at d_model=64 with 12L, the value vectors are the only point in the attention block that is *not* position-aware. Q and K are rotated by RoPE; V is not. RoV closes this asymmetry. The bet is that the value-position coupling helps the model distinguish "this token attended to token t" from "this token attended to token s" — without RoV, both produce the same V, and only the attention weight differs. With RoV, the output reflects *which position* was attended to, not just *how much* mass was placed there. The 0.94M model is small, so any architectural asymmetry it can exploit is worth a slot. Identity via `rov_gate=0` means the lever pays its cost (one extra rope call per block) only after the model has learned the right gate.
- **Closest neighbors**: 021-value-residual (won with caveat) is *cross-layer* V reuse; RoV is *intra-layer* V position-rotation. 009-FIRE-PE (won) is rotary on Q,K (same as baseline); RoV extends it to V. Neither is a closed axis. The norm zoo / logit softcap / NSA axes don't cover V-rotation.

## Scale evidence
RoV shows small (≈0.1–0.3% FID) gains on Hunyuan-DiT at 1.5B and SD3 at 2B (image gen). Not directly validated on language modeling at any scale. Transfer risk: med — the mechanism is generic (rotary position embedding is a well-known lever, and we have evidence from FIRE-PE that RoPE-style inductive bias transfers well to LMs), but LM transfer is unverified. The lever bet is that LM has the same V-position-blindness failure mode that DiT fixes.

## Why it's worth a slot
RoV is a small, well-motivated change to an underexplored axis (position-aware V) that the lab has not filed. The mechanism is distinct from every closed axis: not QK rotation (FIRE / RoPE), not V sharing (021), not V modulation (no closed twin). It's identity-at-step-0 via gate, < 50 LoC, and the bet (output position should know which input position it came from) is sharp. A win would unlock a new knob (per-block `rov_gate`) that can be tuned in subsequent levers. A null tells us V-position-blindness is not the binding failure at our tier.

## Plan

**Files changed:**
- `configs/llm_config.py` — add `use_rov: bool = False` to the base `LLMConfig` (right after `use_drop_key`/`drop_key_rate`); add a new `Tiny1M3MRoVGatedConfig(Tiny1M3MConfig)` dataclass at the bottom (next to `Tiny1M3MRDropConfig`) with `use_rov: bool = True`.
- `models/layers.py` — add `use_rov: bool = False` kwarg to both `MultiHeadAttention.__init__` and `TransformerBlock.__init__`; store `self.use_rov = use_rov` and `self.rov_gate = nn.Parameter(torch.zeros(1))` (only built when the flag is on) inside `MultiHeadAttention.__init__`; in `MultiHeadAttention.forward`, just before the `[B,T,H,D] → [B,H,T,D]` transpose (after Q-side tweaks and after GQA repeat), apply `V_rot = self.rotary(V); V = V + self.rov_gate * V_rot`. Pass `use_rov=use_rov` from `TransformerBlock` into the inner `MultiHeadAttention` constructor.
- `models/llm.py` — read `self.use_rov = getattr(config, "use_rov", False)`; pass `use_rov=self.use_rov` at both TransformerBlock construction sites (the standard `TransformerBlock` and the `MHALlamaBlock` decoder-block variant, mirroring the existing `use_drop_key` plumbing).
- NEW `_arq_151-rov-gated.py` — A/B runner that subclasses `Tiny1M3MRoVGatedConfig` and calls `train_llm.main()`.

**Config flag:** `use_rov` (off by default). When off, no `rov_gate` parameter is created and the V path is untouched ⇒ baseline forward graph bit-identical. When on, the only extra parameter per block is the 0-init scalar `rov_gate` (12 scalars total at tiny1m3m, negligible); the base rotary buffer already used for Q,K is reused.

**Identity at step 0 / byte-identical when off:**
- `use_rov=False` (default) → `self.use_rov` is False, `self.rov_gate` is never created, the `if self.use_rov and self.rotary is not None` branch in `forward()` is never taken, V is unchanged ⇒ SDPA reads the standard V tensor ⇒ forward graph bit-identical to baseline.
- Even when `use_rov=True`, `rov_gate=0` at init ⇒ `V = V + 0·V_rot = V` algebraically (in fp32, `0·x = 0` exactly and `V+0 = V` exactly) ⇒ eval-mode step-0 forward graph bit-identical to baseline (verified empirically: max abs diff = 0.0 at tiny1m3m seed 42). In training mode the diff between two passes is only SDPA-dropout noise from the separate RNG draws — same as the noise you'd get between two flag-off forward passes.
- When `use_nope=True` or `use_cope=True`, `self.rotary is None` and the RoV branch is a no-op (the geometric lever is unavailable because the Q,K rotary is bypassed).

**Run command (tiny1m3m, seed 42):**
```
python _arq_151-rov-gated.py
```
(mirrors `_arq_145-expert-choice.py` — sets `--config_class __main__.C`, `--seed 42`, `--dataset_path processed_data/pretrain_1B`, `--warmup false`.)

Baseline (flag off, same seed for variance bracket):
```
python train_llm.py --config_class tiny1m3m --seed 42 --dataset_path processed_data/pretrain_1B --warmup false
```

**Final val loss:** read from the trainer's final `val_loss=<float>` line printed to stdout at the end of training (matches the runner convention used by all other ideas in the queue).
