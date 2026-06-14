# Plan — 163 Post-Attention V-Mix Depthwise Convolution

## Flag
- `use_v_mix_conv: bool = False` (default off on `LLMConfig`), `v_mix_conv_kernel: int = 3`
  (odd int ≥ 3; spec pin k=3).
- File: `configs/llm_config.py` (add to `LLMConfig`).
- Subclass: `Tiny1M3MVMixConvConfig(Tiny1M3MConfig)` with `use_v_mix_conv=True,
  v_mix_conv_kernel=3` — `@dataclass`-decorated so the parent's `False` default
  is properly overridden (Python dataclass inheritance ignores subclass
  annotations unless the subclass is `@dataclass`-re-decorated, as
  `_arq_161-dyt-temp.py` documents).

## Change
- `configs/llm_config.py`
  - Add `use_v_mix_conv: bool = False` and `v_mix_conv_kernel: int = 3` to
    `LLMConfig` (next to the analogous `use_conv_ffn` family, lines 262–263).
  - Add `@dataclass class Tiny1M3MVMixConvConfig(Tiny1M3MConfig)` with the two
    flags on. Verify `C().use_v_mix_conv` resolves to `True` in the build-smoke
    (the dataclass-inheritance pitfall documented in `_arq_161-dyt-temp.py`).
- `models/layers.py` (MultiHeadAttention)
  - Add kwargs `use_v_mix_conv: bool = False, v_mix_conv_kernel: int = 3` to
    `MultiHeadAttention.__init__`.
  - Build the depthwise conv weight as a raw `nn.Parameter(zeros(d_model, 1, k))`
    with `weight[:, 0, k//2] = 1.0` set inline (NOT `nn.Conv1d(...)` followed by
    `.data` reassignment — the `nn.Conv1d` construction consumes RNG
    (kaiming_uniform_), which would shift the RNG state for every subsequent
    block's `qkvo_proj` random init and break the step-0 byte-identity claim).
    Same raw-`Parameter` pattern as `models/conv_ffn.py:103-105` and the
    `moa_extra_kv` / `moa_router_weight` constructions in MHA.
  - In `MultiHeadAttention.forward`, **between** the
    `attn_output.transpose(1, 2).reshape(B, T, d_model)` (post-SDPA, post-reshape)
    and the `F.linear(attn_output, self.qkvo_proj[self.qkv_size:])` O projection,
    apply:
    ```python
    if self.use_v_mix_conv:
        h = F.conv1d(
            attn_output.transpose(1, 2),
            self.v_mix_conv_weight,
            bias=None,
            stride=1,
            padding=self.v_mix_conv_kernel // 2,
            groups=self.d_model,
        )
        attn_output = h.transpose(1, 2)
    ```
  - Padding = `k//2` symmetric (causal+future) — the attention sublayer has
    already integrated full causal context, so the conv may look at both
    neighbors. Spec pin k=3.
  - Sits BEFORE `use_output_embed` so the conv output is what `output_embed`
    adds to. Sits AFTER all the post-softmax attention-output family
    (`use_head_gain`, `use_attn_output_gate`, `use_attn_output_channel_gate`,
    `use_gated_attn`, `use_talking_heads_out`, `_apply_output_op`) — those
    multiply through and the conv is then applied to their combined result.
    Composes cleanly because all of them are multiplicative scalars on
    `attn_output`; the conv is a linear op.
- `models/layers.py` (TransformerBlock)
  - Add `use_v_mix_conv: bool = False, v_mix_conv_kernel: int = 3` kwargs to
    `TransformerBlock.__init__` (next to `use_conv_ffn`/`conv_ffn_kernel`).
  - Pass through to `MultiHeadAttention(...)` in the standard block build.
- `models/llm.py`
  - Add `self.use_v_mix_conv = getattr(config, "use_v_mix_conv", False)` and
    `self.v_mix_conv_kernel = max(3, int(getattr(config, "v_mix_conv_kernel", 3)))`.
  - Pass through at both `TransformerBlock(...)` construction sites:
    `models/llm.py:780-990` (standard stack) and `models/llm.py:700-725` (YOCO
    upper-half stack, even though YOCO is mutually exclusive with this idea in
    practice — the upper-half MHA still has the flag plumbed for completeness).

## Control
- **Control**: `Tiny1M3MConfig`, seed 42, tier tiny1m3m, dataset
  `processed_data/pretrain_1B`, `--warmup false`, no flag.
- **Treatment**: `Tiny1M3MVMixConvConfig`, same seed (42), same tier, same
  dataset, `--warmup false`, `use_v_mix_conv=True, v_mix_conv_kernel=3`.
- The daemon owns the baseline (no ctrl shipped in the arq file).

## Step-0 byte-identity (self-check §5, empirical)
Per the reviewer findings: `MinimalLLM(use_v_mix_conv=False)` vs
`MinimalLLM(use_v_mix_conv=True)` at the same seed (42) MUST produce
`max_abs_diff < 1e-6` on a fp32 forward across all 12 blocks. The conv weight
MUST be a raw `nn.Parameter(zeros(d_model, 1, k))` with center-tap = 1.0 set
inline — this is the load-bearing identity claim.

Concretely the build-smoke (mirroring 157's evidence pattern):
```python
import torch
from configs.llm_config import Tiny1M3MConfig, Tiny1M3MVMixConvConfig
from models.llm import MinimalLLM

torch.manual_seed(42)
m_off = MinimalLLM(Tiny1M3MConfig())
torch.manual_seed(42)
m_on  = MinimalLLM(Tiny1M3MVMixConvConfig())

x = torch.randn(2, 32, m_off.config.max_seq_len if hasattr(m_off, "config") else 2048,
                dtype=torch.float32)
# ... or use MinimalLLM's expected input shape from train_llm.
y_off = m_off(x); y_on = m_on(x)
assert (y_off - y_on).abs().max() < 1e-6, f"step-0 byte-identity FAILED: {(y_off - y_on).abs().max()}"
```

## Cost
- 12 layers × `k × d_model` = 12 × 3 × 64 = 2,304 extra params (+0.25% of the
  0.94M model at tiny1m3m). Cheap.
- Forward: one extra depthwise Conv1d per block per step. ~`T × d_model × k`
  multiply-adds = 2048 × 64 × 3 = 393K flops/block/step, negligible vs the
  d_model²·T FFN cost.
- Memory: +1 weight tensor of shape `[d_model, 1, k] = [64, 1, 3]` per block.

## Run
- Command (on the box): `/venv/main/bin/python /root/universe-lm/_arq_163-v-mix-conv.py`.
- Tier: tiny1m3m, seed 42, `--warmup false`, dataset `processed_data/pretrain_1B`.
- Expected wall-clock: 12m (default `job_timeout`; bump only if a single step
  looks heavier than the 157-conv-ffn profile — likely not, this lever is
  cheaper).
- Pass/fail bar (copied from `idea.md`):
  - NULL band `|Δ| ≤ 0.01` (val loss change inside the two-ctrl bracket).
  - DRIFT > +0.01.
  - PASS ≤ −0.01.
  - Both WIN and NULL informative — WIN localizes the locality prior to the
    post-attention V-axis (completing the 3-axis test); NULL closes the
    post-attention locality axis alongside the closed pre-attention (143) and
    post-FFN (157) axes.

## Run artifact
- `_arq_163-v-mix-conv.py` (repo root): imports `Tiny1M3MVMixConvConfig as C`
  directly (mirroring `_arq_161-dyt-temp.py`) — DO NOT re-declare `class
  C(Tiny1M3MConfig)` inline; the dataclass inheritance pitfall would silently
  leave `use_v_mix_conv=False`.
- `autoresearch/ideas/163-v-mix-conv/run.json`:
  ```json
  {"name": "163-v-mix-conv", "arq_file": "_arq_163-v-mix-conv.py", "job_timeout": "12m"}
  ```

## Coordination
- `models/layers.py` and `configs/llm_config.py` are SHARED with the parallel
  Claude sessions (multiple MHA/TransformerBlock kwargs already edited this
  week: 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162). Pre-edit
  `git diff` and `git status` confirmed:
  - Working tree has the expected prior changes (the 152–162 series) — I will
    APPEND my edit to the existing kwargs/blocks without touching their lines.
  - No conflicts in the kwarg slots I need (`use_conv_ffn`/`conv_ffn_kernel`
    adjacent in `LLMConfig`; `use_moa`/`moa_num_experts` adjacent in MHA;
    `use_conv_ffn`/`conv_ffn_kernel` adjacent in TransformerBlock).
- No push. Local working tree only.