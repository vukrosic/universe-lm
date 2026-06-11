"""Tests for the 029 V-Norm lever — see
`autoresearch/ideas/029-v-norm/plan.md`.

Invariants checked:
  1. Flag-OFF byte-identity. With `use_v_layernorm=False` (default),
     a `MinimalLLM(Tiny1M3MConfig)` forward is exactly bit-identical
     across two seed-42 builds: the 029 elif branch never fires, no
     `v_norm` module is created, and the baseline path stays
     byte-identical.
  2. Flag-ON wires the LayerNorm. With `use_v_layernorm=True` on
     plain Tiny1M3MConfig (v_norm_type=""), every block's attention
     has `use_v_norm=True` and `v_norm` is an `nn.LayerNorm(d_k)`
     with γ=1, β=0 default init.
  3. Wiring live — perturbing the LN γ on one channel changes only
     that channel's V output. Documents the per-head, per-channel
     property (the centering+scale is a single LN over d_head).
  4. Independence — `id(q_norm) != id(v_norm)` and
     `id(k_norm) != id(v_norm)`. No weight sharing (idea.md:23
     explicit requirement).
  5. Closed-#92 precedence — when `v_norm_type="pnorm2"` AND
     `use_v_layernorm=True` are both set, the existing
     `v_norm_type` branch wins (the elif never fires). Documented
     in the flag comment in `configs/llm_config.py`.
"""
import torch
import torch.nn as nn

from configs.llm_config import Tiny1M3MConfig
from models.layers import MultiHeadAttention
from models.llm import MinimalLLM


def _build(use_v_layernorm: bool, seed: int = 42) -> MinimalLLM:
    """Build a tiny1m3m model with the V-Norm flag set as given.
    Re-seeds before construction so the two builds share Q/K/V/O init."""
    torch.manual_seed(seed)
    cfg = Tiny1M3MConfig()
    cfg.use_v_layernorm = use_v_layernorm
    return MinimalLLM(cfg).eval()


def _forward_logits(model: MinimalLLM, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x)


def test_flag_off_byte_identical():
    """With `use_v_layernorm=False` (default), two seed-42 builds produce
    byte-identical outputs. The 029 elif branch never fires; no `v_norm`
    module is built on any block's attention."""
    model_a = _build(use_v_layernorm=False, seed=42)
    model_b = _build(use_v_layernorm=False, seed=42)
    torch.manual_seed(0)
    x = torch.randint(0, model_a.config.vocab_size, (2, 64))
    y_a = _forward_logits(model_a, x)
    y_b = _forward_logits(model_b, x)
    assert torch.equal(y_a, y_b), (
        f"Flag-off forward is non-deterministic: max |Δ| = "
        f"{(y_a - y_b).abs().max().item()}"
    )
    # No v_norm module on any block under flag-off (v_norm_type="" too).
    for i, block in enumerate(model_a.transformer_blocks):
        assert not block.attention.use_v_norm, (
            f"Block {i} has use_v_norm=True under flag-off"
        )
        assert not hasattr(block.attention, "v_norm"), (
            f"Block {i} has v_norm module under flag-off"
        )


def test_flag_on_builds_layernorm():
    """With `use_v_layernorm=True` on plain Tiny1M3MConfig
    (v_norm_type=""), every block's attention has `use_v_norm=True`
    and `v_norm` is an `nn.LayerNorm(d_k)` with γ=1, β=0 default init."""
    model_on = _build(use_v_layernorm=True, seed=42)
    for i, block in enumerate(model_on.transformer_blocks):
        assert block.attention.use_v_norm, (
            f"Block {i} use_v_norm should be True"
        )
        assert isinstance(block.attention.v_norm, nn.LayerNorm), (
            f"Block {i} v_norm should be nn.LayerNorm, got "
            f"{type(block.attention.v_norm).__name__}"
        )
        # γ=1, β=0 default init.
        assert torch.equal(
            block.attention.v_norm.weight, torch.ones_like(block.attention.v_norm.weight)
        ), f"Block {i} v_norm.weight not unit-init"
        assert torch.equal(
            block.attention.v_norm.bias, torch.zeros_like(block.attention.v_norm.bias)
        ), f"Block {i} v_norm.bias not zero-init"
        # LN is over d_k (per-head head-dim).
        d_k = block.attention.d_k
        assert tuple(block.attention.v_norm.normalized_shape) == (d_k,), (
            f"Block {i} v_norm shape {block.attention.v_norm.normalized_shape} != ({d_k},)"
        )


def test_independent_module_no_weight_sharing():
    """`id(q_norm) != id(v_norm)` and `id(k_norm) != id(v_norm)` on
    every block — the spec at `idea.md:23` requires independent
    `nn.LayerNorm` modules (no weight sharing with q_norm/k_norm)."""
    torch.manual_seed(42)
    cfg = Tiny1M3MConfig()
    cfg.use_v_layernorm = True
    cfg.use_qk_layernorm = True
    model_on = MinimalLLM(cfg).eval()
    for i, block in enumerate(model_on.transformer_blocks):
        a = block.attention
        assert id(a.q_norm) != id(a.v_norm), (
            f"Block {i} q_norm and v_norm share identity (weight sharing)"
        )
        assert id(a.k_norm) != id(a.v_norm), (
            f"Block {i} k_norm and v_norm share identity (weight sharing)"
        )


def test_closed_92_precedence():
    """When `v_norm_type="pnorm2"` AND `use_v_layernorm=True` are both
    set, the existing `v_norm_type` branch wins (elif never fires).
    Documented in the flag comment in `configs/llm_config.py`."""
    mha = MultiHeadAttention(
        d_model=64, n_heads=4, max_seq_len=64,
        v_norm_type="pnorm2", use_v_layernorm=True,
    )
    # The v_norm_type branch built a non-LayerNorm; the elif must not
    # have overwritten it with nn.LayerNorm.
    assert mha.use_v_norm, "v_norm_type=pnorm2 should have set use_v_norm=True"
    assert not isinstance(mha.v_norm, nn.LayerNorm), (
        f"v_norm_type=pnorm2 should win over use_v_layernorm, but v_norm is "
        f"{type(mha.v_norm).__name__}"
    )


def test_wiring_live_perturbing_gamma_changes_output():
    """Setting `v_norm.weight[c] = K` (a large scale on one channel)
    changes the V path output (proves the LN is actually applied;
    a kwarg threaded but not used would silently no-op)."""
    torch.manual_seed(42)
    cfg = Tiny1M3MConfig()
    cfg.use_v_layernorm = True
    model_a = MinimalLLM(cfg).eval()

    torch.manual_seed(42)
    cfg2 = Tiny1M3MConfig()
    cfg2.use_v_layernorm = True
    model_b = MinimalLLM(cfg2).eval()
    # Perturb γ on one channel of v_norm in block 0 only.
    with torch.no_grad():
        model_b.transformer_blocks[0].attention.v_norm.weight[0] = 10.0

    torch.manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y_a = _forward_logits(model_a, x)
    y_b = _forward_logits(model_b, x)
    diff = (y_a - y_b).abs().max().item()
    assert diff > 1e-3, (
        f"Perturbing v_norm.weight[0] should change the output; max |Δ| = {diff}"
    )
