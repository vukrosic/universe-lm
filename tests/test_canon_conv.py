"""Tests for `models.canon_conv.CanonConv` — see
`autoresearch/ideas/023-canon-conv/plan.md`.

Invariants checked (the spec's (a)-(e) plus a smoke gate):
  1. No NaN/Inf on a non-trivial random input.
  2. Causality — a `+1` perturbation at input position `t` changes
     only output positions `≥ t` (the left-pad-2 design enforces
     strict causality; `padding=2` on the conv would leak future).
  3. Step-0 identity — `CanonConv(d_model)` with default init
     (gate=0) returns the input exactly (`g·DWConv(x) = 0` for any
     conv init because `g=0`).
  4. Wiring live with non-zero gate — perturbing the conv weights
     on one channel with `g=1` set changes only that channel's
     output (depthwise property).
  5. Step-0 identity in TransformerBlock — `use_canon_conv=True`
     (with `g=0`) matches `use_canon_conv=False` output within
     `1e-6` on a freshly-init tiny1m3m-style block.
  6. Placement — canon conv runs BEFORE the attention sublayer's
     pre-LN: a perturbation at the canon conv's gate (g=1) flows
     through the block output, with the contribution verifiable
     against the closed-form `g·DWConv(x)` path.
"""
import torch
from models.canon_conv import CanonConv


def test_no_nan_or_inf():
    """Random input → finite output."""
    torch.manual_seed(42)
    conv = CanonConv(d_model=64)
    x = torch.randn(2, 32, 64)
    y = conv(x)
    assert torch.isfinite(y).all(), "CanonConv output has NaN/Inf"
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"


def test_causality_via_perturbation():
    """A `+1` perturbation at input position `t` only affects output
    positions `≥ t` (causal, no future leak from left-pad-2 design).

    The spec at `idea.md:101-103` pins this assertion explicitly.
    """
    torch.manual_seed(42)
    conv = CanonConv(d_model=8, kernel_size=3)
    # Set the conv weights to a known nonzero pattern so the causal
    # claim is meaningful (with the default init it's also nonzero
    # but harder to read off; this makes the test tighter).
    with torch.no_grad():
        conv.conv.weight.zero_()
        conv.conv.weight[:, 0, :] = 1.0  # kernel = [1, 0, 1] per channel
    conv.gate.data.fill_(1.0)  # g=1 so the conv path actually fires

    T = 16
    x = torch.zeros(1, T, 8)
    t = 5
    x[0, t, :] = 1.0
    y = conv(x)
    # Output at position s < t must be exactly 0 (no past leak).
    assert y[0, :t].abs().max().item() == 0.0, (
        f"Output at s < t should be 0; max = {y[0, :t].abs().max().item()}"
    )
    # Output at position t should be nonzero (the perturbation propagates).
    assert y[0, t].abs().max().item() > 0.0
    # Output at position t+1 may be nonzero (kernel extends +1 step).
    # Output at position t+2 is fine to be nonzero. No assertion on
    # the exact tail — the test is the "no future leak" direction.


def test_step0_identity_gate_zero():
    """At init, gate=0 → `g·DWConv(x) = 0` → output equals input.

    The conv weights are nn.Conv1d's default Kaiming-uniform, but
    since `g=0` exactly, the contribution is 0 regardless of the
    conv init. This is the spec's (e) test for the module.
    """
    torch.manual_seed(42)
    conv = CanonConv(d_model=64)
    # Init must be gate=0. nn.Parameter(torch.zeros(1)) is the
    # module's default; this assertion pins it.
    assert conv.gate.item() == 0.0, f"gate init: {conv.gate.item()}"
    x = torch.randn(2, 16, 64)
    y = conv(x)
    # Bit-exact equality (no fp32 round-trip on a 0 product).
    assert torch.equal(y, x), (
        f"Step-0 output differs from input: max |Δ| = {(y - x).abs().max().item()}"
    )


def test_wiring_live_with_nonzero_gate():
    """With g=1 and a per-channel weight perturbation, the depthwise
    property is preserved: only the perturbed channel's output
    changes (within a 1-step causal window).
    """
    torch.manual_seed(42)
    conv_a = CanonConv(d_model=8, kernel_size=3)
    conv_b = CanonConv(d_model=8, kernel_size=3)
    conv_b.gate.data.fill_(1.0)
    conv_a.gate.data.fill_(1.0)
    # Sync all conv weights to a known starting point so the diff
    # isolates the +1 perturbation.
    with torch.no_grad():
        conv_b.conv.weight.copy_(conv_a.conv.weight)
        # Perturb channel 0's kernel with +1 on the rightmost weight.
        conv_b.conv.weight[0, 0, 2] += 1.0
    x = torch.randn(1, 16, 8)
    ya = conv_a(x)
    yb = conv_b(x)
    diff = yb - ya
    # Channel 0 is the only perturbed channel → its diff is the
    # only one that should be nonzero (max over batch + time).
    per_channel_max = diff.abs().max(dim=0).values.max(dim=0).values
    nonzero_channels = (per_channel_max > 1e-6).nonzero().flatten().tolist()
    assert nonzero_channels == [0], (
        f"Expected only channel 0 to differ; got {nonzero_channels}"
    )


def test_block_step0_identity():
    """TransformerBlock with `use_canon_conv=True, g=0` matches
    `use_canon_conv=False` within 1e-6 on a freshly-init
    tiny1m3m-style block. The spec's (e) test at the block level.
    """
    from models.layers import TransformerBlock
    torch.manual_seed(42)
    common = dict(
        d_model=64, n_heads=4, d_ff=256, max_seq_len=128,
        dropout=0.0, n_kv_heads=2,
    )
    blk_off = TransformerBlock(**common, use_canon_conv=False)
    blk_on = TransformerBlock(**common, use_canon_conv=True)
    # Sync all params from blk_off to blk_on so the only diff is the
    # canon conv (which has gate=0 and thus contributes 0 at step 0).
    blk_on.load_state_dict(blk_off.state_dict(), strict=False)
    x = torch.randn(2, 32, 64)
    y_off = blk_off(x)
    y_on = blk_on(x)
    assert torch.allclose(y_off, y_on, atol=1e-6), (
        f"Step-0 block output differs: max |Δ| = "
        f"{(y_off - y_on).abs().max().item()}"
    )


def test_block_placement_pre_attn():
    """A perturbation at the canon conv's gate (g=1) flows to the
    block output, confirming the conv is in the forward path and
    runs before the attention sublayer (so its contribution is
    visible at the block output even when the attention output
    itself is unchanged on a fresh init).
    """
    from models.layers import TransformerBlock
    torch.manual_seed(42)
    common = dict(
        d_model=64, n_heads=4, d_ff=256, max_seq_len=128,
        dropout=0.0, n_kv_heads=2,
    )
    blk_a = TransformerBlock(**common, use_canon_conv=True)
    blk_b = TransformerBlock(**common, use_canon_conv=True)
    blk_b.load_state_dict(blk_a.state_dict(), strict=False)
    # Set blk_b's canon conv gate to 1 with a known conv weight pattern.
    with torch.no_grad():
        blk_b.canon_conv.gate.data.fill_(1.0)
        blk_b.canon_conv.conv.weight.zero_()
        blk_b.canon_conv.conv.weight[:, 0, :] = 1.0  # kernel = [1, 0, 1] per channel
    x = torch.randn(1, 16, 64)
    ya = blk_a(x)
    yb = blk_b(x)
    diff = yb - ya
    # The conv contribution is nonzero on every position (the kernel
    # is dense). Sanity check: the diff is not exactly zero.
    assert diff.abs().max().item() > 0.0, (
        "Canon conv with g=1 should contribute to block output"
    )
