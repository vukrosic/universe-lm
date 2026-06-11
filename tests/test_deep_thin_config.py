"""Sanity check for the 028 deep-thin config — depth/width swap at fixed
~0.94M budget. Builds `MinimalLLM(Tiny1M3MDeepThinConfig())` and asserts
the parameter count lands within ±5% of the baseline budget. See
`autoresearch/ideas/028-deep-thin-config/plan.md`.

Invariants checked:
  1. Baseline `Tiny1M3MConfig` is unchanged by the diff (no shared
     mutable state between the two dataclasses — the trt subclass is
     purely additive). Expected ≈ 939k params, within 920-960k window.
  2. `Tiny1M3MDeepThinConfig` builds and lands ≤ 990k params (the
     +5% ceiling over the 0.94M budget). Expected ≈ 951k (+1.3%).
  3. The 5 architectural fields are actually overridden (catches the
     case where the dataclass inheritance silently picks up the
     baseline values — e.g. if a future edit deletes the override
     and falls back to `Tiny1M3MConfig`'s d_model=64).
"""
import torch

from configs.llm_config import Tiny1M3MConfig, Tiny1M3MDeepThinConfig
from models.llm import MinimalLLM


def _param_count(cfg) -> int:
    """Build a model with the given config and return total param count."""
    torch.manual_seed(42)
    return sum(p.numel() for p in MinimalLLM(cfg).parameters())


def test_baseline_unchanged_by_diff():
    """The 028 additive diff doesn't perturb `Tiny1M3MConfig` itself.
    Window 920-960k allows a legitimate later embedding tweak without
    breaking this test, but catches silent baseline drift."""
    n = _param_count(Tiny1M3MConfig())
    assert 920_000 <= n <= 960_000, (
        f"baseline param count {n} drifted outside the 920-960k window"
    )


def test_deep_thin_lands_in_budget():
    """`Tiny1M3MDeepThinConfig` builds with ≤ 990k params (≈ +5% over
    the 0.94M baseline). Expected ≈ 951k (+1.3%)."""
    n = _param_count(Tiny1M3MDeepThinConfig())
    assert n <= 990_000, f"deep-thin param count {n} exceeds 990k ceiling"
    # Floor as well — catches the case where the override silently
    # drops fields and falls back to a wildly different model.
    assert n >= 900_000, (
        f"deep-thin param count {n} below 900k floor — likely missing "
        f"layer overrides"
    )


def test_arch_fields_overridden():
    """The 5 architectural fields are actually different from baseline."""
    ctrl = Tiny1M3MConfig()
    trt = Tiny1M3MDeepThinConfig()
    assert trt.d_model == 48 and ctrl.d_model == 64, (
        f"d_model override missing: trt={trt.d_model}, ctrl={ctrl.d_model}"
    )
    assert trt.n_heads == 3 and ctrl.n_heads == 4, (
        f"n_heads override missing: trt={trt.n_heads}, ctrl={ctrl.n_heads}"
    )
    assert trt.n_kv_heads == 3 and ctrl.n_kv_heads == 2, (
        f"n_kv_heads override missing: trt={trt.n_kv_heads}, "
        f"ctrl={ctrl.n_kv_heads}"
    )
    assert trt.n_layers == 20 and ctrl.n_layers == 12, (
        f"n_layers override missing: trt={trt.n_layers}, "
        f"ctrl={ctrl.n_layers}"
    )
    assert trt.d_ff == 192 and ctrl.d_ff == 256, (
        f"d_ff override missing: trt={trt.d_ff}, ctrl={ctrl.d_ff}"
    )
    # Non-architectural fields must be inherited unchanged — assert a
    # representative few so a later "rescue" LR bump or batch-size
    # change is caught by this test, not just the runner.
    assert trt.muon_lr == ctrl.muon_lr, "muon_lr drifted on trt"
    assert trt.adamw_lr == ctrl.adamw_lr, "adamw_lr drifted on trt"
    assert trt.batch_size == ctrl.batch_size, "batch_size drifted on trt"
    assert trt.train_tokens == ctrl.train_tokens, "train_tokens drifted on trt"
    assert trt.warmup_ratio == ctrl.warmup_ratio, "warmup_ratio drifted on trt"
    assert trt.schedule_type == ctrl.schedule_type, "schedule_type drifted on trt"
    assert trt.emb_rank == ctrl.emb_rank, "emb_rank drifted on trt"
    assert trt.ffn_variant == ctrl.ffn_variant, "ffn_variant drifted on trt"
    assert trt.seed == ctrl.seed, "seed drifted on trt"
