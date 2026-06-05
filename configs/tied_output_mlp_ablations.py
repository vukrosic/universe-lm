"""Tied output MLP ablation configs (docs/research-plans/tied-output-mlp/plan.md)."""
from dataclasses import dataclass
from configs.llm_config import Tiny1M3MConfig


@dataclass
class Tiny1M3MTiedOutputMLPConfig(Tiny1M3MConfig):
    """B0 — Tied output MLP. Shared Wu/Wd encode+decode, decode zero-init. Step-0 = baseline."""
    use_tied_output_mlp: bool = True


@dataclass
class Tiny1M3MUntiedOutputMLPConfig(Tiny1M3MConfig):
    """B1 — Untied output MLP. Same shape as B0 but separate decode weights.

    Control: isolates whether the *tying* matters. 2x the params of B0.
    Step-0 = baseline via g_decode=0.
    See docs/research-plans/tied-output-mlp/plan.md (Variants)."""
    use_untied_output_mlp: bool = True


@dataclass
class Tiny1M3MTiedLinearOutputMLPConfig(Tiny1M3MConfig):
    """B2 — Tied linear output MLP. NO nonlinearity (B0's activation removed).

    Sanity rung: should fold into the existing linear tied head, so we
    expect ≈ baseline. If B0 ≈ B2, the nonlinearity isn't doing work.
    Step-0 = baseline via g_decode=0.
    See docs/research-plans/tied-output-mlp/plan.md (Variants)."""
    use_tied_linear_output_mlp: bool = True
