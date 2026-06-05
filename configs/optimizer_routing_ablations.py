from dataclasses import dataclass

from configs.llm_config import Tiny1M3MConfig


@dataclass
class Tiny1M3MMuonFor1DNormConfig(Tiny1M3MConfig):
    """R1 — MuonFor1DNorm: route 1-D `norm.weight` to Muon instead of AdamW.

    Step-0 == base: the `muon_for_1d_norm=False` default on the parent
    LLMConfig means the routing is byte-identical to the baseline rule.
    With the flag on, the per-channel norm gain gets orthogonalized updates
    via Muon instead of AdamW — orthogonalize the gain vector.

    See docs/research/optimizer_routing/plan.md.
    """
    muon_for_1d_norm: bool = True


@dataclass
class Tiny1M3MMuonForEmbedConfig(Tiny1M3MConfig):
    """R2 — MuonForEmbed: route `token_embedding` to Muon instead of AdamW.

    The embedding is ~91% of params at vocab=50k. Step-0 == base: the
    `muon_for_embed=False` default on the parent LLMConfig means the
    routing is byte-identical to the baseline rule. With the flag on, the
    token-embedding table (and `emb_proj` if present) gets orthogonalized
    updates via Muon instead of AdamW.

    See docs/research/optimizer_routing/plan.md (Batch 1).
    """
    muon_for_embed: bool = True


@dataclass
class Tiny1M3MMuonForOutputConfig(Tiny1M3MConfig):
    """R3 — MuonForOutput: route the attention output projection (`out_proj`) to Muon.

    Step-0 == base. See docs/research/optimizer_routing/plan.md (Batch 1)."""
    muon_for_output: bool = True
