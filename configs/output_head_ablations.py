"""Output-head ablations — see docs/research/output_head/plan.md.

30 diverse ablations. Reporting rule: leaderboard = plain CE val_loss.
Aux terms (ZLoss, ConfPenalty, LabelSmooth) are train-only; eval stays plain CE.

Knob families (kept small on purpose — diversity comes from cross-family combos):
  - Loss-side aux:    use_z_loss + z_loss_lambda, label_smooth, conf_penalty_beta, use_poly_loss + poly_eps1
  - Logit ops:        use_output_temp, use_vocab_bias, logit_softcap
  - Head structure:   use_untied_head, output_adapter_rank
  - Embed init probe: embedding_scale

Structure: 6 anchors (one per family) + 24 cross-family combos so each
config is testing a different interaction, not the same knob at finer
resolution.
"""
from dataclasses import dataclass
from configs.llm_config import Tiny1M3MConfig


# ---- Anchors — one per family ------------------------------------------------

@dataclass
class Tiny1M3MOutputZLossConfig(Tiny1M3MConfig):
    """A1 anchor — ZLoss λ=1e-4. Loss-side aux baseline."""
    use_z_loss: bool = True
    z_loss_lambda: float = 1e-4


@dataclass
class Tiny1M3MLabelSmoothConfig(Tiny1M3MConfig):
    """A2 anchor — LabelSmoothing ε=0.05."""
    label_smooth: float = 0.05


@dataclass
class Tiny1M3MConfPenaltyConfig(Tiny1M3MConfig):
    """A3 anchor — ConfidencePenalty β=0.01."""
    conf_penalty_beta: float = 0.01


@dataclass
class Tiny1M3MOutputTempConfig(Tiny1M3MConfig):
    """A4 anchor — OutputTemp τ=1, learnable. Logit op baseline."""
    use_output_temp: bool = True


@dataclass
class Tiny1M3MVocabBiasConfig(Tiny1M3MConfig):
    """A5 anchor — VocabBias (per-vocab learnable additive)."""
    use_vocab_bias: bool = True


@dataclass
class Tiny1M3MLogitSoftcapConfig(Tiny1M3MConfig):
    """A6 anchor — Gemma logit softcap=15.0. The value tested at screen10m."""
    logit_softcap: float = 15.0


@dataclass
class Tiny1M3MUntieHeadConfig(Tiny1M3MConfig):
    """A7 anchor — UntieHead. Head structure baseline (NOT budget-matched)."""
    use_untied_head: bool = True


@dataclass
class Tiny1M3MPolyLossConfig(Tiny1M3MConfig):
    """A8 anchor — PolyLoss ε₁=1.0. Loss-side aux baseline.

    Adds the j=1 Taylor correction `ε₁·(1 - p_t)` to CE in the train path
    only. ε₁=1.0 is the paper's "strong default" (Leng et al. 2022,
    arXiv:2204.12511) — the principled next-Taylor-term value, not a tuned
    hyperparameter. Eval stays plain CE. See
    autoresearch/ideas/010-polyloss/plan.md.
    """
    use_poly_loss: bool = True
    poly_eps1: float = 1.0


# ---- Loss-side + logit op (4 combos) ----------------------------------------

@dataclass
class Tiny1M3MZLossOutputTempConfig(Tiny1M3MConfig):
    """C1 — ZLoss + OutputTemp. Does the logit-temp learn the same shape ZLoss enforces?"""
    use_z_loss: bool = True
    z_loss_lambda: float = 1e-4
    use_output_temp: bool = True


@dataclass
class Tiny1M3MZLossVocabBiasConfig(Tiny1M3MConfig):
    """C2 — ZLoss + VocabBias. Logit-collapse-prevention + per-vocab prior."""
    use_z_loss: bool = True
    z_loss_lambda: float = 1e-4
    use_vocab_bias: bool = True


@dataclass
class Tiny1M3MLabelSmoothOutputTempConfig(Tiny1M3MConfig):
    """C3 — LabelSmooth + OutputTemp. Train-side smoothing + logit temp."""
    label_smooth: float = 0.05
    use_output_temp: bool = True


@dataclass
class Tiny1M3MConfPenaltyVocabBiasConfig(Tiny1M3MConfig):
    """C4 — ConfPenalty + VocabBias. Entropy term + learnable per-vocab bias."""
    conf_penalty_beta: float = 0.01
    use_vocab_bias: bool = True


# ---- Logit op + logit op (4 combos) -----------------------------------------

@dataclass
class Tiny1M3MOutputTempVocabBiasConfig(Tiny1M3MConfig):
    """C5 — OutputTemp + VocabBias. Two logit ops stacked."""
    use_output_temp: bool = True
    use_vocab_bias: bool = True


@dataclass
class Tiny1M3MOutputTempLogitSoftcapConfig(Tiny1M3MConfig):
    """C6 — OutputTemp + LogitSoftcap. Does learnable τ + cap conflict?"""
    use_output_temp: bool = True
    logit_softcap: float = 15.0


@dataclass
class Tiny1M3MVocabBiasLogitSoftcapConfig(Tiny1M3MConfig):
    """C7 — VocabBias + LogitSoftcap. Per-vocab prior + symmetric cap."""
    use_vocab_bias: bool = True
    logit_softcap: float = 15.0


@dataclass
class Tiny1M3MOutputTempVocabBiasLogitSoftcapConfig(Tiny1M3MConfig):
    """C8 — OutputTemp + VocabBias + LogitSoftcap. All three logit ops stacked."""
    use_output_temp: bool = True
    use_vocab_bias: bool = True
    logit_softcap: float = 15.0


# ---- Loss-side + logit op + structure (3 combos) ----------------------------

@dataclass
class Tiny1M3MZLossLogitSoftcapUntieHeadConfig(Tiny1M3MConfig):
    """C9 — ZLoss + LogitSoftcap + UntieHead. Aux + cap + separate lm_head."""
    use_z_loss: bool = True
    z_loss_lambda: float = 1e-4
    logit_softcap: float = 15.0
    use_untied_head: bool = True


@dataclass
class Tiny1M3MLabelSmoothOutputTempAdapterConfig(Tiny1M3MConfig):
    """C10 — LabelSmooth + OutputTemp + OutputAdapter(r=16)."""
    label_smooth: float = 0.05
    use_output_temp: bool = True
    output_adapter_rank: int = 16


@dataclass
class Tiny1M3MConfPenaltyVocabBiasAdapterConfig(Tiny1M3MConfig):
    """C11 — ConfPenalty + VocabBias + OutputAdapter(r=16)."""
    conf_penalty_beta: float = 0.01
    use_vocab_bias: bool = True
    output_adapter_rank: int = 16


# ---- Multi-loss combos (3) ---------------------------------------------------

@dataclass
class Tiny1M3MZLossLabelSmoothConfig(Tiny1M3MConfig):
    """C12 — ZLoss + LabelSmooth. Two train-side aux terms."""
    use_z_loss: bool = True
    z_loss_lambda: float = 1e-4
    label_smooth: float = 0.05


@dataclass
class Tiny1M3MZLossConfPenaltyConfig(Tiny1M3MConfig):
    """C13 — ZLoss + ConfPenalty. Anti-collapse + entropy maximization."""
    use_z_loss: bool = True
    z_loss_lambda: float = 1e-4
    conf_penalty_beta: float = 0.01


@dataclass
class Tiny1M3MAllTrainAuxConfig(Tiny1M3MConfig):
    """C14 — ZLoss + LabelSmooth + ConfPenalty. All three train-side aux together."""
    use_z_loss: bool = True
    z_loss_lambda: float = 1e-4
    label_smooth: float = 0.05
    conf_penalty_beta: float = 0.01


# ---- Logit + structure (3) ---------------------------------------------------

@dataclass
class Tiny1M3MOutputAdapterLogitSoftcapConfig(Tiny1M3MConfig):
    """C15 — OutputAdapter(r=16) + LogitSoftcap. Head refit + logit cap."""
    output_adapter_rank: int = 16
    logit_softcap: float = 15.0


@dataclass
class Tiny1M3MUntieHeadLogitSoftcapConfig(Tiny1M3MConfig):
    """C16 — UntieHead + LogitSoftcap. Separate head + symmetric cap."""
    use_untied_head: bool = True
    logit_softcap: float = 15.0


@dataclass
class Tiny1M3MUntieHeadOutputTempConfig(Tiny1M3MConfig):
    """C17 — UntieHead + OutputTemp. Separate head + learnable τ."""
    use_untied_head: bool = True
    use_output_temp: bool = True


# ---- Embedding init probe (3) ------------------------------------------------

@dataclass
class Tiny1M3MEmbeddingScaleHalfConfig(Tiny1M3MConfig):
    """C18 — embedding ×0.5. Embedding scale probe (half the standard)."""
    embedding_scale: float = 0.5


@dataclass
class Tiny1M3MEmbeddingScaleOneConfig(Tiny1M3MConfig):
    """C19 — embedding ×1.0 (no scaling). Tests whether std sqrt(d_model) is load-bearing."""
    embedding_scale: float = 1.0


@dataclass
class Tiny1M3MEmbeddingScaleTwoConfig(Tiny1M3MConfig):
    """C20 — embedding ×2.0. 2x standard scaling."""
    embedding_scale: float = 2.0


# ---- Logit-softcap intensity sweep (3) ---------------------------------------

@dataclass
class Tiny1M3MLogitSoftcap5Config(Tiny1M3MConfig):
    """C21 — softcap=5.0. Aggressive. Tests whether cap tightness matters at this scale."""
    logit_softcap: float = 5.0


@dataclass
class Tiny1M3MLogitSoftcap30Config(Tiny1M3MConfig):
    """C22 — softcap=30.0. The Gemma value. Loosest cap tested."""
    logit_softcap: float = 30.0


@dataclass
class Tiny1M3MLogitSoftcapEmbedScaleConfig(Tiny1M3MConfig):
    """C23 — softcap=15 + embedding ×1.0. Two orthogonal logit-input probes."""
    logit_softcap: float = 15.0
    embedding_scale: float = 1.0


# ---- Cross-domain probes (7) ------------------------------------------------

@dataclass
class Tiny1M3MOutputAdapterUntieHeadConfig(Tiny1M3MConfig):
    """C24 — UntieHead + OutputAdapter(r=16). Max head capacity. NOT budget-matched."""
    use_untied_head: bool = True
    output_adapter_rank: int = 16


@dataclass
class Tiny1M3MZLossUntieHeadConfig(Tiny1M3MConfig):
    """C25 — ZLoss + UntieHead. Aux + head untie."""
    use_z_loss: bool = True
    z_loss_lambda: float = 1e-4
    use_untied_head: bool = True


@dataclass
class Tiny1M3MOutputAdapter32Config(Tiny1M3MConfig):
    """C26 — OutputAdapter(r=32) alone. Larger head refit, no other changes."""
    output_adapter_rank: int = 32


@dataclass
class Tiny1M3MOutputTempEmbedScaleConfig(Tiny1M3MConfig):
    """C27 — OutputTemp + embedding ×1.0. Learnable τ + orthogonal logit input probe."""
    use_output_temp: bool = True
    embedding_scale: float = 1.0


@dataclass
class Tiny1M3MVocabBiasEmbedScaleConfig(Tiny1M3MConfig):
    """C28 — VocabBias + embedding ×1.0. Per-vocab bias + orthogonal logit input probe."""
    use_vocab_bias: bool = True
    embedding_scale: float = 1.0


@dataclass
class Tiny1M3MLabelSmoothLogitSoftcapConfig(Tiny1M3MConfig):
    """C29 — LabelSmooth + LogitSoftcap. Train-side smoothing + symmetric cap."""
    label_smooth: float = 0.05
    logit_softcap: float = 15.0


@dataclass
class Tiny1M3MAllOutputHeadConfig(Tiny1M3MConfig):
    """C30 — every OH knob on at once. Saturate the head. Stress test of interactions."""
    use_z_loss: bool = True
    z_loss_lambda: float = 1e-4
    label_smooth: float = 0.05
    conf_penalty_beta: float = 0.01
    use_output_temp: bool = True
    use_vocab_bias: bool = True
    logit_softcap: float = 15.0
    output_adapter_rank: int = 16
