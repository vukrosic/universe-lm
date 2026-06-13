import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from configs.llm_config import LLMConfig
from models.layers import TransformerBlock, make_norm
from models.mhc import MultiStreamResidual
from models.yoco import GlobalKVHead, YOCOLlamaBlock


class TiedOutputMLP(nn.Module):
    """B0 Tied output MLP — shared Wu/Wd, autoencoder-tied encode+decode.

    encode:  h0 = x + Wd(act(Wu(x)))        (one FFN, run on the embedding)
    decode:  z  = x + g * Wu^T(act(Wd^T(x))) (same Wu, Wd, transposed)

    `g` (g_decode) is a learnable scalar (ReZero-style, init 0) so the
    decode path is a no-op at step 0 — the model earns the additive
    output path during training. Encode is NOT gated at init, so
    step-0 embeddings differ from the baseline by the standard-init
    `Wu·x` contribution. That's a known B0 design issue; flag it.

    Net new params: 2 × d_model × d_ff (one FFN's worth of Wu + Wd),
    both 2-D so they route to Muon under the existing rule.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Wu: d_model -> d_ff (encode "up" / decode "down" via .t())
        # Wd: d_ff -> d_model (encode "down" / decode "up" via .t())
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wd = nn.Linear(d_ff, d_model, bias=False)
        self.activation = nn.GELU()
        # ReZero scalar gate on the decode path: starts at 0, the optimizer
        # grows it. Without this the decode path would perturb the output
        # representation at step 0 (encode path still does, see class docstring).
        self.g_decode = nn.Parameter(torch.zeros(1))

    def encode(self, x_emb: torch.Tensor) -> torch.Tensor:
        """Run once on the (scaled) embedding, before the block loop."""
        return x_emb + self.Wd(self.activation(self.Wu(x_emb)))

    def decode(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Run once after the final norm, before the tied unembed.

        Uses the transposed Wu/Wd weights via F.linear so the encode and
        decode paths share the same parameter tensors (tied autoencoder).
        """
        # Wd^T projects d_model -> d_ff (decode's "up")
        # Wu^T projects d_ff -> d_model (decode's "down")
        hidden = self.activation(F.linear(x_norm, self.Wd.weight.t()))
        out = F.linear(hidden, self.Wu.weight.t())
        return x_norm + self.g_decode * out


class UntiedOutputMLP(nn.Module):
    """B1 Untied output MLP — separate Wu/Wd for encode and decode.

    Same shape as TiedOutputMLP, but the decode path uses fresh weights
    (not the encode weights' transpose). Doubles the parameter cost vs
    B0 (4 × d_model × d_ff instead of 2 × d_model × d_ff).

    Control: isolates whether the *tying* matters. If B1 ≈ B0, tying is
    free regularization; if B1 > B0, the constraint is a cost.

    `g_decode` is a learnable scalar (ReZero-style, init 0) so the decode
    path is a no-op at step 0. Encode is NOT gated at init (same B0
    caveat): step-0 embeddings differ from the baseline by the
    standard-init `Wu·x` contribution. Flag it.

    Net new params: 4 × d_model × d_ff (two FFNs' worth of Wu + Wd),
    all 2-D so they route to Muon under the existing rule.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Encode path: same shape as B0's Wu/Wd.
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wd = nn.Linear(d_ff, d_model, bias=False)
        # Decode path: separate fresh weights (no transpose, no sharing).
        self.Wu_decode = nn.Linear(d_ff, d_model, bias=False)
        self.Wd_decode = nn.Linear(d_model, d_ff, bias=False)
        self.activation = nn.GELU()
        # ReZero scalar gate on the decode path: starts at 0, the optimizer
        # grows it. Encode path is also ungated at init (B0 caveat).
        self.g_decode = nn.Parameter(torch.zeros(1))

    def encode(self, x_emb: torch.Tensor) -> torch.Tensor:
        """Run once on the (scaled) embedding, before the block loop."""
        return x_emb + self.Wd(self.activation(self.Wu(x_emb)))

    def decode(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Run once after the final norm, before the tied unembed.

        Untied: separate decode weights (no transpose of encode weights).
        """
        hidden = self.activation(self.Wd_decode(x_norm))
        out = self.Wu_decode(hidden)
        return x_norm + self.g_decode * out


class TiedLinearOutputMLP(nn.Module):
    """B2 Tied linear output MLP — B0 with NO nonlinearity.

    encode:  h0 = x + Wd(Wu(x))         (no activation between)
    decode:  z  = x + g * Wu^T(Wd^T(x)) (no activation, same Wu, Wd, transposed)

    Linear cousin of B0. The plan flags B2 as a sanity rung — it should
    fold into the existing linear tied head (which is `x @ token_embedding.T`),
    so we expect ≈ baseline. If B0 ≈ B2, the nonlinearity isn't doing work.

    Same g_decode zero-init as B0. Net new params: 2·d_model·d_ff (one FFN),
    both 2-D so they route to Muon under the existing rule.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Wu: d_model -> d_ff (encode "up" / decode "down" via .t())
        # Wd: d_ff -> d_model (encode "down" / decode "up" via .t())
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wd = nn.Linear(d_ff, d_model, bias=False)
        # No activation — that's the B2 ablation.
        # ReZero scalar gate on the decode path: starts at 0, the optimizer
        # grows it. Decode is a no-op at step 0; encode is NOT gated at init
        # (same B0 caveat).
        self.g_decode = nn.Parameter(torch.zeros(1))

    def encode(self, x_emb: torch.Tensor) -> torch.Tensor:
        """Run once on the (scaled) embedding, before the block loop."""
        return x_emb + self.Wd(self.Wu(x_emb))  # no activation

    def decode(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Run once after the final norm, before the tied unembed.

        Linear (no activation): Wu^T then Wd^T, shared with encode via
        F.linear + weight transpose so both ends point at the same
        parameter tensors (tied autoencoder, but linear).
        """
        # Wd^T projects d_model -> d_ff (decode's "up")
        # Wu^T projects d_ff -> d_model (decode's "down")
        hidden = F.linear(x_norm, self.Wd.weight.t())  # no activation
        out = F.linear(hidden, self.Wu.weight.t())
        return x_norm + self.g_decode * out


class MinimalLLM(nn.Module):
    """Minimal dense LLM"""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        # Token embeddings.
        # emb_rank is None -> full (vocab x d_model) table (default).
        # emb_rank=r -> low-rank factorization: (vocab x r) @ (r x d_model).
        self.emb_rank = getattr(config, "emb_rank", None)
        if self.emb_rank is None:
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.emb_proj = None
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, self.emb_rank)
            self.emb_proj = nn.Linear(self.emb_rank, config.d_model, bias=False)
        self.use_smear_gate = getattr(config, "use_smear_gate", False)
        if self.use_smear_gate:
            self.smear_gate = nn.Parameter(torch.zeros(config.d_model))
        self.position_dropout = nn.Dropout(config.dropout)
        self.use_unet_skips = getattr(config, "use_unet_skips", False)
        if self.use_unet_skips:
            cfg_skip_count = getattr(config, "unet_skip_count", None)
            self.unet_skip_count = (
                cfg_skip_count if cfg_skip_count is not None
                else (config.n_layers // 2)
            )
            if self.unet_skip_count > config.n_layers // 2:
                raise ValueError(
                    f"unet_skip_count={self.unet_skip_count} exceeds "
                    f"n_layers//2={config.n_layers // 2}; bridges would "
                    "read from un-saved early activations"
                )
            self.unet_gate_type = getattr(config, "unet_gate_type", "raw")
            if self.unet_gate_type not in ("raw", "sigmoid"):
                raise ValueError(
                    f"unet_gate_type must be 'raw' or 'sigmoid', got "
                    f"{self.unet_gate_type!r}"
                )
            gate_init = float(getattr(config, "unet_gate_init", 0.0))
            self.unet_skip_gates = nn.Parameter(
                torch.full(
                    (self.unet_skip_count, config.d_model),
                    gate_init,
                )
            )
            self.unet_bridge_norm = getattr(config, "unet_bridge_norm", False)
            if self.unet_bridge_norm:
                self.unet_bridge_norms = nn.ModuleList(
                    [nn.RMSNorm(config.d_model) for _ in range(self.unet_skip_count)]
                )

        # Transformer blocks
        self.use_value_embed = getattr(config, "use_value_embed", False)
        self.use_query_embed = getattr(config, "use_query_embed", False)
        self.use_key_embed = getattr(config, "use_key_embed", False)
        self.use_output_embed = getattr(config, "use_output_embed", False)
        self.use_q_gain = getattr(config, "use_q_gain", False)
        self.use_k_gain = getattr(config, "use_k_gain", False)
        self.use_deep_value_embed = getattr(config, "use_deep_value_embed", False)
        self.use_ffn_embed = getattr(config, "use_ffn_embed", False)
        self.use_qk_norm_post_rope = getattr(config, "use_qk_norm_post_rope", False)
        self.use_sliding_window = getattr(config, "use_sliding_window", False)
        self.sliding_window_size = getattr(config, "sliding_window_size", 512)
        # #53 NoPE: skip the rotary positional embedding entirely. The
        # Q,K tensors still go through RMSNorm (norm is the Q/K
        # magnitude stabilizer, separate concern from position), but
        # the rotary is bypassed.
        self.use_nope = getattr(config, "use_nope", False)
        self.use_fire_pe = getattr(config, "use_fire_pe", False)
        self.fire_pe_d_phi = getattr(config, "fire_pe_d_phi", 4)
        # 013 — CoPE (content-aware positional encoding, replaces RoPE).
        self.use_cope = getattr(config, "use_cope", False)
        # 020 — Forgetting Transformer (per-head learnable forget gate,
        # multiplicative on attention probabilities post-softmax).
        # Conservative extension of softmax attention; default off →
        # baseline path bit-identical. See
        # `autoresearch/ideas/020-forgetting-attn/plan.md`.
        self.use_fox = getattr(config, "use_fox", False)
        # 022 — Softpick (rectified softmax). Drop-in for `torch.softmax`
        # in the manual attention path; default off → baseline path
        # bit-identical. See `autoresearch/ideas/022-softpick-attention/plan.md`.
        self.use_softpick = getattr(config, "use_softpick", False)
        # 024 — Gated Attention (Qiu et al. 2025, arXiv:2505.06708):
        # per-head scalar input-conditional sigmoid gate on `o_h`, post-AV,
        # pre-merge. Default off → baseline path bit-identical. See
        # `autoresearch/ideas/024-gated-attention/plan.md`.
        self.use_gated_attn = getattr(config, "use_gated_attn", False)
        # #107 Exclusive self-attn: subtract the projection of the head
        # output onto the current token's value vector. Default off →
        # baseline path bit-identical.
        self.use_exclusive_self_attn = getattr(config, "use_exclusive_self_attn", False)
        # 109 — KDA channel gate: per-(head, channel) bounded diagonal
        # `2·σ(g)` gain on V before the AV product. KDA's per-channel
        # `Γ = diag(γ_1, …, γ_d)` decay, ported to softmax attention.
        # Default off → baseline path bit-identical (no Parameter
        # created, no application site taken). See
        # `autoresearch/ideas/109-kda-channel-gate/idea.md`.
        self.use_kda_channel_gate = getattr(config, "use_kda_channel_gate", False)
        # 147 — DropKey (Xu et al. 2022, arXiv:2207.01058). Per-head,
        # per-token Bernoulli mask on K during training. Default off →
        # forward graph bit-identical to baseline. See
        # `autoresearch/ideas/147-dropkey/idea.md`.
        self.use_drop_key = getattr(config, "use_drop_key", False)
        self.drop_key_rate = getattr(config, "drop_key_rate", 0.1)
        # 151 — RoV (Rotary Value Embeddings, gated). When on, the
        # block's MHA applies the same rotary to V as to Q,K and
        # mixes via a per-block scalar `rov_gate` (init 0 ⇒
        # bit-identical to baseline at step 0). Default off → baseline
        # path bit-identical. See `autoresearch/ideas/151-rov-gated/idea.md`.
        self.use_rov = getattr(config, "use_rov", False)
        # 025 — Scalable-Softmax (SSMax): per-head learnable scalar
        # s_h that multiplies the attention logits by s_h · log(n)
        # pre-softmax, where n is the per-query causal key count.
        # Init s_h=1.0 (paper default); step-0 non-bit-identical is
        # explicitly justified (the log-scaling IS the mechanism).
        # Default off → baseline path bit-identical. See
        # `autoresearch/ideas/025-scalable-softmax/plan.md`.
        self.use_ssmax = getattr(config, "use_ssmax", False)
        # 023 — Canon conv (gated depthwise causal Conv1d on the residual
        # stream). One conv per block, pre-attn pre-LN, scalar gate init
        # 0 → step-0 ≡ no-conv baseline. Default off → baseline path
        # bit-identical. See `autoresearch/ideas/023-canon-conv/plan.md`.
        self.use_canon_conv = getattr(config, "use_canon_conv", False)
        # 143 — ShortConv (Hyena ShortConv variant, Poli/Massaroli
        # et al. 2023, arXiv:2302.10866): identity-init depthwise
        # causal Conv1d on the residual stream, one conv per block,
        # pre-attn pre-LN, per-block scalar gate init 0 → step-0 ≡
        # no-conv baseline. Default off → baseline path bit-identical.
        # See `autoresearch/ideas/143-shortconv/idea.md`.
        self.use_short_conv = getattr(config, "use_short_conv", False)
        self.short_conv_kernel = int(getattr(config, "short_conv_kernel", 3))
        # 021 — Value Residual Learning (cross-layer V shortcut).
        # Layer 0 stashes its post-W_V/post-GQA/post-transpose V on
        # `block.attention._v_residual`; the forward loop below reads it
        # after the layer-0 call and passes it as `v_residual=V_1` to
        # every layer l > 0. Per-block scalar `lambda_v` (init 0) lives
        # on each MHA; step-0 ≡ baseline. Default off → baseline path
        # bit-identical. See `autoresearch/ideas/021-value-residual/plan.md`.
        self.use_value_residual = getattr(config, "use_value_residual", False)
        # 117 — Soft MoE (Puigcerver et al. 2024): when True, the
        # block's FFN is replaced with `SoftMoEFFN` (E parallel
        # narrower FFNs + softmax dispatch/combine). Default off →
        # baseline path bit-identical (no `SoftMoEFFN` module built).
        # See `models/soft_moe.py` +
        # `autoresearch/ideas/117-soft-moe/idea.md`.
        self.use_soft_moe = getattr(config, "use_soft_moe", False)
        self.soft_moe_n_experts = getattr(config, "soft_moe_n_experts", 4)
        self.soft_moe_n_slots = getattr(config, "soft_moe_n_slots", 4)
        # 145 — Expert-Choice MoE (Zhou et al. 2022): when True, the
        # block's FFN is replaced with `ExpertChoiceMoE` (E parallel
        # full-width FFNs + top-k-per-expert router). Default off →
        # baseline path bit-identical (no `ExpertChoiceMoE` module
        # built). See `models/expert_choice_moe.py` +
        # `autoresearch/ideas/145-expert-choice/idea.md`.
        self.use_expert_choice_moe = getattr(config, "use_expert_choice_moe", False)
        self.n_moe_experts = getattr(config, "n_moe_experts", 4)
        # 150 — Cross-Layer Feedback Attention (Holtzman et al. 2020,
        # Feedback Transformer). When on, each block reads from a
        # cache of the previous K blocks' pre-FFN states via a small
        # `XLayerCrossAttn` head. K=2 by default (the spec pin).
        # Per-block `xlayer_gate=0` init ⇒ step-0 ≡ no-feedback
        # baseline. Default off → baseline path bit-identical (no
        # `XLayerCrossAttn` module built). See
        # `models/xlayer_attn.py` and
        # `autoresearch/ideas/150-xlayer-feedback/idea.md`.
        self.use_xlayer_feedback = getattr(config, "use_xlayer_feedback", False)
        self.xlayer_k = max(1, int(getattr(config, "xlayer_k", 2)))
        # 149 — TTT-Linear (Sun et al. 2024, arXiv:2407.04620). When
        # True, the block's FFN is replaced with `TTTFeedForward`
        # (squared_relu FFN whose up_proj is a `TTTLinear` — per-input
        # closed-form fast-weight update). `ttt_lr_init=0.0` (default)
        # keeps step-0 bit-identical to a vanilla `SquaredReLUFeedForward`.
        # Default off → baseline FFN path bit-identical (no
        # `TTTFeedForward` module built). See
        # `models/ttt_linear.py` + `autoresearch/ideas/149-ttt-linear/idea.md`.
        self.use_ttt_ffn = getattr(config, "use_ttt_ffn", False)
        self.ttt_lr_init = getattr(config, "ttt_lr_init", 0.0)
        # 118 — Mixture-of-Depths (Raposo et al. 2024, arXiv:2404.02258):
        # when on, each block builds a per-token `MoDRouter` and gates
        # the block's residual update to the top-k = `mod_capacity · T`
        # tokens. Default off → baseline path bit-identical. See
        # `models/mod_router.py` +
        # `autoresearch/ideas/118-mixture-of-depths/idea.md`.
        self.use_mod = getattr(config, "use_mod", False)
        self.mod_capacity = getattr(config, "mod_capacity", 0.5)
        self.mod_router_hidden = getattr(config, "mod_router_hidden", 64)
        # 131 — LayerDrop (Fan, Grave, Joulin 2019, arXiv:1904.09728,
        # ICLR 2020). Whole-layer stochastic depth, independent of
        # DropPath (111). Default off → baseline path bit-identical
        # (no gate computed, no rescale). See
        # `autoresearch/ideas/131-layer-drop/idea.md`.
        self.use_layerdrop = getattr(config, "use_layerdrop", False)
        self.layerdrop_p = getattr(config, "layerdrop_p", 0.2)
        self.layerdrop_schedule = getattr(config, "layerdrop_schedule", "constant")
        # 129 — YOCO (Sun et al. 2024, arXiv:2405.05254). When on,
        # split the model into a lower half (layers 0..yoco_split-1,
        # standard TransformerBlock with SWA on) and an upper half
        # (layers yoco_split..n_layers-1, YOCOLlamaBlock with shared
        # KV). The `GlobalKVHead` projects the lower-half final
        # residual stream to `(K_g, V_g)` ONCE per forward, and the
        # upper-half MHA reads them via the `shared_kv` kwarg
        # (skipping the W_K, W_V slices of the merged qkvo_proj).
        # Default off → no GlobalKVHead built, no upper-half
        # ModuleList, baseline forward graph bit-identical.
        # See `models/yoco.py` and
        # `autoresearch/ideas/129-yoco/idea.md`.
        self.use_yoco = getattr(config, "use_yoco", False)
        self.yoco_split = int(getattr(config, "yoco_split", 6))
        self.yoco_lower_window = int(getattr(config, "yoco_lower_window", 512))
        self.rope_base = getattr(config, "rope_base", 10000)
        self.use_tied_qk = getattr(config, "use_tied_qk", False)
        self.use_mla = getattr(config, "use_mla", False)
        self.mla_latent_dim = getattr(config, "mla_latent_dim", None)
        self.attention_dilation = getattr(config, "attention_dilation", 1)
        self.use_post_norm = getattr(config, "use_post_norm", False)
        self.use_layernorm = getattr(config, "use_layernorm", False)
        self.use_linear_attn = getattr(config, "use_linear_attn", False)
        self.use_diff_attn = getattr(config, "use_diff_attn", False)
        self.use_nsa_global = getattr(config, "use_nsa_global", False)
        self.nsa_block = getattr(config, "nsa_block", 64)
        self.use_hybrid_heads = getattr(config, "use_hybrid_heads", False)
        self.norm_type = getattr(config, "norm_type", "rmsnorm")
        self.qk_norm_type = getattr(config, "qk_norm_type", "rmsnorm")
        self.v_norm_type = getattr(config, "v_norm_type", "")
        # #16 QK-Norm (Dehghani et al. 2023, ViT-22B, arXiv:2302.05442):
        # when True, override the Q/K norm from RMSNorm to LayerNorm,
        # bounding the per-head logit. Default off → bit-identical
        # baseline. See autoresearch/ideas/016-qk-norm/plan.md.
        self.use_qk_layernorm = getattr(config, "use_qk_layernorm", False)
        # 029 — V-Norm (Wortsman et al. 2023, arXiv:2309.14322):
        # when True, add a per-head `nn.LayerNorm(d_head)` on V before
        # the AV product, the symmetric partner of 016's QK-Norm. Default
        # off → bit-identical baseline (no v_norm module built). See
        # autoresearch/ideas/029-v-norm/plan.md.
        self.use_v_layernorm = getattr(config, "use_v_layernorm", False)
        self.use_multiscale_heads = getattr(config, "use_multiscale_heads", False)
        self.use_parallel_block = getattr(config, "use_parallel_block", False)
        self.use_attn_sink = getattr(config, "use_attn_sink", False)
        # 017 — Sub-LN / Sandwich block (residual-stream re-bounding).
        self.use_sub_ln = getattr(config, "use_sub_ln", False)
        # Query-tweaks: 29 new flags (see docs/research-plans/query-tweaks/plan.md).
        self.q_norm_type = getattr(config, "q_norm_type", self.qk_norm_type)
        self.use_alibi_bias = getattr(config, "use_alibi_bias", False)
        self.use_q_temp_token = getattr(config, "use_q_temp_token", False)
        self.use_cosine_attn = getattr(config, "use_cosine_attn", False)
        self.use_qk_bilinear = getattr(config, "use_qk_bilinear", False)
        self.use_talking_heads_q = getattr(config, "use_talking_heads_q", False)
        self.use_per_head_rope_base = getattr(config, "use_per_head_rope_base", False)
        self.partial_rotary_p = getattr(config, "partial_rotary_p", 1.0)
        self.use_q_expansion = getattr(config, "use_q_expansion", False)
        self.use_decoupled_content_pos = getattr(config, "use_decoupled_content_pos", False)
        self.use_antisym_qk = getattr(config, "use_antisym_qk", False)
        self.use_q_per_head_bias = getattr(config, "use_q_per_head_bias", False)
        self.use_q_per_channel_gain = getattr(config, "use_q_per_channel_gain", False)
        self.use_q_hd_gain = getattr(config, "use_q_hd_gain", False)
        self.use_q_norm_gate = getattr(config, "use_q_norm_gate", False)
        self.use_q_lowrank_refine = getattr(config, "use_q_lowrank_refine", False)
        self.q_lowrank_refine_rank = getattr(config, "q_lowrank_refine_rank", 8)
        self.use_q_layerscale = getattr(config, "use_q_layerscale", False)
        self.use_q_softplus_gain = getattr(config, "use_q_softplus_gain", False)
        self.use_q_head_mix = getattr(config, "use_q_head_mix", False)
        self.use_q_time_conv = getattr(config, "use_q_time_conv", False)
        self.use_q_ema_smooth = getattr(config, "use_q_ema_smooth", False)
        self.q_ema_alpha = getattr(config, "q_ema_alpha", 0.0)
        self.use_q_feature_map = getattr(config, "use_q_feature_map", False)
        self.q_feature_map_hidden = getattr(config, "q_feature_map_hidden", 64)
        self.use_q_per_token_rope = getattr(config, "use_q_per_token_rope", False)
        self.q_per_token_rope_hidden = getattr(config, "q_per_token_rope_hidden", 32)
        self.use_q_noise_reg = getattr(config, "use_q_noise_reg", False)
        # #55 layer tying (ALBERT-style): when tie_layer_groups=N, every
        # group of N consecutive blocks shares weights. We create only
        # n_layers // N unique blocks and the forward pass cycles through
        # them. U-Net skips are disabled when tying is active (a skip
        # from block 0 to block n_layers-1 would be a cycle).
        self.tie_layer_groups = max(1, getattr(config, "tie_layer_groups", 1))
        if self.tie_layer_groups > 1 and getattr(config, "use_unet_skips", False):
            raise ValueError("tie_layer_groups > 1 is incompatible with use_unet_skips")
        n_unique = config.n_layers // self.tie_layer_groups
        deep_value_embed_hidden = getattr(config, "deep_value_embed_hidden", None)
        value_embed_rank = self.emb_rank if self.emb_rank is not None else config.d_model
        # #86 Interleaved global attention: when global_attn_every_k > 0,
        # every k-th block (1-indexed) drops the sliding window and runs
        # full causal attention — a periodic global layer on top of the
        # otherwise-local stack. Only meaningful when use_sliding_window
        # is on; with it off, every block is already full attention.
        self.global_attn_every_k = max(0, getattr(config, "global_attn_every_k", 0))

        def _block_uses_swa(i: int) -> bool:
            if self.use_yoco:
                # Lower half always uses SWA (default 512) — the
                # upper-half block stack handles its own attention
                # pattern (full causal, sharing the lower half's KV
                # cache).
                return i < self.yoco_split
            if not self.use_sliding_window:
                return False
            if self.global_attn_every_k > 0 and ((i + 1) % self.global_attn_every_k == 0):
                return False  # this is a global (full-attention) layer
            return True

        # 129 — YOCO (Sun et al. 2024, arXiv:2405.05254): when the
        # flag is on, build a SEPARATE upper-half ModuleList of
        # `YOCOLlamaBlock` instances (use_shared_kv=True via the
        # subclass), plus a single `GlobalKVHead` module that
        # projects the lower-half final residual stream to
        # `(K_g, V_g)`. `transformer_blocks` is still built with
        # the full `n_unique` slot count for compatibility with the
        # `tie_layer_groups` cycle in the existing forward loop, but
        # the YOCO branch reads from `yoco_upper_blocks` instead
        # for positions `i >= yoco_split`. To keep the wiring
        # simple we also disable layer tying on the YOCO upper
        # half (each upper layer has its own weights). Default off
        # → no GlobalKVHead, no yoco_upper_blocks, baseline
        # forward graph bit-identical. See
        # `autoresearch/ideas/129-yoco/idea.md`.
        if self.use_yoco:
            if self.yoco_split < 1 or self.yoco_split >= config.n_layers:
                raise ValueError(
                    f"yoco_split={self.yoco_split} must be in "
                    f"[1, n_layers={config.n_layers})"
                )
            self.global_kv_head = GlobalKVHead(
                d_model=config.d_model,
                kv_size=config.n_kv_heads * (config.d_model // config.n_heads),
            )
            # Build the upper-half stack separately so it can be
            # different from the lower half (YOCOLlamaBlock with
            # use_shared_kv=True). For simplicity at tiny1m3m we
            # disable tying on the upper half.
            self.yoco_upper_blocks = nn.ModuleList(
                [
                    YOCOLlamaBlock(
                        config.d_model,
                        config.n_heads,
                        config.d_ff,
                        config.max_seq_len,
                        config.dropout,
                        n_kv_heads=config.n_kv_heads,
                        ffn_variant=config.ffn_variant,
                        use_embed_residual=getattr(config, "use_embed_residual", False),
                        use_attn_output_gate=getattr(config, "use_attn_output_gate", False),
                        use_value_channel_gate=getattr(config, "use_value_channel_gate", False),
                        use_attn_output_channel_gate=getattr(config, "use_attn_output_channel_gate", False),
                        use_exclusive_self_attn=self.use_exclusive_self_attn,
                        use_kda_channel_gate=self.use_kda_channel_gate,
                        # 147 — DropKey: per-head Bernoulli gate on K.
                        use_drop_key=self.use_drop_key,
                        drop_key_rate=self.drop_key_rate,
                        # 151 — RoV (Rotary Value Embeddings, gated):
                        # per-block scalar `rov_gate` mixes the rotary-
                        # rotated V into V via `V ← V + rov_gate·V_rot`.
                        # Init 0 ⇒ bit-identical to baseline at step 0.
                        use_rov=self.use_rov,
                        use_talking_heads_out=getattr(config, "use_talking_heads_out", False),
                        out_op=getattr(config, "out_op", ""),
                        use_re_zero=getattr(config, "use_re_zero", False),
                        resid_mode=getattr(config, "resid_mode", ""),
                        n_layers=config.n_layers,
                        use_layerscale=getattr(config, "use_layerscale", False),
                        use_layer_scale=getattr(config, "use_layer_scale", False),
                        layer_scale_init=getattr(config, "layer_scale_init", 1e-4),
                        use_value_embed=self.use_value_embed,
                        use_query_embed=self.use_query_embed,
                        use_key_embed=self.use_key_embed,
                        use_output_embed=self.use_output_embed,
                        use_q_gain=self.use_q_gain,
                        use_k_gain=self.use_k_gain,
                        use_deep_value_embed=self.use_deep_value_embed,
                        deep_value_embed_hidden=deep_value_embed_hidden,
                        use_ffn_embed=self.use_ffn_embed,
                        use_qk_norm_post_rope=self.use_qk_norm_post_rope,
                        # Upper-half blocks run with sliding window OFF
                        # (the shared K_g, V_g are the global context
                        # source — adding a per-layer SWA would only
                        # mask out useful global signal).
                        use_sliding_window=False,
                        sliding_window_size=self.yoco_lower_window,
                        use_nope=self.use_nope,
                        rope_base=self.rope_base,
                        use_fire_pe=self.use_fire_pe,
                        fire_pe_d_phi=self.fire_pe_d_phi,
                        use_gated_attn=self.use_gated_attn,
                        use_cope=self.use_cope,
                        use_fox=self.use_fox,
                        use_softpick=self.use_softpick,
                        use_ssmax=self.use_ssmax,
                        use_canon_conv=self.use_canon_conv,
                        # 143 — ShortConv pass-through to the YOCO
                        # upper-half block. See
                        # `autoresearch/ideas/143-shortconv/idea.md`.
                        use_short_conv=self.use_short_conv,
                        short_conv_kernel=self.short_conv_kernel,
                        use_value_residual=self.use_value_residual,
                        use_drop_path=getattr(config, "use_drop_path", False),
                        drop_path_max=getattr(config, "drop_path_max", 0.1),
                        use_soft_moe=self.use_soft_moe,
                        soft_moe_n_experts=self.soft_moe_n_experts,
                        soft_moe_n_slots=self.soft_moe_n_slots,
                        # 145 — Expert-Choice MoE pass-through to the
                        # YOCO upper-half block. Default off → FFN path
                        # is bit-identical. See
                        # `autoresearch/ideas/145-expert-choice/idea.md`.
                        use_expert_choice_moe=self.use_expert_choice_moe,
                        n_moe_experts=self.n_moe_experts,
                        # 149 — TTT-Linear pass-through to the YOCO
                        # upper-half block. Default off → FFN path
                        # bit-identical. See
                        # `autoresearch/ideas/149-ttt-linear/idea.md`.
                        use_ttt_ffn=self.use_ttt_ffn,
                        ttt_lr_init=self.ttt_lr_init,
                        use_mod=self.use_mod,
                        mod_capacity=self.mod_capacity,
                        mod_router_hidden=self.mod_router_hidden,
                        # 148 — Focal Modulation pass-through to the
                        # YOCO upper-half block. Default off → MHA
                        # path is bit-identical. See
                        # `autoresearch/ideas/148-focal-mod/idea.md`.
                        use_focal_mod=getattr(config, "use_focal_mod", False),
                        focal_mod_kernels=getattr(config, "focal_mod_kernels", (3, 5, 7)),
                        use_tied_qk=self.use_tied_qk,
                        use_mla=self.use_mla,
                        mla_latent_dim=self.mla_latent_dim,
                        attention_dilation=self.attention_dilation,
                        use_post_norm=self.use_post_norm,
                        use_layernorm=self.use_layernorm,
                        use_linear_attn=self.use_linear_attn,
                        use_diff_attn=self.use_diff_attn,
                        use_nsa_global=self.use_nsa_global,
                        nsa_block=self.nsa_block,
                        use_hybrid_heads=self.use_hybrid_heads,
                        norm_type=self.norm_type,
                        qk_norm_type=self.qk_norm_type,
                        v_norm_type=self.v_norm_type,
                        use_qk_layernorm=self.use_qk_layernorm,
                        use_v_layernorm=self.use_v_layernorm,
                        use_multiscale_heads=self.use_multiscale_heads,
                        use_parallel_block=self.use_parallel_block,
                        use_attn_sink=self.use_attn_sink,
                        use_sub_ln=self.use_sub_ln,
                        q_norm_type=self.q_norm_type,
                        use_alibi_bias=self.use_alibi_bias,
                        use_q_temp_token=self.use_q_temp_token,
                        use_cosine_attn=self.use_cosine_attn,
                        use_qk_bilinear=self.use_qk_bilinear,
                        use_talking_heads_q=self.use_talking_heads_q,
                        use_per_head_rope_base=self.use_per_head_rope_base,
                        partial_rotary_p=self.partial_rotary_p,
                        use_q_expansion=self.use_q_expansion,
                        use_decoupled_content_pos=self.use_decoupled_content_pos,
                        use_antisym_qk=self.use_antisym_qk,
                        use_q_per_head_bias=self.use_q_per_head_bias,
                        use_q_per_channel_gain=self.use_q_per_channel_gain,
                        use_q_hd_gain=self.use_q_hd_gain,
                        use_q_norm_gate=self.use_q_norm_gate,
                        use_q_lowrank_refine=self.use_q_lowrank_refine,
                        q_lowrank_refine_rank=self.q_lowrank_refine_rank,
                        use_q_layerscale=self.use_q_layerscale,
                        use_q_softplus_gain=self.use_q_softplus_gain,
                        use_q_head_mix=self.use_q_head_mix,
                        use_q_time_conv=self.use_q_time_conv,
                        use_q_ema_smooth=self.use_q_ema_smooth,
                        q_ema_alpha=self.q_ema_alpha,
                        use_q_feature_map=self.use_q_feature_map,
                        q_feature_map_hidden=self.q_feature_map_hidden,
                        use_q_per_token_rope=self.use_q_per_token_rope,
                        q_per_token_rope_hidden=self.q_per_token_rope_hidden,
                        use_q_noise_reg=self.use_q_noise_reg,
                        value_embed_rank=value_embed_rank,
                        # 150 — Cross-Layer Feedback pass-through to
                        # the YOCO upper-half block. Default off →
                        # baseline path bit-identical. See
                        # `autoresearch/ideas/150-xlayer-feedback/idea.md`.
                        use_xlayer_feedback=self.use_xlayer_feedback,
                        xlayer_k=self.xlayer_k,
                    )
                    for _ in range(config.n_layers - self.yoco_split)
                ]
            )
        else:
            self.global_kv_head = None
            self.yoco_upper_blocks = None

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.max_seq_len,
                    config.dropout,
                    n_kv_heads=config.n_kv_heads,
                    ffn_variant=config.ffn_variant,
                    use_embed_residual=getattr(config, "use_embed_residual", False),
                    use_attn_output_gate=getattr(config, "use_attn_output_gate", False),
                    use_value_channel_gate=getattr(config, "use_value_channel_gate", False),
                    use_attn_output_channel_gate=getattr(config, "use_attn_output_channel_gate", False),
                    use_exclusive_self_attn=self.use_exclusive_self_attn,
                    use_kda_channel_gate=self.use_kda_channel_gate,
                    # 147 — DropKey: per-head Bernoulli gate on K.
                    use_drop_key=self.use_drop_key,
                    drop_key_rate=self.drop_key_rate,
                    # 151 — RoV (Rotary Value Embeddings, gated):
                    # per-block scalar `rov_gate` mixes the rotary-
                    # rotated V into V via `V ← V + rov_gate·V_rot`.
                    # Init 0 ⇒ bit-identical to baseline at step 0.
                    use_rov=self.use_rov,
                    use_talking_heads_out=getattr(config, "use_talking_heads_out", False),
                    out_op=getattr(config, "out_op", ""),
                    use_re_zero=getattr(config, "use_re_zero", False),
                    resid_mode=getattr(config, "resid_mode", ""),
                    n_layers=config.n_layers,
                    use_layerscale=getattr(config, "use_layerscale", False),
                    use_layer_scale=getattr(config, "use_layer_scale", False),
                    layer_scale_init=getattr(config, "layer_scale_init", 1e-4),
                    use_value_embed=self.use_value_embed,
                    use_query_embed=self.use_query_embed,
                    use_key_embed=self.use_key_embed,
                    use_output_embed=self.use_output_embed,
                    use_q_gain=self.use_q_gain,
                    use_k_gain=self.use_k_gain,
                    use_deep_value_embed=self.use_deep_value_embed,
                    deep_value_embed_hidden=deep_value_embed_hidden,
                    use_ffn_embed=self.use_ffn_embed,
                    use_qk_norm_post_rope=self.use_qk_norm_post_rope,
                    use_sliding_window=_block_uses_swa(i),
                    # YOCO uses `yoco_lower_window` (default 512) on
                    # the lower half; otherwise the standard
                    # `sliding_window_size` from the config (default
                    # 512). Both default to 512 at tiny1m3m, but
                    # keep them distinct for clarity.
                    sliding_window_size=(
                        self.yoco_lower_window if self.use_yoco
                        else self.sliding_window_size
                    ),
                    use_nope=self.use_nope,
                    use_fire_pe=self.use_fire_pe,
                    fire_pe_d_phi=self.fire_pe_d_phi,
                    use_gated_attn=self.use_gated_attn,
                    use_cope=self.use_cope,
                    use_fox=self.use_fox,
                    use_softpick=self.use_softpick,
                    use_ssmax=self.use_ssmax,
                    use_canon_conv=self.use_canon_conv,
                    # 143 — ShortConv pass-through to the standard
                    # transformer block. See
                    # `autoresearch/ideas/143-shortconv/idea.md`.
                    use_short_conv=self.use_short_conv,
                    short_conv_kernel=self.short_conv_kernel,
                    use_value_residual=self.use_value_residual,
                    # 117 — Soft MoE pass-through to the block.
                    use_soft_moe=self.use_soft_moe,
                    soft_moe_n_experts=self.soft_moe_n_experts,
                    soft_moe_n_slots=self.soft_moe_n_slots,
                    # 145 — Expert-Choice MoE pass-through to the block.
                    use_expert_choice_moe=self.use_expert_choice_moe,
                    n_moe_experts=self.n_moe_experts,
                    # 149 — TTT-Linear pass-through to the standard
                    # transformer block. Default off → FFN path bit-
                    # identical. See
                    # `autoresearch/ideas/149-ttt-linear/idea.md`.
                    use_ttt_ffn=self.use_ttt_ffn,
                    ttt_lr_init=self.ttt_lr_init,
                    # 118 — Mixture-of-Depths pass-through to the block.
                    use_mod=self.use_mod,
                    mod_capacity=self.mod_capacity,
                    mod_router_hidden=self.mod_router_hidden,
                    # 148 — Focal Modulation pass-through to the
                    # block. Default off → MHA path is bit-identical.
                    # See `autoresearch/ideas/148-focal-mod/idea.md`.
                    use_focal_mod=getattr(config, "use_focal_mod", False),
                    focal_mod_kernels=getattr(config, "focal_mod_kernels", (3, 5, 7)),
                    # 111 — DropPath / Stochastic Depth. Pass-through
                    # to the block; the per-step Bernoulli sample runs
                    # inside `TransformerBlock.forward` keyed off the
                    # `layer_index` kwarg passed by the model loop.
                    use_drop_path=getattr(config, "use_drop_path", False),
                    drop_path_max=getattr(config, "drop_path_max", 0.1),
                    # 131 — LayerDrop. Pass-through to the block.
                    use_layerdrop=self.use_layerdrop,
                    layerdrop_p=self.layerdrop_p,
                    layerdrop_schedule=self.layerdrop_schedule,
                    rope_base=self.rope_base,
                    use_tied_qk=self.use_tied_qk,
                    use_mla=self.use_mla,
                    mla_latent_dim=self.mla_latent_dim,
                    attention_dilation=self.attention_dilation,
                    use_post_norm=self.use_post_norm,
                    use_layernorm=self.use_layernorm,
                    use_linear_attn=self.use_linear_attn,
                    use_diff_attn=self.use_diff_attn,
                    use_nsa_global=self.use_nsa_global,
                    nsa_block=self.nsa_block,
                    use_hybrid_heads=self.use_hybrid_heads,
                    norm_type=self.norm_type,
                    qk_norm_type=self.qk_norm_type,
                    v_norm_type=self.v_norm_type,
                    # #16 QK-Norm pass-through to the block.
                    use_qk_layernorm=self.use_qk_layernorm,
                    # 029 — V-Norm pass-through to the block.
                    use_v_layernorm=self.use_v_layernorm,
                    use_multiscale_heads=self.use_multiscale_heads,
                    use_parallel_block=self.use_parallel_block,
                    use_attn_sink=self.use_attn_sink,
                    use_sub_ln=self.use_sub_ln,
                    q_norm_type=self.q_norm_type,
                    use_alibi_bias=self.use_alibi_bias,
                    use_q_temp_token=self.use_q_temp_token,
                    use_cosine_attn=self.use_cosine_attn,
                    use_qk_bilinear=self.use_qk_bilinear,
                    use_talking_heads_q=self.use_talking_heads_q,
                    use_per_head_rope_base=self.use_per_head_rope_base,
                    partial_rotary_p=self.partial_rotary_p,
                    use_q_expansion=self.use_q_expansion,
                    use_decoupled_content_pos=self.use_decoupled_content_pos,
                    use_antisym_qk=self.use_antisym_qk,
                    use_q_per_head_bias=self.use_q_per_head_bias,
                    use_q_per_channel_gain=self.use_q_per_channel_gain,
                    use_q_hd_gain=self.use_q_hd_gain,
                    use_q_norm_gate=self.use_q_norm_gate,
                    use_q_lowrank_refine=self.use_q_lowrank_refine,
                    q_lowrank_refine_rank=self.q_lowrank_refine_rank,
                    use_q_layerscale=self.use_q_layerscale,
                    use_q_softplus_gain=self.use_q_softplus_gain,
                    use_q_head_mix=self.use_q_head_mix,
                    use_q_time_conv=self.use_q_time_conv,
                    use_q_ema_smooth=self.use_q_ema_smooth,
                    q_ema_alpha=self.q_ema_alpha,
                    use_q_feature_map=self.use_q_feature_map,
                    q_feature_map_hidden=self.q_feature_map_hidden,
                    use_q_per_token_rope=self.use_q_per_token_rope,
                    q_per_token_rope_hidden=self.q_per_token_rope_hidden,
                    use_q_noise_reg=self.use_q_noise_reg,
                    # 134 — Mega EMA on V. Pass-through to each
                    # MultiHeadAttention. The construction-time assert
                    # requires 2·n_kv_heads == n_heads; tiny1m3m
                    # satisfies this (n_kv_heads=2, n_heads=4). Default
                    # off → baseline path bit-identical.
                    use_mega=getattr(config, "use_mega", False),
                    mega_beta=getattr(config, "mega_beta", 0.9),
                    mega_use_input=getattr(config, "mega_use_input", True),
                    value_embed_rank=value_embed_rank,
                    # 150 — Cross-Layer Feedback Attention pass-through
                    # to the block. Default off → baseline path bit-
                    # identical (no `XLayerCrossAttn` module built, no
                    # `xlayer_gate` param allocated). See
                    # `autoresearch/ideas/150-xlayer-feedback/idea.md`.
                    use_xlayer_feedback=self.use_xlayer_feedback,
                    xlayer_k=self.xlayer_k,
                )
                for i in range(n_unique)
            ]
        )

        # 116 — Hyper-Connections (mHC, Xie et al. 2024): when on,
        # wrap each tied-block slot with a per-position
        # `MultiStreamResidual` that applies (A_l, B_l, C_l) mixing on
        # n_resid parallel residual streams. Per-position (not per-
        # unique-block) so tied layers still get distinct mixings.
        # Default off → no wrappers built, baseline path bit-identical.
        # See `autoresearch/ideas/116-hyper-connections/idea.md`.
        self.use_hyper_connections = getattr(config, "use_hyper_connections", False)
        self.hc_n_resid = max(1, getattr(config, "hc_n_resid", 4))
        if self.use_hyper_connections:
            if config.d_model % self.hc_n_resid != 0:
                raise ValueError(
                    f"d_model ({config.d_model}) must be divisible by "
                    f"hc_n_resid ({self.hc_n_resid})"
                )
            self.hc_wrappers = nn.ModuleList(
                [
                    MultiStreamResidual(
                        self.transformer_blocks[i // self.tie_layer_groups],
                        n_resid=self.hc_n_resid,
                        d_model=config.d_model,
                    )
                    for i in range(config.n_layers)
                ]
            )
        else:
            self.hc_wrappers = None

        # #20 embedding residual: rms-norm the original embedding once at the top,
        # re-injected into every block.
        self.use_embed_residual = getattr(config, "use_embed_residual", False)
        if self.use_embed_residual:
            self.x0_norm = nn.RMSNorm(config.d_model)

        # B0 Tied output MLP — see docs/research-plans/tied-output-mlp/plan.md.
        # Autoencoder-tied shared Wu/Wd: encode runs once on the embedding,
        # decode runs once after the final norm. One extra FFN's worth of
        # params (2·d_model·d_ff), both 2-D so they go to Muon.
        self.use_tied_output_mlp = getattr(config, "use_tied_output_mlp", False)
        if self.use_tied_output_mlp:
            self.tied_output_mlp = TiedOutputMLP(config.d_model, config.d_ff)

        # B1 Untied output MLP — same shape as B0 but with separate decode
        # weights. Control for B0: isolates whether the tying matters vs
        # "just more output capacity." Costs 2× the params of B0
        # (4·d_model·d_ff). Default off (getattr fallback) keeps the
        # baseline byte-identical.
        self.use_untied_output_mlp = getattr(config, "use_untied_output_mlp", False)
        if self.use_untied_output_mlp:
            self.untied_output_mlp = UntiedOutputMLP(config.d_model, config.d_ff)

        # B2 Tied linear output MLP — B0 with NO nonlinearity. Sanity rung
        # for B0: should fold into the existing linear tied head, so we
        # expect ≈ baseline. Costs the same as B0 (2·d_model·d_ff).
        # Default off (getattr fallback) keeps the baseline byte-identical.
        self.use_tied_linear_output_mlp = getattr(
            config, "use_tied_linear_output_mlp", False
        )
        if self.use_tied_linear_output_mlp:
            self.tied_linear_output_mlp = TiedLinearOutputMLP(
                config.d_model, config.d_ff
            )

        # OH4 OutputTemp: logits /= τ, learnable scalar (τ=1 init). 1-D param,
        # routes to AdamW. τ=1 at init is an exact no-op, so step 0 == baseline.
        # See docs/research/output_head/plan.md (Batch 2).
        self.use_output_temp = getattr(config, "use_output_temp", False)
        if self.use_output_temp:
            self.output_temp_tau = nn.Parameter(torch.ones(1))
        # OH5 VocabBias: logits += b_v, learnable per-vocab bias (b=0 init).
        # 1-D param of size vocab_size, routes to AdamW. b=0 at init is an
        # exact no-op, so step 0 == baseline. Logit op — flows into eval CE
        # legitimately. See docs/research/output_head/plan.md (Batch 2).
        self.use_vocab_bias = getattr(config, "use_vocab_bias", False)
        if self.use_vocab_bias:
            self.vocab_bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 144 — Mixture of Softmaxes (Yang, Chen, et al. 2017,
        # arXiv:1711.03953, "Breaking the Softmax Bottleneck"). When
        # `use_mos=True`, allocate K-1 fresh vocab-sized LM heads
        # (`mos_heads_extra`) plus a small mix projection `mos_pi_proj`
        # of shape `(d_model → K)`. Head 0 is computed functionally
        # from the existing tied `lm_head` (full case) or from the
        # factorized `(emb_proj, token_embedding)` composition
        # (factorized case, tiny1m3m) — this guarantees step-0
        # bit-identity with the baseline (head 0's logits equal
        # the baseline's tied-head logits at init). The K-1 fresh
        # heads are NOT tied to `token_embedding`; their
        # `(K-1)·vocab·d_model` param cost is the lever's headline
        # expense. Default off → no MoS module built, baseline
        # forward graph bit-identical. See
        # `autoresearch/ideas/144-mos/idea.md`.
        self.use_mos = getattr(config, "use_mos", False)
        self.n_mos_components = max(1, int(getattr(config, "n_mos_components", 4)))
        if self.use_mos:
            K = self.n_mos_components
            self.mos_heads_extra = nn.ModuleList(
                [
                    nn.Linear(config.d_model, config.vocab_size, bias=False)
                    for _ in range(K - 1)
                ]
            )
            self.mos_pi_proj = nn.Linear(config.d_model, K, bias=True)

        # Output layers
        self.norm = make_norm(config.d_model, self.norm_type, self.use_layernorm)
        self.output_dropout = nn.Dropout(config.dropout)

        # OH7 UntieHead (OutputHead Batch 3 — see docs/research/output_head/plan.md):
        # separate lm_head weight from token_embedding. Costs vocab_size × d_model
        # extra params (NOT budget-matched). Probe — is weight-tying load-bearing?
        # Default off (getattr fallback) keeps the tied-head baseline byte-identical.
        # Flag is read via getattr so it doesn't require editing llm_config.py.
        self.use_untied_head = getattr(config, "use_untied_head", False)

        # Language modeling head (tied with embeddings).
        # Full case: standard tied Linear. Factorized case: lm_head is computed
        # functionally in forward() through the SAME two matrices, so input and
        # output embeddings stay tied with zero extra params.
        self.output_adapter_rank = getattr(config, "output_adapter_rank", None)
        if self.use_untied_head:
            # Untied head: independent [vocab, d_model] weight, same init scheme
            # as token_embedding (normal std=0.02, matching _init_weights for
            # nn.Embedding). 2-D param → routes to Muon under the existing rule.
            self.lm_head = nn.Parameter(torch.empty(config.vocab_size, config.d_model))
            torch.nn.init.normal_(self.lm_head, mean=0.0, std=0.02)
        elif self.emb_rank is None:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
        else:
            self.lm_head = None
        if self.output_adapter_rank is None:
            self.output_adapter_in = None
            self.output_adapter_out = None
        else:
            self.output_adapter_in = nn.Linear(
                config.d_model, self.output_adapter_rank, bias=False
            )
            self.output_adapter_out = nn.Linear(
                self.output_adapter_rank, config.vocab_size, bias=False
            )

        self.apply(self._init_weights)

        # Start the additive output path as an exact no-op, so step 0 matches the
        # tied-head baseline and the adapter earns any improvement during training.
        if self.output_adapter_out is not None:
            nn.init.zeros_(self.output_adapter_out.weight)

        # #22 zero-init residual projections: AFTER the global init, zero the
        # attention output projection (O-slice of the fused qkvo tensor) and the
        # FFN down-projection so every block is an exact identity at step 0.
        if getattr(config, "zero_init_resid", False):
            with torch.no_grad():
                for block in self.transformer_blocks:
                    block.attention.qkvo_proj[block.attention.qkv_size:].zero_()
                    nn.init.zeros_(block.feed_forward.down_proj.weight)
        # 024 — Gated Attention: AFTER the global init, re-zero the gate
        # projection (weight + bias). `_init_weights` re-inits every
        # `nn.Linear` with `normal_(std=0.02)`, which overwrites the
        # zero-init in MHA.__init__. We need W=0, b=0 here so 2·σ(0) = 1
        # exactly at step 0 (spec: "step-0 ≡ baseline to floating-point").
        if getattr(config, "use_gated_attn", False):
            with torch.no_grad():
                for block in self.transformer_blocks:
                    nn.init.zeros_(block.attention.gated_attn_proj.weight)
                    nn.init.zeros_(block.attention.gated_attn_proj.bias)
        # 144 — Mixture of Softmaxes: AFTER the global init, force the
        # mix projection to a one-hot at step 0. `W_π.weight = 0`,
        # `W_π.bias = [+1e4, -1e4, -1e4, -1e4]` ⇒ `softmax(W_π·h) =
        # [1, 0, 0, 0]` exactly in fp32 (the `exp(-2e4)` terms
        # underflow to 0). The downstream `logsumexp` then reduces to
        # `log_softmax(W_0 · h)` — bit-identical to the standard tied
        # head at step 0.
        if self.use_mos:
            with torch.no_grad():
                nn.init.zeros_(self.mos_pi_proj.weight)
                self.mos_pi_proj.bias.fill_(-1e4)
                self.mos_pi_proj.bias[0] = 1e4

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed_input(self, x):
        """Token embedding lookup + scaling + position_dropout.

        Returns (tok, x_post, x0, ve):
          tok     : raw embedding lookup (B, T, R) or (B, T, d_model); the
                    source for any value/query/key/output/ffn/deep
                    embedding branches.
          x_post  : residual-stream input to the transformer stack
                    (post emb_proj if emb_rank is set, post emb_scale,
                    post any encode-only output MLP).
          x0      : optional pre-norm residual-stream reference used by
                    the #20 embed-residual re-injection.
          ve      : raw token embedding passed to attention as the V/Q/K
                    injection source when any of those flags is on;
                    None otherwise.
        """
        # Token embeddings
        tok = self.token_embedding(x)  # rank-r (or d_model) lookup, reused as value-embed source
        # #76 embedding scale: -1.0 (default) = use sqrt(d_model).
        # Any other value overrides the scaling.
        emb_scale = getattr(self.config, 'embedding_scale', -1.0)
        if emb_scale < 0:
            emb_scale = math.sqrt(self.config.d_model)
        if self.emb_rank is None:
            x_post = tok * emb_scale
        else:
            x_post = self.emb_proj(tok) * emb_scale
        # B0 Tied output MLP: encode runs once on the (scaled) embedding,
        # before the block loop. Modify the input to the stack by adding
        # Wd·φ(Wu·x). See TiedOutputMLP docstring for step-0 caveat.
        if self.use_tied_output_mlp:
            x_post = self.tied_output_mlp.encode(x_post)
        # B1 Untied output MLP: same shape as B0 but with separate decode
        # weights. Encode path uses the same Wu/Wd pattern as B0 (no
        # transpose here — the tying is on the decode side only). See
        # UntiedOutputMLP docstring for the B0-shared step-0 caveat.
        if self.use_untied_output_mlp:
            x_post = self.untied_output_mlp.encode(x_post)
        # B2 Tied linear output MLP: same encode pattern as B0 (Wu then Wd,
        # no activation). See TiedLinearOutputMLP docstring for the
        # B0-shared step-0 caveat.
        if self.use_tied_linear_output_mlp:
            x_post = self.tied_linear_output_mlp.encode(x_post)
        # #29 value-embed source: the raw token embedding, injected into each V
        # #30 query-embed source: same `tok` (raw embedding). Both Q-embed and
        # V-embed can read from the same source, so we only branch once.
        # #31 key-embed source: same `tok` too.
        # #33 output-embed source: same `tok` (raw embedding). All four
        # share the same `ve` plumbing.
        ve = tok if (self.use_value_embed or self.use_query_embed or self.use_key_embed or self.use_output_embed or self.use_deep_value_embed or self.use_ffn_embed) else None
        if self.use_smear_gate:
            prev = torch.zeros_like(x_post)
            prev[:, 1:] = x_post[:, :-1]
            x_post = x_post + self.smear_gate * prev
        x_post = self.position_dropout(x_post)

        # #20 original embedding, normed once, re-injected into every block
        x0 = self.x0_norm(x_post) if self.use_embed_residual else None

        return tok, x_post, x0, ve

    def _run_post_embed(self, x, x0, ve):
        """Run the transformer stack from the post-embedding residual
        stream through to logits. Called by `forward` after the
        standard `_embed_input` and by `seqmix_forward` after the
        embedding-level mixup. The body is unchanged from the prior
        monolithic `forward` — pulled out only so the seqmix path can
        reuse it without re-implementing the loop.
        """
        # Pass through transformer blocks
        unet_skips = []
        # 021 — Value Residual: V_1 is a forward-pass-local stash;
        # layer 0 writes its post-W_V/post-GQA/post-transpose V to
        # `block.attention._v_residual`, and we then pass it as
        # `v_residual=V_1` to every layer l > 0. The forward-pass
        # index `i` (not the unique-block index) gates the stash so
        # layer tying still stashes from the first physical position
        # and blends into all later positions. None ⇒ MHA's
        # `use_value_residual` branch is the stash branch (layer 0)
        # or the lever is off altogether (`if self.use_value_residual:`
        # guard in MHA).
        v_residual = None
        # 129 — YOCO: `shared_kv` is the (K_g, V_g) tensor pair shared
        # across all upper-half blocks. Computed ONCE on the lower
        # half's final residual stream after the last lower-half
        # block (i = yoco_split - 1). Stays None on the lower half so
        # those blocks use the standard K, V projection path.
        shared_kv = None
        # 150 — Cross-Layer Feedback: `xlayer_mem` is a forward-pass-
        # local list of pre-FFN residual states from the previous
        # blocks. The current block reads from this list (Q from the
        # current pre-FFN x, K/V from the previous K pre-FFN states)
        # and appends its own pre-FFN x to the list before returning.
        # Stays as an empty list when the lever is off (the block
        # branch is no-op on the baseline path).
        xlayer_mem: list = []
        for i in range(self.config.n_layers):
            # 129 — YOCO dispatch: for i >= yoco_split, use the
            # upper-half ModuleList (YOCOLlamaBlock with
            # use_shared_kv=True) and pass `shared_kv` through. For
            # i < yoco_split, use the standard transformer_blocks
            # slot (with SWA on). Default off → standard path.
            if self.use_yoco and i >= self.yoco_split:
                block = self.yoco_upper_blocks[i - self.yoco_split]
                block_shared_kv = shared_kv
            else:
                block = self.transformer_blocks[i // self.tie_layer_groups]
                block_shared_kv = None
            if self.use_unet_skips and i >= self.config.n_layers - self.unet_skip_count:
                skip_idx = self.config.n_layers - 1 - i
                gate = self.unet_skip_gates[skip_idx]
                if self.unet_gate_type == "sigmoid":
                    gate = torch.sigmoid(gate)
                skip = unet_skips[skip_idx]
                if self.unet_bridge_norm:
                    skip = self.unet_bridge_norms[skip_idx](skip)
                x = x + gate * skip
            # 116 — Hyper-Connections: when on, the per-position
            # wrapper applies (A_l, B_l, C_l) stream mixing around the
            # tied block. Default off → direct block call (baseline path
            # bit-identical). Wrapper signature matches `block.forward`
            # exactly, so no extra plumbing.
            # YOCO + Hyper-Connections is currently unsupported (the
            # wrapper would need extra plumbing for shared_kv); reject
            # loudly if both are on simultaneously.
            if self.use_hyper_connections:
                assert not self.use_yoco, (
                    "use_hyper_connections + use_yoco is not supported "
                    "(the Hyper-Connections wrapper does not plumb "
                    "shared_kv through to the upper-half block)"
                )
                x = self.hc_wrappers[i](
                    x, x0, ve, v_residual=v_residual, layer_index=i
                )
            else:
                x = block(
                    x, x0, ve,
                    v_residual=v_residual,
                    layer_index=i,
                    shared_kv=block_shared_kv,
                    # 150 — Cross-Layer Feedback: forward-pass-local
                    # list of pre-FFN x from the previous K blocks.
                    # The block reads from it and appends its own
                    # pre-FFN x. `None` when the lever is off → the
                    # block branch is a no-op and the baseline path
                    # is bit-identical. We only allocate the list
                    # when the lever is on; the block itself guards
                    # on `use_xlayer_feedback` and `xlayer_mem is
                    # not None` to keep the baseline path zero-
                    # overhead.
                    xlayer_mem=(xlayer_mem if self.use_xlayer_feedback else None),
                )
            if self.use_value_residual and i == 0:
                # After layer-0 MHA forward, V_1 is stashed at
                # `block.attention._v_residual` (post-transpose,
                # shape `[B, n_heads, T, d_k]`). Capture for layers 1..N-1.
                v_residual = block.attention._v_residual
            if self.use_unet_skips and i < self.unet_skip_count:
                unet_skips.append(x)
            # 129 — YOCO: compute (K_g, V_g) once at the boundary
            # between the lower and upper halves. The next iteration
            # of the loop (i + 1 == yoco_split) will read this as
            # `shared_kv`.
            if self.use_yoco and i == self.yoco_split - 1:
                shared_kv = self.global_kv_head(x)

        # Output projection
        x = self.norm(x)
        # B0 Tied output MLP: decode runs after the final norm, before the
        # output dropout and the tied unembed. g_decode=0 at init, so the
        # decode path is a no-op at step 0 and the model earns it during
        # training.
        if self.use_tied_output_mlp:
            x = self.tied_output_mlp.decode(x)
        # B1 Untied output MLP: decode runs after the final norm, before
        # the output dropout and the tied unembed. Same g_decode=0 init
        # trick as B0 (decode is a no-op at step 0).
        if self.use_untied_output_mlp:
            x = self.untied_output_mlp.decode(x)
        # B2 Tied linear output MLP: decode runs after the final norm,
        # before the output dropout and the tied unembed. Same g_decode=0
        # init trick as B0 (decode is a no-op at step 0).
        if self.use_tied_linear_output_mlp:
            x = self.tied_linear_output_mlp.decode(x)
        x = self.output_dropout(x)
        # 144 — Mixture of Softmaxes: when on, replace the single
        # vocab-sized head with K parallel heads plus a per-token mix.
        # Head 0 is computed functionally from the existing tied
        # `lm_head` (full case) or the factorized composition
        # (factorized case) so it equals the baseline tied head at
        # step 0. Heads 1..K-1 are fresh `nn.Linear`s in
        # `mos_heads_extra`. We return `log p_mixed` (a
        # log-probability) instead of raw logits. `F.cross_entropy`
        # accepts this directly because `logsumexp(log_p) =
        # log(Σ exp(log_p)) = log(1) = 0`, so `F.cross_entropy(log_p,
        # label) = -log_p[label]` — the correct NLL of the mixture.
        # `argmax` and `softmax` over log_p give the same
        # predictions/probs as the underlying distribution. The
        # post-head logit tweaks (logit_softcap / output_temp /
        # vocab_bias / output_adapter) operate on raw logits, not
        # log-probs, so they are SKIPPED in the MoS path — MoS is its
        # own complete output mechanism. See
        # `autoresearch/ideas/144-mos/idea.md`.
        if self.use_mos:
            # Compute the mixture log-probability in chunks along B*T to
            # keep peak memory bounded. The naive path materializes
            # `(B, T, K, V)` for `log_softmax` — at tiny1m3m (B=2,
            # T=2048, K=4, V=49152, fp32) that's 3.0 GiB for
            # `logits_k` plus another 3.0 GiB for the `log_softmax`
            # output plus the broadcast add and `logsumexp` output,
            # totaling ~9 GiB just for the MoS forward — too big for
            # an RTX 3060 12GB. We chunk over the leading token
            # dimension so per-chunk memory is O(chunk · K · V · 4B).
            # Default `mos_chunk_size=128` keeps the per-chunk peak
            # around 150 MB (5 tensors of size chunk·V = 25 MB each,
            # ~125 MB concurrent). Identity at step 0 still holds:
            # for every chunk the result is `log_softmax(W_0 · h)`
            # (because `mos_pi_proj` is init one-hot at [+1e4, -1e4,
            # -1e4, -1e4] ⇒ `log π = [0, -2e4, -2e4, -2e4]` ⇒ the
            # k>0 contributions underflow in `logaddexp` and reduce
            # to the k=0 contribution). The chunks are concatenated
            # before returning, so the output is bit-identical to
            # the un-chunked reference at step 0.
            K = self.n_mos_components
            chunk = int(getattr(self.config, "mos_chunk_size", 128))
            V = self.config.vocab_size
            x_flat = x.reshape(-1, x.shape[-1])  # (N, d_model)
            N = x_flat.shape[0]
            log_p_chunks = []
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                xc = x_flat[start:end]  # (n, d_model)
                # Head 0 (= tied lm_head) functionally so head 0's
                # gradient flows back into `token_embedding` (and
                # `emb_proj` in the factorized case).
                if self.emb_rank is None:
                    logits_0 = F.linear(xc, self.token_embedding.weight)
                else:
                    z = F.linear(xc, self.emb_proj.weight.t())
                    logits_0 = F.linear(z, self.token_embedding.weight)
                # log_p_0 = log_softmax(logits_0). Use a non-in-place
                # `sub` (NOT `sub_`) so the autograd graph stays intact
                # — `sub_` would clobber the version counter on
                # `logits_0` and `cross_entropy.backward()` would
                # throw "variable has been modified by an inplace
                # operation" (this was a bug in the round-1 chunked
                # recode that surfaced only when the trainer's
                # gradient was actually requested). Memory cost is
                # one extra (n, V) tensor per chunk (~25 MB at
                # n=128, V=49152, fp32) — acceptable.
                log_z_0 = torch.logsumexp(logits_0, dim=-1, keepdim=True)
                log_p_0 = logits_0 - log_z_0
                # Mix weights: π = softmax(W_π · x) over K components.
                log_pi = F.log_softmax(self.mos_pi_proj(xc), dim=-1)  # (n, K)
                # log_p_mixed = logsumexp_k (log_pi_k + log_p_k).
                # Build incrementally so we never materialize (n, K, V).
                log_p_mixed = log_p_0 + log_pi[:, 0:1]  # (n, V)
                for k_idx in range(1, K):
                    logits_h = self.mos_heads_extra[k_idx - 1](xc)  # (n, V)
                    log_z = torch.logsumexp(logits_h, dim=-1, keepdim=True)
                    log_p_h = logits_h - log_z  # non-in-place → log_p_k
                    log_p_mixed = torch.logaddexp(
                        log_p_mixed, log_pi[:, k_idx:k_idx + 1] + log_p_h
                    )
                log_p_chunks.append(log_p_mixed)
            log_p_mixed = torch.cat(log_p_chunks, dim=0)  # (N, V)
            # Restore the original leading shape (typically (B, T, V)).
            leading = x.shape[:-1]
            log_p_mixed = log_p_mixed.reshape(*leading, V)
            return log_p_mixed

        # OH7 UntieHead: when on, use the independent [vocab, d_model] weight
        # instead of the tied embedding table. In the factorized case
        # (emb_rank is not None), the untied head uses the d_model-dimensional
        # x directly, bypassing the emb_proj reduction (the untied head is
        # full d_model → vocab, not r → vocab). See docs/research/output_head/plan.md
        # (Batch 3). Default off = tied (byte-identical to baseline).
        if self.use_untied_head:
            logits = x @ self.lm_head.T
        elif self.emb_rank is None:
            logits = self.lm_head(x)
        else:
            # Tied factorized head: d_model -> r (via emb_proj^T) -> vocab (via
            # token_embedding^T). Reuses the exact embedding matrices.
            z = F.linear(x, self.emb_proj.weight.t())          # (..., r)
            logits = F.linear(z, self.token_embedding.weight)  # (..., vocab)
        if self.output_adapter_out is not None:
            logits = logits + self.output_adapter_out(self.output_adapter_in(x))

        # #71 logit softcap (Gemma-style): logit_softcap=0.0 disables.
        # Applied right before the loss — gradient flows through tanh.
        softcap = getattr(self.config, 'logit_softcap', 0.0)
        if softcap > 0.0:
            logits = softcap * torch.tanh(logits / softcap)

        # OH4 OutputTemp: logits /= τ (τ=1 init = no-op). Logit op, so it
        # flows into eval CE legitimately per the output_head plan's
        # Reporting rule. See docs/research/output_head/plan.md (Batch 2).
        if self.use_output_temp:
            logits = logits / self.output_temp_tau
        # OH5 VocabBias: logits += b_v (b=0 init = no-op). Logit op, so it
        # flows into eval CE legitimately per the output_head plan's
        # Reporting rule. See docs/research/output_head/plan.md (Batch 2).
        if self.use_vocab_bias:
            logits = logits + self.vocab_bias

        return logits

    def forward(self, x):
        # Standard path: embed, then run the post-embed stack.
        _, x_post, x0, ve = self._embed_input(x)
        return self._run_post_embed(x_post, x0, ve)

    def seqmix_forward(self, x, y, alpha, generator=None):
        """133 — SeqMix token-level mixup (Guo, Mao, Zhang 2019,
        arXiv:1908.02951, extended to LM).

        Samples λ ~ Beta(α, α) once per call (the canonical per-batch
        mixup rate). Builds a paired batch by shuffling `x` along the
        batch axis (`x_b = x[perm]`). Looks up embeddings for both,
        mixes at the embedding level:

            emb_mixed = λ · emb_a + (1 − λ) · emb_b

        Then runs the post-embed stack ONCE on the mixed residual
        stream and computes the mixed-CE loss

            L = λ · CE(logits, y_a) + (1 − λ) · CE(logits, y_b)

        Returns `(loss, logits, lam)` so the trainer can keep the
        existing aux-loss composition pattern (entropy reg, z-loss,
        born-again, rdrop, etc. stay zero unless their flag is on).

        Identity at step 0: with `use_seqmix=False` this method is
        never invoked. With `use_seqmix=True` the mixed residual stream
        differs from the unmixed baseline by `O((1−λ) · ‖emb_b‖)` at
        step 0 — the lever's documented non-bit-identical signature
        (acknowledged in the idea spec).

        Args:
            x: input ids, [B, T].
            y: target ids, [B, T] (the trainer's standard `labels`;
               this method handles the next-token shift internally).
            alpha: Beta(α, α) shape parameter (paper default 0.4).
            generator: optional torch.Generator for reproducible λ
               draws.
        """
        B = x.size(0)
        # Per-batch λ (single scalar — matches the canonical mixup
        # formulation; per-token/per-position λ is a paper extension
        # not used here).
        if alpha <= 0.0:
            lam = 1.0
        else:
            lam = float(
                torch.distributions.Beta(alpha, alpha).sample().item()
            )
        # Shuffle the batch to produce the paired sequence.
        if generator is not None:
            perm = torch.randperm(B, generator=generator, device=x.device)
        else:
            perm = torch.randperm(B, device=x.device)
        x_b = x[perm]
        y_b = y[perm]

        # Token-embedding lookup for both sequences.
        tok_a = self.token_embedding(x)
        tok_b = self.token_embedding(x_b)
        # Mix in the raw embedding space. When emb_rank is set, the
        # emb_proj is a single linear layer, so mixing the raw
        # rank-r embedding and then projecting is identical to
        # projecting then mixing (linearity of emb_proj). Mixup
        # operates on the pre-projection embedding for symmetry with
        # the LM-extension literature.
        tok_mixed = lam * tok_a + (1.0 - lam) * tok_b

        # Build the post-embed residual stream manually (we already
        # mixed at the raw-embedding level, so we skip the shared
        # `_embed_input` path and apply emb_proj + emb_scale + the
        # encode-only output MLPs + smear + position_dropout here).
        emb_scale = getattr(self.config, 'embedding_scale', -1.0)
        if emb_scale < 0:
            emb_scale = math.sqrt(self.config.d_model)
        if self.emb_rank is None:
            x_post = tok_mixed * emb_scale
        else:
            x_post = self.emb_proj(tok_mixed) * emb_scale
        if self.use_tied_output_mlp:
            x_post = self.tied_output_mlp.encode(x_post)
        if self.use_untied_output_mlp:
            x_post = self.untied_output_mlp.encode(x_post)
        if self.use_tied_linear_output_mlp:
            x_post = self.tied_linear_output_mlp.encode(x_post)
        if self.use_smear_gate:
            prev = torch.zeros_like(x_post)
            prev[:, 1:] = x_post[:, :-1]
            x_post = x_post + self.smear_gate * prev
        x_post = self.position_dropout(x_post)

        # For the seqmix path we set x0=None and ve=None: the original
        # token identity has already been mixed into the residual
        # stream, so re-injecting either the per-position x0 reference
        # or the raw ve would double-count the token signal. Setting
        # them to None keeps the embed-residual / value-embed branches
        # inert for the mixed input.
        x0 = None
        ve = None
        logits = self._run_post_embed(x_post, x0, ve)

        # Mixed-CE loss: λ · CE(logits, y_a) + (1−λ) · CE(logits, y_b).
        # The trainer normally handles the shift_labels step. To keep
        # this method drop-in for the trainer's per-batch loop we
        # compute CE on the standard next-token alignment here.
        shift_a = torch.full_like(y, -100)
        shift_a[:, :-1] = y[:, 1:]
        shift_b = torch.full_like(y_b, -100)
        shift_b[:, :-1] = y_b[:, 1:]
        ce_a = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            shift_a.view(-1),
            ignore_index=-100,
        )
        ce_b = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            shift_b.view(-1),
            ignore_index=-100,
        )
        loss = lam * ce_a + (1.0 - lam) * ce_b
        return loss, logits, lam
