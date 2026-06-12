import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from configs.llm_config import LLMConfig
from models.layers import TransformerBlock, make_norm


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
        # 021 — Value Residual Learning (cross-layer V shortcut).
        # Layer 0 stashes its post-W_V/post-GQA/post-transpose V on
        # `block.attention._v_residual`; the forward loop below reads it
        # after the layer-0 call and passes it as `v_residual=V_1` to
        # every layer l > 0. Per-block scalar `lambda_v` (init 0) lives
        # on each MHA; step-0 ≡ baseline. Default off → baseline path
        # bit-identical. See `autoresearch/ideas/021-value-residual/plan.md`.
        self.use_value_residual = getattr(config, "use_value_residual", False)
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
            if not self.use_sliding_window:
                return False
            if self.global_attn_every_k > 0 and ((i + 1) % self.global_attn_every_k == 0):
                return False  # this is a global (full-attention) layer
            return True

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
                    use_talking_heads_out=getattr(config, "use_talking_heads_out", False),
                    out_op=getattr(config, "out_op", ""),
                    use_re_zero=getattr(config, "use_re_zero", False),
                    resid_mode=getattr(config, "resid_mode", ""),
                    n_layers=config.n_layers,
                    use_layerscale=getattr(config, "use_layerscale", False),
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
                    sliding_window_size=self.sliding_window_size,
                    use_nope=self.use_nope,
                    use_fire_pe=self.use_fire_pe,
                    fire_pe_d_phi=self.fire_pe_d_phi,
                    use_gated_attn=self.use_gated_attn,
                    use_cope=self.use_cope,
                    use_fox=self.use_fox,
                    use_softpick=self.use_softpick,
                    use_ssmax=self.use_ssmax,
                    use_canon_conv=self.use_canon_conv,
                    use_value_residual=self.use_value_residual,
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
                    value_embed_rank=value_embed_rank,
                )
                for i in range(n_unique)
            ]
        )

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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # Token embeddings
        tok = self.token_embedding(x)  # rank-r (or d_model) lookup, reused as value-embed source
        # #76 embedding scale: -1.0 (default) = use sqrt(d_model).
        # Any other value overrides the scaling.
        emb_scale = getattr(self.config, 'embedding_scale', -1.0)
        if emb_scale < 0:
            emb_scale = math.sqrt(self.config.d_model)
        if self.emb_rank is None:
            x = tok * emb_scale
        else:
            x = self.emb_proj(tok) * emb_scale
        # B0 Tied output MLP: encode runs once on the (scaled) embedding,
        # before the block loop. Modify the input to the stack by adding
        # Wd·φ(Wu·x). See TiedOutputMLP docstring for step-0 caveat.
        if self.use_tied_output_mlp:
            x = self.tied_output_mlp.encode(x)
        # B1 Untied output MLP: same shape as B0 but with separate decode
        # weights. Encode path uses the same Wu/Wd pattern as B0 (no
        # transpose here — the tying is on the decode side only). See
        # UntiedOutputMLP docstring for the B0-shared step-0 caveat.
        if self.use_untied_output_mlp:
            x = self.untied_output_mlp.encode(x)
        # B2 Tied linear output MLP: same encode pattern as B0 (Wu then Wd,
        # no activation). See TiedLinearOutputMLP docstring for the
        # B0-shared step-0 caveat.
        if self.use_tied_linear_output_mlp:
            x = self.tied_linear_output_mlp.encode(x)
        # #29 value-embed source: the raw token embedding, injected into each V
        # #30 query-embed source: same `tok` (raw embedding). Both Q-embed and
        # V-embed can read from the same source, so we only branch once.
        # #31 key-embed source: same `tok` too.
        # #33 output-embed source: same `tok` (raw embedding). All four
        # share the same `ve` plumbing.
        ve = tok if (self.use_value_embed or self.use_query_embed or self.use_key_embed or self.use_output_embed or self.use_deep_value_embed or self.use_ffn_embed) else None
        if self.use_smear_gate:
            prev = torch.zeros_like(x)
            prev[:, 1:] = x[:, :-1]
            x = x + self.smear_gate * prev
        x = self.position_dropout(x)

        # #20 original embedding, normed once, re-injected into every block
        x0 = self.x0_norm(x) if self.use_embed_residual else None

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
        for i in range(self.config.n_layers):
            block = self.transformer_blocks[i // self.tie_layer_groups]
            if self.use_unet_skips and i >= self.config.n_layers - self.unet_skip_count:
                skip_idx = self.config.n_layers - 1 - i
                gate = self.unet_skip_gates[skip_idx]
                if self.unet_gate_type == "sigmoid":
                    gate = torch.sigmoid(gate)
                skip = unet_skips[skip_idx]
                if self.unet_bridge_norm:
                    skip = self.unet_bridge_norms[skip_idx](skip)
                x = x + gate * skip
            x = block(x, x0, ve, v_residual=v_residual)
            if self.use_value_residual and i == 0:
                # After layer-0 MHA forward, V_1 is stashed at
                # `block.attention._v_residual` (post-transpose,
                # shape `[B, n_heads, T, d_k]`). Capture for layers 1..N-1.
                v_residual = block.attention._v_residual
            if self.use_unet_skips and i < self.unet_skip_count:
                unet_skips.append(x)

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
