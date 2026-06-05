import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from configs.llm_config import LLMConfig
from models.layers import TransformerBlock, make_norm


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
            self.unet_skip_count = config.n_layers // 2
            self.unet_skip_gates = nn.Parameter(
                torch.zeros(self.unet_skip_count, config.d_model)
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
        self.use_multiscale_heads = getattr(config, "use_multiscale_heads", False)
        self.use_parallel_block = getattr(config, "use_parallel_block", False)
        self.use_attn_sink = getattr(config, "use_attn_sink", False)
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
                    use_multiscale_heads=self.use_multiscale_heads,
                    use_parallel_block=self.use_parallel_block,
                    use_attn_sink=self.use_attn_sink,
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

        # Output layers
        self.norm = make_norm(config.d_model, self.norm_type, self.use_layernorm)
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head (tied with embeddings).
        # Full case: standard tied Linear. Factorized case: lm_head is computed
        # functionally in forward() through the SAME two matrices, so input and
        # output embeddings stay tied with zero extra params.
        self.output_adapter_rank = getattr(config, "output_adapter_rank", None)
        if self.emb_rank is None:
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
        for i in range(self.config.n_layers):
            block = self.transformer_blocks[i // self.tie_layer_groups]
            if self.use_unet_skips and i >= self.config.n_layers - self.unet_skip_count:
                skip_idx = self.config.n_layers - 1 - i
                x = x + self.unet_skip_gates[skip_idx] * unet_skips[skip_idx]
            x = block(x, x0, ve)
            if self.use_unet_skips and i < self.unet_skip_count:
                unet_skips.append(x)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        if self.emb_rank is None:
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

        return logits
