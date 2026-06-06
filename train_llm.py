import argparse
import time
import os
import torch
import logging
import random
import numpy as np
from torch.utils.data import DataLoader

# Fix tokenizer parallelism warning when using DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.llm_config import (
    LLMConfig,
    Tiny1M3MConfig,
    Screen10M20MConfig,
    Full10M200MConfig,
    Full135M2700MConfig,
)
from configs.dataset_config import DataConfig
from training.trainer import train_minimal_llm
from training.device import DEVICE_CHOICES, describe_device, resolve_device
from utils.helpers import set_seed, format_time
from utils.logger import setup_logging


# Worker init function to ensure each worker has a deterministic seed
# Global seed used by worker_init_fn (set in main)
_GLOBAL_SEED = 42

def worker_init_fn(worker_id):
    worker_seed = _GLOBAL_SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def print_system_info(requested_device: str):
    device = resolve_device(requested_device)
    print(f"Device: {describe_device(device)}")
    print(f"PyTorch: {torch.__version__}\n")


def prepare_datasets(data_cfg, tokenizer, cache_dir="./processed_data"):
    import json
    import shutil
    from datasets import load_from_disk, load_dataset, Dataset
    from data.loader import tokenize_and_chunk, finalize_dataset

    # CASE 0: Dataset path is already a processed on-disk dataset
    # We check if the path passed in data_cfg.dataset_path is a directory containing a dataset dict
    if os.path.isdir(data_cfg.dataset_path):
        # Check for metadata to validate max_seq_len consistency
        metadata_path = os.path.join(data_cfg.dataset_path, "prep_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    prep_metadata = json.load(f)
                    prep_max_seq = prep_metadata.get("max_seq_len")
                    if prep_max_seq and prep_max_seq != data_cfg.seq_length:
                        print("\n" + "="*70)
                        print("⚠️  WARNING: max_seq_len MISMATCH DETECTED!")
                        print("="*70)
                        print(f"  Data was prepared with max_seq_len: {prep_max_seq}")
                        print(f"  Current config has max_seq_len:     {data_cfg.seq_length}")
                        print(f"\n  This WILL cause RoPE cache shape mismatch errors!")
                        print(f"  Use the pre-built dataset from the README (python data/download_hf_data.py)")
                        print(f"  and keep max_seq_len=2048. Changing the data is not recommended —")
                        print(f"  ask user first (if you are AI).")
                        print("="*70 + "\n")
                        raise ValueError(f"max_seq_len mismatch: prepared={prep_max_seq}, config={data_cfg.seq_length}. Run: python data/prepare_mix_data.py --target_tokens 25_000_000 or adjust the number of tokens")
                    else:
                        print(f"✓ Validated: Data prepared with max_seq_len={prep_max_seq}")
            except json.JSONDecodeError:
                print("Warning: Could not read prep_metadata.json")
        
        # Heuristic: check if it has dataset_dict.json or state.json or just load it
        try:
            print(f"Loading pre-processed dataset from {data_cfg.dataset_path}...")
            # We assume it's a dataset with "input_ids" and "labels"
            ds = load_from_disk(data_cfg.dataset_path)
            
            # Set format to torch for preprocessed datasets
            if hasattr(ds, 'set_format'):
                # Single dataset
                if "input_ids" in ds.column_names and "labels" in ds.column_names:
                    ds.set_format(type="torch", columns=["input_ids", "labels"])
            
            # If it's a DatasetDict (train, val), return it
            if isinstance(ds, dict) or hasattr(ds, "keys"):
                if "train" in ds and "val" in ds:
                    # Set format for both splits
                    if hasattr(ds["train"], 'set_format'):
                        ds["train"].set_format(type="torch", columns=["input_ids", "labels"])
                        ds["val"].set_format(type="torch", columns=["input_ids", "labels"])
                    return ds["train"], ds["val"]
                elif "train" in ds:
                    # Splitting manually if only train exists
                    print("Found only 'train' split. Creating validation split...")
                    splitted = ds["train"].train_test_split(test_size=0.1, seed=42)
                    # Set format for both splits
                    splitted["train"].set_format(type="torch", columns=["input_ids", "labels"])
                    splitted["test"].set_format(type="torch", columns=["input_ids", "labels"])
                    return splitted["train"], splitted["test"]
            
            # If it's a single Dataset (just rows)
            print("Loaded single dataset. Splitting into train/val...")
            splitted = ds.train_test_split(test_size=0.1, seed=42)
            # Set format for both splits
            splitted["train"].set_format(type="torch", columns=["input_ids", "labels"])
            splitted["test"].set_format(type="torch", columns=["input_ids", "labels"])
            return splitted["train"], splitted["test"]

        except Exception as e:

            # Fallback: try loading "train" and "val" subdirectories directly
            try:
                train_path = os.path.join(data_cfg.dataset_path, "train")
                val_path = os.path.join(data_cfg.dataset_path, "val")
                if os.path.exists(train_path) and os.path.exists(val_path):
                    print(f"Loading separate train/val datasets from {data_cfg.dataset_path}...")
                    train_ds = load_from_disk(train_path)
                    val_ds = load_from_disk(val_path)
                    
                    if hasattr(train_ds, 'set_format'):
                        train_ds.set_format(type="torch", columns=["input_ids", "labels"])
                    if hasattr(val_ds, 'set_format'):
                        val_ds.set_format(type="torch", columns=["input_ids", "labels"])
                    return train_ds, val_ds
            except Exception as e2:
                print(f"Sub-directory load failed: {e2}")

            print(f"Could not load as direct dataset ({e}). Falling back to HF loading...")

    # cache_dir provided via argument
    train_cache = os.path.join(cache_dir, "train")
    val_cache = os.path.join(cache_dir, "val")
    info_path = os.path.join(cache_dir, "dataset_info.json")

    # Define what config parameters invalidate the cache
    config_state = {
        "dataset_path": data_cfg.dataset_path,
        "dataset_name": data_cfg.dataset_name,
        "tokenizer_name": data_cfg.tokenizer_name,
        "seq_length": data_cfg.seq_length,
        "num_samples": data_cfg.num_samples,
    }

    # 1. Try to load valid cache
    if os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                if json.load(f) == config_state:
                    print(f"Loading cached datasets from {cache_dir}...")
                    return load_from_disk(train_cache), load_from_disk(val_cache)
            print("Cache configuration mismatch. Rebuilding...")
        except Exception as e:
            print(f"Cache check failed ({e}). Rebuilding...")
    
    # 2. Rebuild cache
    if os.path.exists(cache_dir):
        print(f"Cleaning old cache at {cache_dir}...")
        shutil.rmtree(cache_dir)
    
    # Ensure directory exists immediately
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load and split
    print("Loading raw dataset and splitting documents...")
    raw_dataset = load_dataset(
        data_cfg.dataset_path,
        data_cfg.dataset_name,
        split=data_cfg.split,
        cache_dir=data_cfg.cache_dir,
        streaming=True,
    )
    
    # Streaming requires taking samples explicitly
    raw_samples = list(raw_dataset.take(data_cfg.num_samples))
    num_val = int(len(raw_samples) * 0.1)
    num_train = len(raw_samples) - num_val
    
    raw_train = Dataset.from_list(raw_samples[:num_train])
    raw_val = Dataset.from_list(raw_samples[num_train:])
    print(f"Split into {len(raw_train):,} train docs and {len(raw_val):,} val docs")
    
    # Tokenize and save
    print("Tokenizing train set...")
    train_ds = finalize_dataset(tokenize_and_chunk(raw_train, tokenizer, data_cfg), data_cfg)
    train_ds.save_to_disk(train_cache)
    
    print("Tokenizing validation set...")
    val_ds = finalize_dataset(tokenize_and_chunk(raw_val, tokenizer, data_cfg), data_cfg)
    val_ds.save_to_disk(val_cache)

    # Save cache info
    os.makedirs(cache_dir, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(config_state, f, indent=2)
    print("Saved dataset cache info.")

    return train_ds, val_ds


def build_eval_milestones(train_tokens: int) -> tuple[int, ...]:
    """Return denser validation checkpoints for longer baseline runs."""
    if train_tokens <= 100_000:
        return tuple(range(64))  # toy/nano runs: eval after every step
    if train_tokens <= 8_000_000:
        return (0, 25, 50, 75, 100, 150, 200, 300, 400)
    if train_tokens <= 25_000_000:
        return (0, 50, 100, 200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500)
    if train_tokens <= 50_000_000:
        return (0, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000)
    if train_tokens <= 100_000_000:
        return (0, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000)
    return (0, 500, 1000, 2000, 4000, 8000, 12000, 20000, 30000, 40000, 50000)


def main():
    global _GLOBAL_SEED
    logger = setup_logging(log_dir="./logs")
    logger.info("Starting training")

    parser = argparse.ArgumentParser(description="Train MoE Model")
    parser.add_argument("--muon_lr", type=float, help="Override Muon learning rate")
    parser.add_argument("--adamw_lr", type=float, help="Override AdamW learning rate")
    parser.add_argument("--train_tokens", type=int, help="Override train_tokens")
    parser.add_argument("--use_embed_residual", type=str, help="Re-inject token embedding each block, #20 (true/false)")
    parser.add_argument("--zero_init_resid", type=str, help="Zero-init attn-O + ffn-down projections, #22 (true/false)")
    parser.add_argument("--use_value_embed", type=str, help="Inject (factorized) token embedding into attention V, #29 (true/false)")
    parser.add_argument("--use_query_embed", type=str, help="Inject (factorized) token embedding into attention Q, #30 (true/false)")
    parser.add_argument("--use_key_embed", type=str, help="Inject (factorized) token embedding into attention K, #31 (true/false)")
    parser.add_argument("--use_output_embed", type=str, help="Inject (factorized) token embedding into attention output (post-O), #33 (true/false)")
    parser.add_argument("--use_q_gain", type=str, help="Per-head learnable Q-gain (post-RoPE), #37 (true/false)")
    parser.add_argument("--use_k_gain", type=str, help="Per-head learnable K-gain (post-RoPE), #42 (true/false)")
    parser.add_argument("--use_deep_value_embed", type=str, help="2-layer non-linear V-embed (GELU bottleneck), #45 (true/false)")
    parser.add_argument("--deep_value_embed_hidden", type=int, help="Hidden dim for deep V-embed (default 96)")
    parser.add_argument("--use_ffn_embed", type=str, help="Token identity into FFN input, #47 (true/false)")
    parser.add_argument("--use_qk_norm_post_rope", type=str, help="QK-RMSNorm after RoPE (modded-nanogpt variant), #49 (true/false)")
    parser.add_argument("--use_sliding_window", type=str, help="Use local causal sliding-window attention (true/false)")
    parser.add_argument("--sliding_window_size", type=int, help="Override sliding-window attention size")
    parser.add_argument("--rope_base", type=int, help="Override RoPE base")
    parser.add_argument("--n_kv_heads", type=int, help="Override KV head count for GQA/MHA sweeps")
    parser.add_argument("--ffn_variant", type=str, choices=["squared_relu", "swiglu", "gelu", "satrelu"], help="Override FFN activation variant")
    parser.add_argument("--use_layernorm", type=str, help="Use LayerNorm instead of RMSNorm (true/false)")
    parser.add_argument("--norm_type", type=str, help="Residual-stream norm, #90: rmsnorm/layernorm/peak/manhattan/squash/center/manifold/pnorm<p>")
    parser.add_argument("--qk_norm_type", type=str, help="Norm on Q,K before attention dot product, #91 (e.g. pnorm1.5)")
    parser.add_argument("--v_norm_type", type=str, help="Norm on V before the attention weighted sum, #92 (e.g. pnorm1.5; ''=off)")
    parser.add_argument("--use_multiscale_heads", type=str, help="Per-head graded sliding-window sizes, #97 (true/false)")
    parser.add_argument("--use_parallel_block", type=str, help="Parallel attention+FFN block (PaLM/GPT-J), #98 (true/false)")
    parser.add_argument("--use_attn_sink", type=str, help="Attention sink slot / softmax-off-by-one, #99 (true/false)")
    parser.add_argument("--use_unet_skips", type=str, help="U-net cross-layer skip connections (true/false)")
    parser.add_argument("--unet_skip_count", type=int, help="Number of U-net skip pairs")
    parser.add_argument("--unet_gate_type", type=str, choices=["raw", "sigmoid"], help="U-net gate parameterization (raw or sigmoid)")
    parser.add_argument("--unet_gate_init", type=float, help="Initial U-net gate value (e.g. 0.0 for raw default, -1.5 for sigmoid speedrun)")
    parser.add_argument("--use_smear_gate", type=str, help="Token-smear gate (mix previous token) (true/false)")
    parser.add_argument("--use_attn_output_gate", type=str, help="Gated attention output (true/false)")
    parser.add_argument("--use_layerscale", type=str, help="Learnable per-channel residual gate (LayerScale), (true/false)")
    parser.add_argument("--use_post_norm", type=str, help="Use post-norm transformer blocks (true/false)")
    parser.add_argument("--use_tied_qk", type=str, help="Use tied-QK attention projections (true/false)")
    parser.add_argument("--use_mla", type=str, help="Use MLA-style latent K/V projections (true/false)")
    parser.add_argument("--mla_latent_dim", type=int, help="Override MLA latent dimension")
    parser.add_argument("--use_linear_attn", type=str, help="Use positive-feature linear attention (true/false)")
    parser.add_argument("--logit_softcap", type=float, help="Override final-logit softcap; 0 disables")
    parser.add_argument("--attention_dilation", type=int, help="Override attention dilation inside the local window")
    parser.add_argument("--global_attn_every_k", type=int, help="Every k-th layer uses full attention instead of SWA, #86 (0=off)")
    parser.add_argument("--use_diff_attn", type=str, help="Differential Attention (DIFF Transformer), #87 (true/false)")
    parser.add_argument("--use_nsa_global", type=str, help="NSA-style local + block-compressed global attention, #88 (true/false)")
    parser.add_argument("--nsa_block", type=int, help="Block size for NSA compressed-global pooling, #88 (default 64)")
    parser.add_argument("--use_hybrid_heads", type=str, help="Half heads local / half global, every layer, #89 (true/false)")
    parser.add_argument("--d_ff", type=int, help="Override FFN hidden dim (for param-matched comparisons)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "tiny1m", "screen10m", "10m", "135m"],
        help="Preset config to load",
    )
    parser.add_argument("--config_class", type=str, help="Python path to config class (e.g., configs.llm_config.LLMConfig)")
    parser.add_argument("--load_checkpoint", type=str, help="Path to checkpoint file to load weights from. Point at a full gate.pt/latest.pt to RESUME the same run (restores optimizer + scheduler + step + RNG).")
    parser.add_argument("--stop_at_step", type=int, help="Pause the run at the first eval milestone >= this step: saves a resumable gate.pt and exits for human review. Resume with --load_checkpoint <dir>/gate.pt.")
    parser.add_argument("--compile", type=str, help="Whether to compile the model (true/false)")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=DEVICE_CHOICES,
        help="Device to use: auto prefers CUDA, then MPS, then CPU",
    )
    parser.add_argument("--dataset_path", type=str, help="Path to preprocessed dataset directory")
    parser.add_argument("--eval_every", type=int, help="Override eval_every steps")
    parser.add_argument("--save_every", type=int, help="Override save_every steps")
    parser.add_argument("--batch_size", type=int, help="Override batch_size")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Override gradient_accumulation_steps")
    parser.add_argument("--log_every", type=int, default=100, help="Logging frequency in steps")
    parser.add_argument("--warmup", type=str, default="true", help="Whether to perform untimed compilation warmup (true/false)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    print_system_info(args.device)

    # Set global seed for reproducibility
    _GLOBAL_SEED = args.seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Load Config
    preset_map = {
        "default": LLMConfig,
        "tiny1m": Tiny1M3MConfig,
        "screen10m": Screen10M20MConfig,
        "10m": Full10M200MConfig,
        "135m": Full135M2700MConfig,
    }

    if args.config_class:
        import importlib
        try:
            module_name, class_name = args.config_class.rsplit(".", 1)
            module = importlib.import_module(module_name)
            ConfigClass = getattr(module, class_name)
            print(f"Loading config from {args.config_class}")
            config = ConfigClass()
        except Exception as e:
            print(f"Error loading config class {args.config_class}: {e}")
            raise e
    else:
        print(f"Loading config preset: {args.config}")
        config = preset_map[args.config]()

    # Override config with args
    config.seed = args.seed
    if args.muon_lr is not None:
        config.muon_lr = args.muon_lr
    if args.adamw_lr is not None:
        config.adamw_lr = args.adamw_lr
    if args.train_tokens is not None:
        config.train_tokens = args.train_tokens
    if args.use_embed_residual is not None:
        config.use_embed_residual = (args.use_embed_residual.lower() == "true")
    if args.zero_init_resid is not None:
        config.zero_init_resid = (args.zero_init_resid.lower() == "true")
    if args.use_value_embed is not None:
        config.use_value_embed = (args.use_value_embed.lower() == "true")
    if args.use_query_embed is not None:
        config.use_query_embed = (args.use_query_embed.lower() == "true")
    if args.use_key_embed is not None:
        config.use_key_embed = (args.use_key_embed.lower() == "true")
    if args.use_output_embed is not None:
        config.use_output_embed = (args.use_output_embed.lower() == "true")
    if args.use_q_gain is not None:
        config.use_q_gain = (args.use_q_gain.lower() == "true")
    if args.use_k_gain is not None:
        config.use_k_gain = (args.use_k_gain.lower() == "true")
    if args.use_deep_value_embed is not None:
        config.use_deep_value_embed = (args.use_deep_value_embed.lower() == "true")
    if args.deep_value_embed_hidden is not None:
        config.deep_value_embed_hidden = args.deep_value_embed_hidden
    if args.use_ffn_embed is not None:
        config.use_ffn_embed = (args.use_ffn_embed.lower() == "true")
    if args.use_qk_norm_post_rope is not None:
        config.use_qk_norm_post_rope = (args.use_qk_norm_post_rope.lower() == "true")
    if args.use_sliding_window is not None:
        config.use_sliding_window = (args.use_sliding_window.lower() == "true")
    if args.sliding_window_size is not None:
        config.sliding_window_size = args.sliding_window_size
    if args.rope_base is not None:
        config.rope_base = args.rope_base
    if args.n_kv_heads is not None:
        config.n_kv_heads = args.n_kv_heads
    if args.ffn_variant is not None:
        config.ffn_variant = args.ffn_variant
    if args.use_layernorm is not None:
        config.use_layernorm = (args.use_layernorm.lower() == "true")
    if args.use_post_norm is not None:
        config.use_post_norm = (args.use_post_norm.lower() == "true")
    if args.use_tied_qk is not None:
        config.use_tied_qk = (args.use_tied_qk.lower() == "true")
    if args.use_mla is not None:
        config.use_mla = (args.use_mla.lower() == "true")
    if args.mla_latent_dim is not None:
        config.mla_latent_dim = args.mla_latent_dim
    if args.use_linear_attn is not None:
        config.use_linear_attn = (args.use_linear_attn.lower() == "true")
    if args.logit_softcap is not None:
        config.logit_softcap = args.logit_softcap
    if args.attention_dilation is not None:
        config.attention_dilation = args.attention_dilation
    if args.global_attn_every_k is not None:
        config.global_attn_every_k = args.global_attn_every_k
    if args.use_diff_attn is not None:
        config.use_diff_attn = (args.use_diff_attn.lower() == "true")
    if args.use_nsa_global is not None:
        config.use_nsa_global = (args.use_nsa_global.lower() == "true")
    if args.nsa_block is not None:
        config.nsa_block = args.nsa_block
    if args.use_hybrid_heads is not None:
        config.use_hybrid_heads = (args.use_hybrid_heads.lower() == "true")
    if args.d_ff is not None:
        config.d_ff = args.d_ff
    if args.norm_type is not None:
        config.norm_type = args.norm_type.lower()
    if args.qk_norm_type is not None:
        config.qk_norm_type = args.qk_norm_type.lower()
    if args.v_norm_type is not None:
        config.v_norm_type = args.v_norm_type.lower()
    if args.use_multiscale_heads is not None:
        config.use_multiscale_heads = (args.use_multiscale_heads.lower() == "true")
    if args.use_parallel_block is not None:
        config.use_parallel_block = (args.use_parallel_block.lower() == "true")
    if args.use_attn_sink is not None:
        config.use_attn_sink = (args.use_attn_sink.lower() == "true")
    if args.use_unet_skips is not None:
        config.use_unet_skips = (args.use_unet_skips.lower() == "true")
    if args.unet_skip_count is not None:
        config.unet_skip_count = args.unet_skip_count
    if args.unet_gate_type is not None:
        config.unet_gate_type = args.unet_gate_type
    if args.unet_gate_init is not None:
        config.unet_gate_init = args.unet_gate_init
    if args.use_smear_gate is not None:
        config.use_smear_gate = (args.use_smear_gate.lower() == "true")
    if args.use_attn_output_gate is not None:
        config.use_attn_output_gate = (args.use_attn_output_gate.lower() == "true")
    if args.use_layerscale is not None:
        config.use_layerscale = (args.use_layerscale.lower() == "true")
    if args.compile is not None:
        config.compile_model = (args.compile.lower() == "true")
    config.device = args.device
    if args.stop_at_step is not None:
        config.stop_at_step = args.stop_at_step
    if args.eval_every is not None:
        config.eval_every = args.eval_every
    if args.save_every is not None:
        config.save_every = args.save_every
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.log_every is not None:
        config.log_every = args.log_every
    
    # Define custom milestones for validation curves and autosetup logging.
    # These are denser than the old schedule so future runs produce better
    # learning-curve plots without changing any finished runs.
    config.eval_milestones = build_eval_milestones(config.train_tokens)
    if config.train_tokens <= 8_000_000:
        config.log_every = 25
    elif config.train_tokens <= 25_000_000:
        config.log_every = 50
    elif config.train_tokens <= 50_000_000:
        config.log_every = 100
    elif config.train_tokens <= 100_000_000:
        config.log_every = 250
    else:
        config.log_every = 1000
    config.eval_every = None
    
    # Allow command line override ONLY if explicitly provided (argparse default check)
    if args.log_every != 100: # 100 is the default in parser
        config.log_every = args.log_every
    
    use_warmup = (args.warmup.lower() == "true")

    
    output_dir = args.output_dir

    # Calculate required documents dynamically
    # Assume avg 1000 tokens per doc (conservative estimate)
    # Safety factor 2.0 to ensure enough data
    avg_tokens_per_doc = 1000
    safety_factor = 2.0
    total_tokens_needed = config.train_tokens
    calc_num_docs = int((total_tokens_needed / avg_tokens_per_doc) * safety_factor)
    
    # For very short runs (debugging), we verify we have at least some docs.
    if calc_num_docs < 100:
        calc_num_docs = 100
        
    print(f"Dynamic Data Calculation:")
    print(f"  Batch: {config.batch_size}, Seq: {config.max_seq_len}, Accumulation: {config.gradient_accumulation_steps}")
    print(f"  Target tokens: {total_tokens_needed:,}")
    print(f"  Est. docs needed (factor {safety_factor}): {calc_num_docs:,}")
    
    num_docs = calc_num_docs

    print("Loading dataset with Hugging Face Datasets API...")
    data_cfg = DataConfig(
        dataset_path=args.dataset_path if args.dataset_path else "auto",
        seq_length=config.max_seq_len,
        num_samples=num_docs,
        cache_dir="./hf_cache",
    )
    
    # Show which dataset was resolved (especially useful for auto-detection)
    if not args.dataset_path:
        print(f"📂 Auto-detected dataset: {data_cfg.dataset_path}")

    from data.loader import setup_tokenizer
    
    # Setup tokenizer first to get vocab size
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size

    # Prepare datasets (handles caching automatically)
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)
    
    logger.info(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")

    # Generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(args.seed)

    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=resolve_device(config.device).type == "cuda",
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)


    print("\nModel configuration")
    print("-" * 70)
    print(f"d_model: {config.d_model}, layers: {config.n_layers}, heads: {config.n_heads}")
    print(f"ff dim: {config.d_ff}")
    print(f"device: {config.device} -> {describe_device(resolve_device(config.device))}")
    print(f"train tokens: {config.train_tokens:,}")
    print(f"batch size: {config.batch_size}")
    print(f"vocab size: {config.vocab_size}\n")
    logger.info(f"Model configuration: {vars(config)}")

    train_minimal_llm(
        config,
        train_loader,
        val_loader,
        output_dir=output_dir,
        load_weights_path=args.load_checkpoint,
        config_name=args.config_class or args.config,
        run_seed=args.seed,
    )


if __name__ == "__main__":
    main()
