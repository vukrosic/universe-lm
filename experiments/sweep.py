"""Sweep harness: run N variants of train_llm.py, log results, output CSV."""

import argparse
import copy
import csv
import json
import os
import sys
import time
import dataclasses
from pathlib import Path
from typing import Any

import yaml

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.llm_config import (
    LLMConfig,
    ResearchConfig,
    FastResearchConfig,
    UniverseSmokeConfig,
    FiveMillionConfig,
    TwentyFiveMillionConfig,
    FiftyMillionConfig,
    HundredMillionConfig,
    OneHundredThirtyFiveMillionConfig,
    FiveHundredMillionConfig,
    OneBillionConfig,
)
from data.loader import setup_tokenizer
from training.trainer import train_minimal_llm
from utils.helpers import set_seed


PRESET_MAP = {
    "default": LLMConfig,
    "research": ResearchConfig,
    "fast_research": FastResearchConfig,
    "smoke": UniverseSmokeConfig,
    "5m": FiveMillionConfig,
    "25m": TwentyFiveMillionConfig,
    "50m": FiftyMillionConfig,
    "100m": HundredMillionConfig,
    "135m": OneHundredThirtyFiveMillionConfig,
    "500m": FiveHundredMillionConfig,
    "1b": OneBillionConfig,
}


def resolve_config(base_preset: str, train_tokens: int, seed: int, variant_name: str, overrides: dict) -> LLMConfig:
    """Build a resolved config for one variant."""
    if base_preset not in PRESET_MAP:
        raise ValueError(f"Unknown preset '{base_preset}'. Available: {list(PRESET_MAP.keys())}")
    config = PRESET_MAP[base_preset]()
    config.train_tokens = train_tokens
    config.seed = seed
    for key, val in overrides.items():
        if hasattr(config, key):
            setattr(config, key, val)
        else:
            # Allow new flags (e.g. use_qk_norm) to be passed even if not in base config
            pass
    return config


def count_params(config: LLMConfig) -> int:
    """Approximate total parameters from config."""
    d_model = config.d_model
    n_layers = config.n_layers
    n_heads = config.n_heads
    d_ff = config.d_ff
    n_kv = config.n_kv_heads
    vocab = config.vocab_size

    embed = vocab * d_model
    layers = n_layers * (
        # Q projection: d_model -> d_model (but with n_heads active)
        d_model * d_model
        # K projection
        + d_model * d_model
        # V projection
        + d_model * d_model
        # O projection
        + d_model * d_model
        # FFN gate1
        + d_model * d_ff
        # FFN gate2
        + d_model * d_ff
        # FFN output (d_ff -> d_model is two matmuls so we count both)
    ) + embed  # token embedding
    # Note: this is approximate; rope and rmsnorm add negligible params
    return layers


def run_variant(
    config: LLMConfig,
    variant_name: str,
    sweep_name: str,
    dry_run: bool = False,
) -> dict:
    """Run one variant. Returns dict with final metrics."""

    variant_dir = Path("runs") / sweep_name / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    config_path = variant_dir / "config.json"
    config_dict = {
        "variant": variant_name,
        "sweep": sweep_name,
        **{
            f.name: getattr(config, f.name)
            for f in dataclasses.fields(config)
        },
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    if dry_run:
        print(f"\n  [DRY RUN] Would train variant: {variant_name}")
        print(f"    train_tokens={config.train_tokens:,}, seed={config.seed}")
        print(f"    d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")
        print(f"    overrides applied, config saved to {config_path}")
        return {"variant": variant_name, "skipped": True}

    # Build data config — reuse the same logic as train_llm.py
    from configs.dataset_config import DataConfig
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    avg_tokens_per_doc = 1000
    safety_factor = 2.0
    calc_num_docs = max(100, int((config.train_tokens / avg_tokens_per_doc) * safety_factor))

    data_cfg = DataConfig(
        dataset_path="auto",
        seq_length=config.max_seq_len,
        num_samples=calc_num_docs,
        cache_dir="./hf_cache",
    )

    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size

    # Prepare datasets (handles caching)
    from train_llm import prepare_datasets
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)

    # Build data loaders
    from torch.utils.data import DataLoader
    import torch

    g = torch.Generator()
    g.manual_seed(config.seed)

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        worker_init_fn=None,  # use default
        generator=g,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Run training
    set_seed(config.seed)
    start_time = time.time()

    result = train_minimal_llm(
        config,
        train_loader,
        val_loader,
        output_dir=str(variant_dir / "checkpoint"),
        load_weights_path=None,
        compare_baseline=False,
    )

    wall_time = time.time() - start_time
    final_metrics = result["metrics"]
    history = result["history"]

    # Write val_loss.csv from history
    val_loss_path = variant_dir / "val_loss.csv"
    with open(val_loss_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "val_loss", "val_ppl", "elapsed_min"])
        for i, step in enumerate(history.get("steps", [])):
            vl = history["val_losses"][i] if i < len(history.get("val_losses", [])) else 0
            vp = history["val_perplexities"][i] if i < len(history.get("val_perplexities", [])) else 0
            et = history["elapsed_times"][i] if i < len(history.get("elapsed_times", [])) else 0
            writer.writerow([step, vl, vp, et])

    # Write final_metrics.json
    final_metrics_path = variant_dir / "final_metrics.json"
    params = count_params(config)
    metrics_out = {
        **final_metrics,
        "wall_time_s": round(wall_time, 1),
        "params": params,
        "variant": variant_name,
        "sweep": sweep_name,
    }
    with open(final_metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    return {
        "variant": variant_name,
        "skipped": False,
        "final_val_loss": final_metrics["val_loss"],
        "val_ppl": final_metrics["val_perplexity"],
        "train_tokens": config.train_tokens,
        "wall_time_s": round(wall_time, 1),
        "params": params,
    }


def run_sweep(yaml_path: str, dry_run: bool = False) -> list[dict]:
    """Load sweep YAML and run all variants. Returns per-variant result dicts."""
    with open(yaml_path) as f:
        sweep_cfg = yaml.safe_load(f)

    sweep_name = Path(yaml_path).stem
    base_preset = sweep_cfg["base_preset"]
    train_tokens = sweep_cfg.get("train_tokens", 100_000_000)
    seed = sweep_cfg.get("seed", 42)
    variants = sweep_cfg.get("variants", [])

    print(f"\nSweep: {sweep_name}")
    print(f"  base_preset : {base_preset}")
    print(f"  train_tokens: {train_tokens:,}")
    print(f"  seed        : {seed}")
    print(f"  variants    : {len(variants)}")

    results = []
    for v in variants:
        name = v["name"]
        overrides = v.get("overrides", {})
        print(f"\n{'='*60}")
        print(f"Running variant: {name}")
        print(f"{'='*60}")

        resolved = resolve_config(base_preset, train_tokens, seed, name, overrides)
        result = run_variant(resolved, name, sweep_name, dry_run=dry_run)
        results.append(result)

        if not dry_run:
            print(f"\n  Result: val_loss={result['final_val_loss']:.4f}, "
                  f"ppl={result['val_ppl']:.2f}, "
                  f"wall_time={result['wall_time_s']:.0f}s")

    return results


def write_results_csv(results: list[dict], sweep_name: str) -> Path:
    """Write results CSV to experiments/results/<sweep>.csv."""
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sweep_name}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "final_val_loss", "ppl", "train_tokens", "wall_time_s", "params"],
        )
        writer.writeheader()
        for r in results:
            if not r.get("skipped"):
                writer.writerow({
                    "variant": r["variant"],
                    "final_val_loss": round(r["final_val_loss"], 6),
                    "ppl": round(r["val_ppl"], 4),
                    "train_tokens": r["train_tokens"],
                    "wall_time_s": r["wall_time_s"],
                    "params": r["params"],
                })

    return out_path


def print_table(results: list[dict]) -> None:
    """Print sorted results table."""
    rows = [r for r in results if not r.get("skipped")]
    rows.sort(key=lambda x: x["final_val_loss"])

    print("\n" + "=" * 70)
    print(f"{'variant':<20} {'final_val_loss':>15} {'ppl':>10} {'wall_time_s':>12} {'params':>12}")
    print("-" * 70)
    for r in rows:
        print(f"{r['variant']:<20} {r['final_val_loss']:>15.6f} {r['val_ppl']:>10.4f} "
              f"{r['wall_time_s']:>12.0f} {r['params']:>12,}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run config variant sweep")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to sweep YAML (e.g. experiments/sweeps/qk_norm.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned variants and resolved configs, don't train")

    args = parser.parse_args()

    results = run_sweep(args.config, dry_run=args.dry_run)

    if not args.dry_run:
        sweep_name = Path(args.config).stem
        csv_path = write_results_csv(results, sweep_name)
        print(f"\nResults CSV: {csv_path}")
        print_table(results)
    else:
        print("\n[DRY RUN] No training performed.")


if __name__ == "__main__":
    main()