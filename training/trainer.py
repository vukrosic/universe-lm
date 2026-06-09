import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Optional, Callable, Dict, Any
from configs.llm_config import LLMConfig
from models.llm import MinimalLLM
from optimizers.muon import Muon
from optimizers.cautious_adamw import CautiousAdamW
from optimizers.soap import SOAP
from training.checkpointing import (
    capture_git_metadata,
    capture_rng_state,
    restore_rng_state,
    save_training_checkpoint,
)
from training.device import resolve_device
from training.evaluation import evaluate_model
from utils.helpers import set_seed, format_time


class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience: int = 30, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_step = 0
        
    def __call__(self, val_loss: float, step: int) -> bool:
        """Returns True if training should stop"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_step = step
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\n⏹️  Early stopping triggered at step {step}")
                print(f"   Best loss: {self.best_loss:.4f} at step {self.best_step}")
                return True
            return False


def load_exact_step_baseline(config: LLMConfig) -> Dict[int, float]:
    """Load exact-step baseline values for log-only comparisons."""
    if config.train_tokens < 150_000_000:
        return {}
    baseline_path = Path("baselines/10m_baseline.json")
    if not baseline_path.exists():
        return {}
    with open(baseline_path) as f:
        baseline = json.load(f)
    return dict(zip(baseline.get("steps", []), baseline.get("val_losses", [])))


def default_metrics_history() -> Dict[str, list]:
    return {
        "steps": [],
        "val_losses": [],
        "val_accuracies": [],
        "val_perplexities": [],
        "elapsed_times": [],
        "learning_rates": [],
    }


def setup_muon_optimizer(model: nn.Module, config: LLMConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []
    soap_params = []

    for name, param in model.named_parameters():
        is_muon_candidate = (
            param.ndim == 2
            and 'token_embedding' not in name
            and 'norm' not in name
            and param.requires_grad
        )
        # R1 lever — MuonFor1DNorm: route 1-D norm params (e.g. `norm.weight`)
        # to Muon instead of AdamW. Default off → step-0 identical to baseline.
        # See docs/research/optimizer_routing/plan.md.
        if (
            getattr(config, "muon_for_1d_norm", False)
            and param.ndim == 1
            and 'norm' in name
            and param.requires_grad
        ):
            is_muon_candidate = True
        # R2 lever — MuonForEmbed: route `token_embedding` (and `emb_proj`,
        # since they're related) to Muon instead of AdamW. The embedding is
        # ~91% of params at vocab=50k — does it want orthogonalized updates?
        # Default off → step-0 identical to baseline. See
        # docs/research/optimizer_routing/plan.md.
        if (
            getattr(config, "muon_for_embed", False)
            and ('token_embedding' in name or 'emb_proj' in name)
            and param.ndim == 2
            and param.requires_grad
        ):
            is_muon_candidate = True
        # R3 lever — MuonForOutput: route the attention output projection
        # (`out_proj` / `W_O`) to Muon instead of AdamW. The output projection
        # is 2-D but might be deliberately kept in AdamW. Default off →
        # step-0 identical to baseline. See docs/research/optimizer_routing/plan.md.
        if (
            getattr(config, "muon_for_output", False)
            and 'out_proj' in name
            and param.ndim == 2
            and param.requires_grad
        ):
            is_muon_candidate = True
        # R4 lever — SOAP: route 2-D non-Muon AdamW params to the SOAP
        # optimizer (Adam in Shampoo's eigenbasis). Only 2-D params
        # (`token_embedding`, `emb_proj`, `out_proj`) — 1-D scalars
        # and `*.norm.weight` stay on plain AdamW (eigendecomp is
        # meaningless on 1-D). Default off → step-0 identical to
        # baseline. See autoresearch/ideas/003-soap/plan.md.
        if (
            getattr(config, "use_soap", False)
            and param.ndim == 2
            and not is_muon_candidate
            and param.requires_grad
        ):
            soap_params.append(param)
            continue
        if is_muon_candidate:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")
    print(f"  SOAP parameters: {sum(p.numel() for p in soap_params):,}")

    muon_optimizer = Muon(
        muon_params,
        lr=config.muon_lr,
        momentum=config.muon_momentum,
        ns_steps=getattr(config, "muon_ns_steps", 5),
        orthogonalize=getattr(config, "muon_orthogonalize", True),
        coeffs_mode=getattr(config, "muon_coeffs_mode", "polar_express"),
        shape_scale=getattr(config, "muon_shape_scale", True),
        scale_mode=getattr(config, "muon_scale_mode", "shape_aspect"),
        adamw_lr=getattr(config, "adamw_lr", 0.006),
        nesterov=getattr(config, "muon_nesterov", True),
        lazy_ortho_steps=getattr(config, "muon_lazy_ortho_steps", 1),
        cautious=getattr(config, "use_cautious_muon", False),
    )
    device = resolve_device(getattr(config, "device", "auto"))
    # Cautious-AdamW gate (Liang et al. 2024) — see
    # autoresearch/ideas/002-cautious-adamw/plan.md. "none" = baseline
    # `torch.optim.AdamW` (bit-identical to today); other values select
    # which AdamW bucket(s) the sign-mask fires on.
    _cautious_mode = getattr(config, "use_cautious_adamw", "none")
    if _cautious_mode != "none":
        if _cautious_mode == "embedding":
            _mask_buckets = ("token_embedding", "emb_proj")
        elif _cautious_mode == "gain":
            _mask_buckets = ("norm.weight",)
        elif _cautious_mode == "all":
            _mask_buckets = ()
            _mask_all = True
        else:
            raise ValueError(f"Unknown use_cautious_adamw: {_cautious_mode!r}")
        if _cautious_mode != "all":
            _mask_all = False
        adamw_optimizer = CautiousAdamW(
            adamw_params,
            mask_buckets=_mask_buckets,
            mask_all=_mask_all,
            lr=config.adamw_lr,
            weight_decay=config.weight_decay,
            fused=device.type == "cuda",
        )
    else:
        adamw_optimizer = torch.optim.AdamW(
            adamw_params,
            lr=config.adamw_lr,
            weight_decay=config.weight_decay,
            fused=device.type == "cuda",
        )

    optimizers = [muon_optimizer, adamw_optimizer]
    # SOAP gate (Vyas et al. 2024) — see
    # autoresearch/ideas/003-soap/plan.md. `use_soap=True` swaps the
    # eligible 2-D params (`token_embedding`, `emb_proj`, `out_proj`)
    # to a SOAP optimizer; 1-D params stay on the existing AdamW.
    # `use_soap=False` (default) → `soap_params=[]` → no SOAP
    # optimizer instantiated, bit-identical to baseline.
    if getattr(config, "use_soap", False) and soap_params:
        soap_optimizer = SOAP(
            soap_params,
            lr=config.adamw_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay,
            precondition_frequency=getattr(
                config, "use_soap_precondition_freq", 10),
            precondition_eps=1e-6,
        )
        optimizers.append(soap_optimizer)

    return optimizers


def train_model(
    model: nn.Module,
    config: LLMConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizers: List[torch.optim.Optimizer],
    schedulers: Optional[List] = None,
    checkpoint_state: Optional[Dict[str, Any]] = None,
    early_stopper: Optional[EarlyStopping] = None,
    output_dir: Optional[str] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    log_every: int = 100,
) -> Any:
    """
    Generic training function that can be used by experiments.
    
    Args:
        model: Model to train
        config: Model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizers: List of optimizers
        schedulers: Optional list of learning rate schedulers
        early_stopper: Optional early stopping handler
        output_dir: Optional directory to save outputs
        extra_config: Optional dict of extra config to save with metrics
    
    Returns:
        model, final_metrics, metrics_history
    """
    device = resolve_device(getattr(config, "device", "auto"))
    model = model.to(device, dtype=torch.bfloat16) if device.type == "cuda" else model.to(device)
    
    if schedulers is None:
        schedulers = []

    current_loss_val = float(checkpoint_state.get("metrics", {}).get("train_loss", 0.0)) if checkpoint_state else 0.0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_start_time = time.time()
    metrics_history = default_metrics_history()
    if checkpoint_state and checkpoint_state.get("metrics_history"):
        loaded_history = checkpoint_state["metrics_history"]
        metrics_history = {
            "steps": list(loaded_history.get("steps", [])),
            "val_losses": list(loaded_history.get("val_losses", [])),
            "val_accuracies": list(loaded_history.get("val_accuracies", [])),
            "val_perplexities": list(loaded_history.get("val_perplexities", [])),
            "elapsed_times": list(loaded_history.get("elapsed_times", [])),
            "learning_rates": list(loaded_history.get("learning_rates", [])),
        }

    # Training loop
    model.train()
    step = int(checkpoint_state.get("step", 0)) if checkpoint_state else 0
    tokens_seen = int(checkpoint_state.get("tokens_seen", 0)) if checkpoint_state else 0
    desc = "Training"
    pbar = tqdm(total=config.train_tokens, desc=desc, unit="tokens", initial=min(tokens_seen, config.train_tokens))
    
    stopped_early = False
    gated = False  # set True when we pause at a pre-registered gate step for human review
    gate_step = getattr(config, "stop_at_step", None)
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
    exact_step_baseline = load_exact_step_baseline(config)

    def write_live_metrics(current_metrics: Dict[str, Any], current_step: int, current_tokens: int) -> None:
        if not output_path:
            return
        live_metrics = {
            'final_metrics': current_metrics,
            'total_time_minutes': (time.time() - train_start_time) / 60,
            'stopped_early': stopped_early,
            'actual_steps': current_step,
            'tokens_seen': current_tokens,
            'history': metrics_history,
            # #65 self-identifying run (forward-only; live writes
            # also stamp identity so the latest .json always answers
            # "which run am I?" even mid-training)
            'run_name': output_path.name,
            'config_name': config.__class__.__name__,
            'seed': getattr(config, 'seed', None),
            'flags': config.active_flags(),
            **capture_git_metadata(),
        }
        if extra_config:
            live_metrics['experiment_config'] = extra_config
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(live_metrics, f, indent=2)

    while tokens_seen < config.train_tokens:
        for batch_idx, batch in enumerate(train_loader):
            if tokens_seen >= config.train_tokens:
                break

            # Handle different batch formats
            if isinstance(batch, dict):
                x = batch["input_ids"]
                y = batch["labels"]
                attention_mask = batch.get("attention_mask")
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    x, attention_mask, y = batch
                elif len(batch) == 2:
                    x, y = batch
                    attention_mask = None
                else:
                    raise ValueError(f"Unexpected batch structure with {len(batch)} elements.")
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            x, y = x.to(device), y.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Count tokens in this batch (approx: batch_size * seq_len)
            batch_tokens = x.numel()

            # A10 EntropyReg aux loss: collect stashed terms from every
            # MultiHeadAttention that computed one this forward. Sum across
            # layers; treat λ as a per-layer coefficient (tune for depth).
            # Default (no flag set): no module sets _entropy_reg_loss, so
            # this stays 0 and CE behavior is unchanged.
            def _collect_entropy_reg(m):
                total = next(m.parameters()).new_zeros(())
                for block in m.transformer_blocks:
                    term = getattr(block.attention, "_entropy_reg_loss", None)
                    if term is not None:
                        total = total + term
                        del block.attention._entropy_reg_loss  # avoid double-count next step
                return total

            # OH1 ZLoss (output-head ablation #1): aux term = λ·mean(logsumexp(logits)²).
            # Train-only; eval stays plain CE (see reporting rule in
            # docs/research/output_head/plan.md). λ=0 → no-op. Uses
            # getattr so the base LLMConfig is unchanged.
            z_loss_lambda = getattr(config, "z_loss_lambda", 0.0)
            use_z_loss = getattr(config, "use_z_loss", False) and z_loss_lambda > 0.0

            # OH2 LabelSmooth (output-head ablation #2): CE with smoothing ε.
            # Train-only; eval stays plain CE (see reporting rule in
            # docs/research/output_head/plan.md). ε=0 → no-op (PyTorch treats
            # 0.0 as no smoothing, but the conditional is cleaner and keeps
            # the call site byte-identical when the flag is off / absent).
            label_smooth = getattr(config, "label_smooth", 0.0)

            # Forward pass (optimized to avoid large contiguous copies of logits)
            if config.use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    # Shift labels instead of logits to save ~3GB VRAM
                    # We set the last token to -100 so cross_entropy ignores it
                    shift_labels = torch.full_like(y, -100)
                    shift_labels[:, :-1] = y[:, 1:]

                    ce_loss = F.cross_entropy(
                        logits.view(-1, config.vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        label_smoothing=label_smooth if label_smooth > 0 else 0.0,
                    )
                    entropy_reg_loss = _collect_entropy_reg(model)
                    z_loss = (
                        z_loss_lambda * (logits.logsumexp(dim=-1) ** 2).mean()
                        if use_z_loss
                        else logits.new_zeros(())
                    )
                    # OH3 ConfPenalty (output-head ablation #3): aux term = -β·H(softmax(logits)).
                    # Anti-overconfidence regularizer. Train-only; eval stays plain CE
                    # (see reporting rule in docs/research/output_head/plan.md).
                    # β=0 → no-op (zero scalar added). Uses getattr so the base
                    # LLMConfig is unchanged.
                    conf_penalty_beta = getattr(config, "conf_penalty_beta", 0.0)
                    if conf_penalty_beta > 0:
                        probs = logits.float().softmax(-1)
                        ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
                        conf_penalty = -conf_penalty_beta * ent
                    else:
                        conf_penalty = logits.new_zeros(())
                    loss = (ce_loss + entropy_reg_loss + z_loss + conf_penalty) / config.gradient_accumulation_steps
                loss.backward()
            else:
                logits = model(x)
                shift_labels = torch.full_like(y, -100)
                shift_labels[:, :-1] = y[:, 1:]

                ce_loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=label_smooth if label_smooth > 0 else 0.0,
                )
                entropy_reg_loss = _collect_entropy_reg(model)
                z_loss = (
                    z_loss_lambda * (logits.logsumexp(dim=-1) ** 2).mean()
                    if use_z_loss
                    else logits.new_zeros(())
                )
                # OH3 ConfPenalty (output-head ablation #3): aux term = -β·H(softmax(logits)).
                # Anti-overconfidence regularizer. Train-only; eval stays plain CE
                # (see reporting rule in docs/research/output_head/plan.md).
                # β=0 → no-op (zero scalar added). Uses getattr so the base
                # LLMConfig is unchanged.
                conf_penalty_beta = getattr(config, "conf_penalty_beta", 0.0)
                if conf_penalty_beta > 0:
                    probs = logits.float().softmax(-1)
                    ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
                    conf_penalty = -conf_penalty_beta * ent
                else:
                    conf_penalty = logits.new_zeros(())
                loss = (ce_loss + entropy_reg_loss + z_loss + conf_penalty) / config.gradient_accumulation_steps
                loss.backward()

            # Detach z-loss to a python float for logging only
            z_loss_val = z_loss.detach().item()

            # Detach entropy reg to a python float for logging only (graph no longer needed)
            entropy_reg_val = entropy_reg_loss.detach().item()

            # Detach conf penalty to a python float for logging only (graph no longer needed)
            conf_penalty_val = conf_penalty.detach().item()

            # Optimizer step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
                for scheduler in schedulers:
                    scheduler.step()

            # Track current loss as a scalar only every 100 steps to avoid sync bottleneck
            if step % 100 == 0 or step == 0:
                current_loss_val = ce_loss.item()
                
            # Logging
            if step % log_every == 0 or stopped_early:
                with torch.no_grad():
                    # Calculate accuracy using the shifted labels mask
                    predictions = logits.argmax(dim=-1)
                    mask = (shift_labels != -100)
                    accuracy = (predictions[mask] == shift_labels[mask]).float().mean().item()
                    
                    # Use the scalar value we polled above
                    perplexity = math.exp(min(current_loss_val if 'current_loss_val' in locals() else ce_loss.item(), 20))
                    current_lr = schedulers[0].get_last_lr()[0] if schedulers else optimizers[0].param_groups[0]['lr']

                # Update progress bar
                tokens_per_step = config.batch_size * config.max_seq_len * config.gradient_accumulation_steps
                est_total_steps = config.train_tokens // tokens_per_step
                
                pbar.set_postfix({
                    'step': f'{step}/{est_total_steps}',
                    'loss': f'{current_loss_val:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ent': f'{entropy_reg_val:+.2e}',
                    'cp': f'{conf_penalty_val:+.2e}',
                    'lr': f'{current_lr:.5f}'
                })
                # Console print for visibility
                if step % (log_every * 10) == 0 or stopped_early:
                    print(f" [Step {step}] Loss: {current_loss_val:.4f} | EntReg: {entropy_reg_val:+.2e} | ConfPen: {conf_penalty_val:+.2e} | Acc: {accuracy:.3f} | LR: {current_lr:.6f}")
            
            pbar.update(batch_tokens)
            tokens_seen += batch_tokens

            if stopped_early:
                current_loss_val = ce_loss.item()
                break

            # Evaluation
            is_milestone = False
            if config.eval_milestones and step in config.eval_milestones:
                is_milestone = True
            elif config.eval_every is not None and step % config.eval_every == 0 and step > 0:
                is_milestone = True

            if is_milestone:
                eval_metrics = evaluate_model(model, val_loader, config)
                elapsed_time = (time.time() - train_start_time) / 60
                current_lr = schedulers[0].get_last_lr()[0] if schedulers else optimizers[0].param_groups[0]['lr']
                
                # Track metrics
                metrics_history['steps'].append(step)
                metrics_history['val_losses'].append(eval_metrics['val_loss'])
                metrics_history['val_accuracies'].append(eval_metrics['val_accuracy'])
                metrics_history['val_perplexities'].append(eval_metrics['val_perplexity'])
                metrics_history['elapsed_times'].append(elapsed_time)
                metrics_history['learning_rates'].append(current_lr)
                
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}, "
                      f"LR: {current_lr:.5f}")
                if step in exact_step_baseline:
                    baseline_loss = exact_step_baseline[step]
                    delta = eval_metrics['val_loss'] - baseline_loss
                    print(f"   Baseline@step {step}: {baseline_loss:.4f} | delta: {delta:+.4f}")

                live_metrics = {**eval_metrics, 'train_loss': current_loss_val, 'train/conf_penalty_loss': conf_penalty_val}
                write_live_metrics(live_metrics, step, tokens_seen)
                if output_path:
                    save_training_checkpoint(
                        output_path / "latest.pt",
                        model,
                        config,
                        optimizers,
                        schedulers,
                        live_metrics,
                        step,
                        tokens_seen,
                        metrics_history,
                    )
                    save_training_checkpoint(
                        output_path / f"checkpoint_step_{step}.pt",
                        model,
                        config,
                        optimizers,
                        schedulers,
                        live_metrics,
                        step,
                        tokens_seen,
                        metrics_history,
                    )
                
                # Early stopping check
                if early_stopper is not None:
                    if early_stopper(eval_metrics['val_loss'], step):
                        current_loss_val = ce_loss.item()
                        stopped_early = True
                        break

                # Gate stop: pause the run at a pre-registered step so a human can
                # judge the curve before spending the rest of the compute. We only
                # gate on a milestone (we just evaluated + checkpointed above), so
                # gate.pt is a full, resumable state with a fresh eval point.
                if gate_step is not None and step >= gate_step:
                    if output_path:
                        save_training_checkpoint(
                            output_path / "gate.pt",
                            model,
                            config,
                            optimizers,
                            schedulers,
                            live_metrics,
                            step,
                            tokens_seen,
                            metrics_history,
                        )
                        print(f"\n⏸️  Gate reached: stop_at_step={gate_step} (paused at step {step}). "
                              f"Saved resumable checkpoint for human review.\n"
                              f"    Resume the SAME run with: --load_checkpoint {output_path / 'gate.pt'}")
                    else:
                        print(f"\n⏸️  Gate reached at step {step}, but no --output_dir set, so no checkpoint was saved.")
                    gated = True
                    break

            step += 1
        
        # If we finished the inner loop but didn't stop early, 
        # ensure we have the most recent loss from the very last batch
        if not stopped_early and 'ce_loss' in locals():
            current_loss_val = ce_loss.item()

        if stopped_early or gated:
            break

    pbar.close()

    # Final evaluation (if not stopped early)
    if (not stopped_early and not gated) or tokens_seen >= config.train_tokens:
        # If the last in-loop milestone already evaluated these exact weights
        # (last completed step == step - 1, no optimizer step since), reuse that
        # result instead of re-evaluating — avoids a duplicate final data point.
        if metrics_history['steps'] and metrics_history['steps'][-1] == step - 1:
            final_eval = {
                'val_loss': metrics_history['val_losses'][-1],
                'val_accuracy': metrics_history['val_accuracies'][-1],
                'val_perplexity': metrics_history['val_perplexities'][-1],
                'train_loss': current_loss_val,
            }
        else:
            final_eval = evaluate_model(model, val_loader, config)
            final_eval['train_loss'] = current_loss_val
            elapsed_time = (time.time() - train_start_time) / 60
            current_lr = schedulers[0].get_last_lr()[0] if schedulers else optimizers[0].param_groups[0]['lr']

            metrics_history['steps'].append(step)
            metrics_history['val_losses'].append(final_eval['val_loss'])
            metrics_history['val_accuracies'].append(final_eval['val_accuracy'])
            metrics_history['val_perplexities'].append(final_eval['val_perplexity'])
            metrics_history['elapsed_times'].append(elapsed_time)
            metrics_history['learning_rates'].append(current_lr)
    else:
        # Use best metrics if stopped early
        if metrics_history['val_losses']:
            best_idx = metrics_history['val_losses'].index(min(metrics_history['val_losses']))
            final_eval = {
                'val_loss': metrics_history['val_losses'][best_idx],
                'val_accuracy': metrics_history['val_accuracies'][best_idx],
                'val_perplexity': metrics_history['val_perplexities'][best_idx],
                'train_loss': current_loss_val if 'current_loss_val' in locals() else 0.0,
            }
        else:
            final_eval = {
                'val_loss': current_loss_val if 'current_loss_val' in locals() else 0.0,
                'val_accuracy': accuracy if 'accuracy' in locals() else 0.0,
                'val_perplexity': perplexity if 'perplexity' in locals() else 0.0,
                'train_loss': current_loss_val if 'current_loss_val' in locals() else 0.0,
            }
    
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_time_seconds = time.time() - train_start_time
    
    if stopped_early:
        print(f"   ⚠️  Training stopped early at step {step}")
    
    # Save outputs if directory specified
    if output_dir:
        # Save metrics
        metrics_file = output_path / "metrics.json"
        metrics_data = {
            'final_metrics': final_eval,
            'total_time_minutes': total_time_seconds / 60,
            'stopped_early': stopped_early,
            'actual_steps': step,
            'history': metrics_history,
            **capture_git_metadata(),
        }
        if extra_config:
            metrics_data['experiment_config'] = extra_config
            
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"   📁 Metrics saved to {metrics_file}")
        
        
        # Save model checkpoint
        checkpoint_path = output_path / "model.pt"
        save_training_checkpoint(
            checkpoint_path,
            model,
            config,
            optimizers,
            schedulers,
            final_eval,
            step,
            tokens_seen,
            metrics_history,
        )
    
    return {
        'model': model,
        'final_metrics': final_eval,
        'metrics_history': metrics_history,
        'training_time': total_time_seconds,
        'steps': step,
        'tokens_seen': tokens_seen,
        'train_loss': current_loss_val if 'current_loss_val' in locals() else 0.0,
        'gated': gated,
    }



def warmup_compiled_kernels(
    model: nn.Module,
    config: LLMConfig,
    train_loader: DataLoader,
    device: torch.device,
    num_steps: int = 3
) -> None:
    """
    Warm up all compiled kernels (forward, backward, optimizer).
    Caller is responsible for resetting state afterwards.
    """
    print(f"🔥 Warming up kernels ({num_steps} steps)...")
    model.train()
    
    # Temporary optimizer to warm up optimizer kernels too
    temp_optimizers = setup_muon_optimizer(model, config)
    
    warmup_iter = iter(train_loader)
    
    for _ in range(num_steps):
        try:
            batch = next(warmup_iter)
        except StopIteration:
            warmup_iter = iter(train_loader)
            batch = next(warmup_iter)
        
        # Parse batch
        if isinstance(batch, dict):
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
        else:
            x, y = batch[0].to(device), batch[-1].to(device)
        
        # Forward + Backward
        if config.use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, config.vocab_size),
                    y[:, 1:].reshape(-1)
                )
            loss.backward()
        else:
            logits = model(x)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, config.vocab_size),
                y[:, 1:].reshape(-1)
            )
            loss.backward()
        
        # Optimizer step (warms up optimizer kernels)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        for opt in temp_optimizers:
            opt.step()
            opt.zero_grad()
    
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    
    # Cleanup temp optimizers
    del temp_optimizers
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print("✅ Kernels compiled and cached")

def train_minimal_llm(
    config: LLMConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Optional[str] = None,
    load_weights_path: Optional[str] = None,
    compare_baseline: bool = False,
    config_name: Optional[str] = None,
    run_seed: Optional[int] = None,
):
    print(f"\n🚀 Training dense model")
    setup_start = time.time()
    device = resolve_device(getattr(config, "device", "auto"))
    checkpoint_payload = None

    # ============================================
    # 1. Initialize model with the configured seed (varies init when --seed set)
    # ============================================
    set_seed(getattr(config, "seed", 42))
    model = MinimalLLM(config)
    model = model.to(device, dtype=torch.bfloat16) if device.type == "cuda" else model.to(device)
    
    # Load pretrained weights if specified
    if load_weights_path:
        print(f"Loading pretrained weights from {load_weights_path}...")
        checkpoint_payload = torch.load(load_weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
        model.load_state_dict(state_dict, strict=False)

    # ============================================
    # 2. Save initial state BEFORE any forward pass (cloned to CPU)
    # ============================================
    initial_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")

    # ============================================
    # 3. Compile model (if requested)
    # ============================================
    if config.compile_model:
        print("🚀 Compiling model with torch.compile...")
        # Keep a reference to the original model for state restoration
        orig_model = model
        try:
            model = torch.compile(model)
            print("✅ Model compiled successfully")
            
            # ============================================
            # 4. Warm up kernels (dirties model state)
            # ============================================
            warmup_compiled_kernels(model, config, train_loader, device, num_steps=3)
            
            # ============================================
            # 5. Reset model to initial state
            # ============================================
            # Restore state ensuring we use the original model keys to avoid calling load_state_dict on the wrapper
            orig_model.load_state_dict(initial_model_state)
            print("🔄 Model weights reset to initial state")
            
        except Exception as e:
            print(f"⚠️ Compilation failed: {e}")
            print("Continuing in eager mode.")
            # Fallback to original model
            model = orig_model
            # Ensure state is clean
            model.load_state_dict(initial_model_state)
    
    # Free the backup
    del initial_model_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ============================================
    # 6. Create FRESH optimizers (no accumulated state)
    # ============================================
    optimizers = setup_muon_optimizer(model, config)

    # ============================================
    # 7. Create FRESH schedulers
    # ============================================
    # Tokens per optimization step
    tokens_per_opt = config.batch_size * config.max_seq_len * config.gradient_accumulation_steps
    total_steps = config.train_tokens // tokens_per_opt
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    schedule_type = getattr(config, 'schedule_type', 'cosine')
    
    schedulers = []
    for optimizer in optimizers:
        if schedule_type == 'cosine':
            def lr_lambda(current_step, warmup=warmup_steps, total=total_steps):
                if current_step < warmup:
                    return current_step / warmup
                progress = (current_step - warmup) / max(1, total - warmup)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        elif schedule_type == 'linear':
            def lr_lambda(current_step, warmup=warmup_steps, total=total_steps):
                if current_step < warmup:
                    return current_step / warmup
                progress = (current_step - warmup) / max(1, total - warmup)
                return max(0.1, 1.0 - progress)
        elif schedule_type == 'warmup_decay_to_zero':
            def lr_lambda(current_step, warmup=warmup_steps, total=total_steps):
                if current_step < warmup:
                    return current_step / warmup
                progress = (current_step - warmup) / max(1, total - warmup)
                return max(0.0, 1.0 - progress)
        else:  # constant
            def lr_lambda(current_step, warmup=warmup_steps):
                return current_step / warmup if current_step < warmup else 1.0
        
        schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda))

    if checkpoint_payload and "optimizer_state_dicts" in checkpoint_payload:
        optimizer_states = checkpoint_payload.get("optimizer_state_dicts", [])
        scheduler_states = checkpoint_payload.get("scheduler_state_dicts", [])
        for optimizer, state_dict in zip(optimizers, optimizer_states):
            optimizer.load_state_dict(state_dict)
        for scheduler, state_dict in zip(schedulers, scheduler_states):
            scheduler.load_state_dict(state_dict)
        restore_rng_state(checkpoint_payload.get("rng_state"))

    # ============================================
    # 8. Reset RNG for reproducible training
    # ============================================
    if not (checkpoint_payload and "rng_state" in checkpoint_payload):
        set_seed(getattr(config, "seed", 42))
    
    setup_time = time.time() - setup_start
    print(f"⚙️ Setup & Compilation complete in {setup_time:.2f}s")
    print("-" * 70)

    # ============================================
    # 9. Train from scratch (fresh iterator created internally)
    # ============================================
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    train_start = time.time()
    
    results = train_model(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        checkpoint_state=checkpoint_payload,
        early_stopper=None,
        output_dir=output_dir,
        extra_config=None,
        log_every=getattr(config, 'log_every', 100),
    )
    
    total_training_time = results['training_time']
    total_wall_time = setup_time + total_training_time
    final_eval = results['final_metrics']
    metrics_history = results['metrics_history']
    step = results['steps']
    tokens_seen = results['tokens_seen']
    gated = results.get('gated', False)

    # ============================================
    # 10. Unified Saving & Reporting
    # ============================================
    # Save results to plots/ directory with timestamped names
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_dir = Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filenames
    base_filename = f"{config.train_tokens}_{timestamp}"
    metrics_file = plot_dir / f"metrics_{base_filename}.json"
    plot_file = plot_dir / f"val_loss_{base_filename}.png"
    output_path = Path(output_dir) if output_dir else None
    
    # Save comprehensive metrics to plots/
    metrics_data = {
        'final_metrics': final_eval,
        'setup_time_seconds': setup_time,
        'active_training_time_seconds': total_training_time,
        'total_wall_time_seconds': total_wall_time,
        'total_time_minutes': total_wall_time / 60,
        'actual_steps': step,
        'tokens_seen': tokens_seen,
        'train_tokens': config.train_tokens,
        'gated': gated,
        'history': metrics_history,
        # #65 self-identifying run: every metrics.json answers
        # "which run am I?" on its own. DESC in
        # runs/make_evidence_index.py is the curated prose;
        # these fields are the structured truth.
        'run_name': output_path.name if output_path else None,
        'config_name': config_name or config.__class__.__name__,
        'seed': run_seed if run_seed is not None else getattr(config, 'seed', None),
        'flags': config.active_flags(),
        **capture_git_metadata(),
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"   📊 Metrics saved to {metrics_file}")
        
    try:
        from utils.plot_loss import plot_loss
        
        baseline_file = None
        if compare_baseline:
            # Determine closest baseline file based on token count
            known_baselines = {
                8_000_000: "plots/8M.json",
                20_000_000: "plots/20M.json",
                100_000_000: "plots/100M.json"
            }
            
            # Find closest baseline
            closest_tokens = min(known_baselines.keys(), key=lambda x: abs(x - config.train_tokens))
            baseline_file = known_baselines[closest_tokens]
            
            # Verify it exists
            if not os.path.exists(baseline_file):
                print(f"      (Baseline file {baseline_file} not found locally)")
                baseline_file = None
            
        plot_loss(
            str(metrics_file), 
            str(plot_file), 
            title=f"Validation Loss - {config.train_tokens:,} Tokens",
            baseline_file=baseline_file
        )
        print(f"   📈 Plot saved to {plot_file}")
        if baseline_file:
            print(f"      (Compared against baseline: {baseline_file})")
    except Exception as e:
        print(f"   ⚠️ Failed to generate plot: {e}")
    
    # Also save to output_dir if specified (for backward compatibility)
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics copy
        checkpoint_metrics = output_path / "metrics.json"
        with open(checkpoint_metrics, 'w') as f:
            json.dump(metrics_data, f, indent=2)
            
        # Save model
        checkpoint_path = output_path / "model.pt"
        save_training_checkpoint(
            checkpoint_path,
            results['model'],
            config,
            optimizers,
            schedulers,
            final_eval,
            step,
            tokens_seen,
            metrics_history,
        )
        
    
    # Final Output
    print("\n" + "="*70)
    print(" TRAINING RESULTS")
    print("="*70)
    print(f"Warmup & Setup:                  {format_time(setup_time)}")
    print(f"Active Training Time:            {format_time(total_training_time)}")
    print(f"Total Tokens:                    {tokens_seen:,}")
    print("-" * 70)
    print(f"Final Train Loss:                {final_eval.get('train_loss', 0.0):.4f}")
    print(f"Final Val Loss:                  {final_eval['val_loss']:.4f}")
    print(f"Final Val Accuracy:              {final_eval['val_accuracy']:.4f}")
    print("="*70 + "\n")

    return {
        'model': results['model'],
        'metrics': final_eval,
        'history': metrics_history,
        'setup_time': setup_time,
        'training_time': total_training_time,
        'steps': step,
        'tokens_seen': tokens_seen,
        'gated': gated,
    }
