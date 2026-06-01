import json
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from configs.llm_config import LLMConfig


def capture_rng_state() -> Dict[str, Any]:
    """Capture RNG state so a checkpoint can be resumed later."""
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    """Restore RNG state if a checkpoint stored it."""
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch") is not None:
        torch_state = state["torch"]
        if not isinstance(torch_state, torch.Tensor):
            torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
        torch.set_rng_state(torch_state.cpu().contiguous())
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        cuda_states = state["cuda"]
        normalized_cuda_states = []
        for cuda_state in cuda_states:
            if not isinstance(cuda_state, torch.Tensor):
                cuda_state = torch.as_tensor(cuda_state, dtype=torch.uint8)
            normalized_cuda_states.append(cuda_state.cpu().contiguous())
        torch.cuda.set_rng_state_all(normalized_cuda_states)


def capture_git_metadata() -> Dict[str, Any]:
    """Capture repo identity for reproducible artifacts."""
    def run_git(args: List[str]) -> Optional[str]:
        try:
            return subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception:
            return None

    dirty = None
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except Exception:
        pass

    return {
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_branch": run_git(["branch", "--show-current"]),
        "git_dirty": dirty,
    }


def model_state_dict_for_checkpoint(model: nn.Module) -> Dict[str, Any]:
    """Save unwrapped weights when torch.compile wraps the module."""
    return getattr(model, "_orig_mod", model).state_dict()


def save_training_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    config: LLMConfig,
    optimizers: List[torch.optim.Optimizer],
    schedulers: List,
    metrics: Dict[str, Any],
    step: int,
    tokens_seen: int,
    metrics_history: Dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'checkpoint_version': 2,
        'model_state_dict': model_state_dict_for_checkpoint(model),
        'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers],
        'scheduler_state_dicts': [scheduler.state_dict() for scheduler in schedulers],
        'config': config,
        'metrics': metrics,
        'step': step,
        'tokens_seen': tokens_seen,
        'metrics_history': metrics_history,
        'rng_state': capture_rng_state(),
        'git_metadata': capture_git_metadata(),
    }, checkpoint_path)
    print(f"   💾 Checkpoint saved to {checkpoint_path}")
