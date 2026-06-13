# Vast AI Instance

Current Vast box for this repo, verified 2026-06-13:

```bash
ssh -L 8080:localhost:8080 -p 52649 root@1.208.108.242
```

The `-L 8080:localhost:8080` forward keeps any service you start on the remote
machine's port `8080` reachable from your local browser at `http://localhost:8080`.

## What Is Already Installed

The instance has a working Python and CUDA stack:

- Ubuntu 24.04.4
- `/usr/bin/python3` available system-wide
- a ready-to-use environment at `/venv/main`
- PyTorch `2.12.0+cu130`
- CUDA available through PyTorch
- one NVIDIA GeForce RTX 3060
- Python `3.12.13`

The repo is cloned on-box at `/root/universe-lm`, and the default
requirements are already installed into `/venv/main`.

## How To Use It

Activate the environment after logging in:

```bash
source /venv/main/bin/activate
```

Then run Python normally:

```bash
python --version
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

If you prefer not to activate the environment, use the interpreter directly:

```bash
/venv/main/bin/python --version
```

## Notes

- Do not reinstall Python, PyTorch, or CUDA.
- Use the existing `/venv/main` environment for training and experiments.
- Treat `/workspace` as disposable on this instance; it is not backed by a volume.
- Box-specific setup rules live in `/root/AGENTS.md` and `/etc/vast_agents/*.md`.
- Keep the SSH tunnel open while using forwarded services on port `8080`.
