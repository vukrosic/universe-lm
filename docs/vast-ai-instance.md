# Vast AI Instance

## Connecting

```bash
ssh -p 50670 root@154.12.38.116
```

No `-L` tunnel needed for command execution. The tunnel is only needed when you want to expose remote services (e.g., TensorBoard) to your local browser.

## Why This Works

The Vast AI SSH wrapper handles authentication interactively. Non-interactive probes (e.g., `ssh ... -o BatchMode=yes` or `ssh ... echo done`) will fail with "Connection closed by remote host" — this is the wrapper rejecting them, not a connectivity issue.

Always use a real TTY session: just `ssh -p 50670 root@154.12.38.116` with no extra options that skip interactive auth.

## What Is Already Installed

- Ubuntu 24.04.4
- `/usr/bin/python3` available system-wide
- a ready-to-use environment at `/venv/main`
- PyTorch `2.11.0+cu130`
- CUDA available through PyTorch
- one NVIDIA GeForce RTX 3090

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

## Long-Running Work

Use `tmux` to keep sessions alive after you disconnect:

```bash
# Create a named session
tmux new -s lock10m

# Detach with Ctrl+B, then D

# Reconnect later
tmux attach -t lock10m

# List sessions
tmux list-sessions
```

## Notes

- Do not reinstall Python, PyTorch, or CUDA.
- Use the existing `/venv/main` environment for training and experiments.
- Keep the SSH tunnel open while using forwarded services on port `8080`.
