# Remote CLI (`scripts/remote.sh`)

One entry point for connecting to and controlling the remote Vast.ai training
box. Replaces ad-hoc `ssh -p … root@… "grep … tail …"` one-liners. Vast
instances are ephemeral, so **all connection details live in one file** —
when you rent a new box, edit `.remote` and every command follows.

## Setup (once per rental)

```bash
scripts/remote.sh init            # creates .remote from .remote.example
$EDITOR .remote                   # set REMOTE_HOST / REMOTE_PORT for the new box
scripts/remote.sh install         # pip install -r requirements.txt in the venv
scripts/remote.sh data            # (first time) download the dataset
```

`.remote` is gitignored. The PyTorch venv (`REMOTE_VENV`, default `/venv/main`
— the stock Vast image) is activated automatically for every remote python
command, so you never source it by hand.

### `.remote` fields

| Field | Meaning | Default |
|---|---|---|
| `REMOTE_HOST` | `user@ip` of the box | — |
| `REMOTE_PORT` | ssh port Vast assigned | — |
| `REMOTE_DIR` | checkout path on the box | — |
| `REMOTE_VENV` | PyTorch venv on the box | `/venv/main` |
| `LOCAL_FWD` | local port forwarded to remote 8080 | `8080` |

## Commands

| Command | What it does |
|---|---|
| `init` | create `.remote` from the example |
| `connect [session]` | ssh in: tmux session + `localhost:8080` forward |
| `sync` | rsync local code up (excludes data/checkpoints/logs/runs) |
| `install` | `pip install -r requirements.txt` in the venv + verify torch/CUDA |
| `data` | start the dataset download in tmux `data` |
| `launch <config> [name] [-- extra args]` | sync + start training in tmux, tee to `logs/<name>.log` |
| `status [name]` | val-loss points, live step/ETA, final, `RUNNING`/`STOPPED` |
| `logs [-f] [name]` | tail (or follow) `logs/<name>.log` |
| `pull [name] [dest]` | scp `runs/<name>/full/metrics.json` down |
| `promote [name]` | pull metrics → overwrite `baselines/10m_baseline.json` |
| `ssh <cmd…>` | run any command on the box (venv active) — escape hatch |

Default run name is `run`.

## Typical loop (screen → confirm)

```bash
# screen a lever at 10M params / 20M tokens
scripts/remote.sh launch screen10m screen10m_base
scripts/remote.sh status screen10m_base          # poll
scripts/remote.sh logs -f screen10m_base         # or follow live

# a confirmed winner earns a full 200M run
scripts/remote.sh launch 10m my_record -- --muon_lr 0.02
scripts/remote.sh status my_record

# when it wins, pull its curve in as the new baseline
scripts/remote.sh promote my_record
# then commit baselines/10m_baseline.json + add the LEADERBOARD.md row
```

## Notes

- `launch` maps `<name>` consistently to tmux session `<name>`, log
  `logs/<name>.log`, and output `runs/<name>/full` — so `status`/`logs`/`pull`
  all key off the same name.
- `sync` never uploads `processed_data/`, `checkpoints/`, `runs/`, `logs/`, or
  `.git` — code only. Use `data` to build the dataset on the box.
- Provisioning the instance itself is out of scope — use the `vastai` CLI for
  that, then point `.remote` at the new box.
