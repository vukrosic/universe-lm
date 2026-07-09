You are setting up a freshly rented Vast.ai GPU box so this project can run
experiments on it. The box is bare — no repo, no venv, no dependencies yet.

## Box connection
- SSH (non-interactive, use THIS form for commands): `{{SSH_PREFIX}} "<remote command>"`
- Raw SSH command the user pasted (for reference): `{{SSH_RAW}}`
- host: `{{HOST}}`  port: `{{PORT}}`  user: `{{USER}}`
- target repo dir on the box: `{{REMOTE_REPO}}`
- target venv on the box: `{{REMOTE_VENV}}`
- repo to clone: `{{REPO_URL}}`

Your local working directory is this project's repo, so you can read its
instruction files locally for reference (README.md, CONTRIBUTING.md, AGENT.md,
and anything under autoresearch/ such as RUN-CONTRACT.md / PIPELINE.md). The box
should end up running the SAME code.

## Do this, step by step
1. Confirm you can reach the box: `{{SSH_PREFIX}} "nvidia-smi && python3 --version"`.
   If SSH fails, STOP and report the exact error (wrong host/port, key not
   accepted, box still booting). Do not guess past a connection failure.
2. Clone the repo onto the box at `{{REMOTE_REPO}}` if it is not already there
   (`git clone {{REPO_URL}} {{REMOTE_REPO}}`); otherwise `git pull` it.
3. READ the repo's own setup instructions (README / CONTRIBUTING and the
   autoresearch docs) and FOLLOW them. Create/activate the venv at
   `{{REMOTE_VENV}}` and install the project's dependencies (requirements.txt /
   pip / whatever the repo specifies).
4. Smoke-test that a run can actually start: verify torch sees CUDA
   (`python -c "import torch; print(torch.cuda.is_available())"`) and that the
   project's entrypoint imports without error. Do NOT launch a full training run
   — just prove the environment is ready.
5. If you discover the real hardware/CUDA/torch versions, UPDATE the local
   `autoresearch/remote-box.json` `hardware` and `notes` fields to match what you
   actually saw on the box (keep host/port/user/ssh as they are).

## Report
End with a short summary: did SSH work, was the repo cloned/updated, did the venv
+ deps install cleanly, did the CUDA + import smoke check pass, and anything
still broken that needs the user. Be specific about any failure.
