#!/usr/bin/env bash
# Universe release orchestrator.
#
# Subcommands:
#   smoke      <version>          # raw-checkpoint release path (no HF format)
#     -> train -> eval_ppl + generation samples -> upload raw .pt
#
#   train      <version> [args]   # just training, writes COMMIT
#   eval-ppl   <version>          # perplexity-only eval
#   sample     <version> "prompt" # generate from checkpoint
#   publish    <version> <repo>   # upload to HuggingFace (raw .pt)
#
# Each release lives in releases/<version>/ with:
#   COMMIT, train_log.txt, model.pt (symlinked), ppl.json, samples.txt, notes.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

CMD="${1:?usage: release.sh <smoke|train|eval-ppl|sample|publish> <version> [args]}"
VERSION="${2:?need version like v0.0}"
shift 2 || true

RELEASE_DIR="releases/$VERSION"
CKPT_DIR="checkpoints/$VERSION"

require_clean_tree() {
  if [ -n "$(git status --porcelain)" ]; then
    echo "[release] git working tree is dirty. Commit or stash first." >&2
    git status --short >&2
    exit 1
  fi
}

run_train() {
  local config="${1:-smoke}"
  shift || true
  mkdir -p "$CKPT_DIR" "$RELEASE_DIR"
  git rev-parse HEAD > "$RELEASE_DIR/COMMIT"
  echo "[train] version=$VERSION  config=$config  commit=$(cat "$RELEASE_DIR/COMMIT")"
  python train_llm.py \
    --config "$config" \
    --output_dir "$CKPT_DIR" \
    "$@" 2>&1 | tee "$RELEASE_DIR/train_log.txt"
}

run_eval_ppl() {
  local ckpt="$CKPT_DIR/model.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[eval-ppl] no checkpoint at $ckpt" >&2
    exit 1
  fi
  python -m scripts.eval_ppl \
    --checkpoint "$ckpt" \
    --output "$RELEASE_DIR/ppl.json"
}

run_sample() {
  local prompts=(
    "Once upon a time"
    "The capital of France is"
    "import torch\nimport torch.nn as nn\n\nclass"
    "Q: What is 2 + 2?\nA:"
  )
  local ckpt="$CKPT_DIR/model.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[sample] no checkpoint at $ckpt" >&2
    exit 1
  fi
  : > "$RELEASE_DIR/samples.txt"
  for p in "${prompts[@]}"; do
    echo "=== prompt: $p ===" >> "$RELEASE_DIR/samples.txt"
    python -m scripts.generate \
      --checkpoint "$ckpt" \
      --prompt "$(printf '%b' "$p")" \
      --max-new-tokens 80 \
      >> "$RELEASE_DIR/samples.txt"
    echo "" >> "$RELEASE_DIR/samples.txt"
  done
  echo "[sample] wrote $RELEASE_DIR/samples.txt"
}

run_publish() {
  local repo_id="${1:?usage: publish <version> <hf-repo-id>}"
  local ckpt="$CKPT_DIR/model.pt"
  for f in "$ckpt" "$RELEASE_DIR/notes.md"; do
    if [ ! -f "$f" ]; then
      echo "[publish] missing $f" >&2
      exit 1
    fi
  done

  local stage="$RELEASE_DIR/_upload"
  mkdir -p "$stage"
  cp "$ckpt" "$stage/model.pt"
  cp "$RELEASE_DIR/notes.md" "$stage/README.md"
  [ -f "$RELEASE_DIR/ppl.json" ]    && cp "$RELEASE_DIR/ppl.json"    "$stage/"
  [ -f "$RELEASE_DIR/samples.txt" ] && cp "$RELEASE_DIR/samples.txt" "$stage/"

  python -m scripts.upload_to_hf \
    --local-dir "$stage" \
    --repo-id "$repo_id" \
    --commit-message "Universe $VERSION"

  rm -rf "$stage"
  git tag "$VERSION" 2>/dev/null || true
  echo "[publish] tagged $VERSION (run: git push --tags when ready)"
}

case "$CMD" in
  smoke)
    require_clean_tree
    run_train "smoke" "$@"
    run_eval_ppl
    run_sample
    if [ ! -f "$RELEASE_DIR/notes.md" ]; then
      cp releases/TEMPLATE/notes.md "$RELEASE_DIR/notes.md" 2>/dev/null || true
    fi
    echo "[smoke] done. Edit $RELEASE_DIR/notes.md, then: ./release.sh publish $VERSION <hf-repo>"
    ;;
  train)      require_clean_tree; run_train "$@" ;;
  eval-ppl)   run_eval_ppl ;;
  sample)     run_sample ;;
  publish)    run_publish "$@" ;;
  *)
    echo "unknown subcommand: $CMD" >&2
    exit 1
    ;;
esac
