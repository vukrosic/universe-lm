"""Push a Universe release directory to HuggingFace Hub.

Expects a local dir with the raw checkpoint + README + any extras.
No HF transformers format required — users clone universe-lm to load.

Usage:
    python -m scripts.upload_to_hf \\
        --local-dir releases/v0.0/_upload \\
        --repo-id vukrosic/universe-15m-v0.0
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--commit-message", default="release")
    args = parser.parse_args()

    if not args.local_dir.is_dir():
        raise SystemExit(f"local dir not found: {args.local_dir}")

    create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")
    HfApi().upload_folder(
        folder_path=str(args.local_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    print(f"uploaded -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
