import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a model snapshot before running Slurm jobs.")
    parser.add_argument("--model_name", required=True, help="Hugging Face model id, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--cache_dir", default=None, help="Optional cache dir (HF_HOME/HF_HUB_CACHE)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
    snapshot_download(
        repo_id=args.model_name,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=False,
    )
    print("Download complete.")


if __name__ == "__main__":
    main()
