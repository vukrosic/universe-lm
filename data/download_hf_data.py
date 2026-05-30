from datasets import load_dataset
import os
import shutil


def main() -> None:
    # HF cache and processed_data live on the same 32G disk. load_dataset keeps
    # the downloaded parquet (hub/) AND an uncompressed Arrow copy (datasets/),
    # then save_to_disk writes a THIRD copy. All three at once overflow 32G.
    # Fix: drop the parquet once it's loaded (data is backed by the Arrow cache),
    # then drop the Arrow cache after save. Output is identical; training reads
    # only processed_data/.
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    print("Downloading 1B pretraining data...")
    ds = load_dataset("vukrosic/blueberry-1B-pretrain")

    # parquet in hub/ no longer needed -> free ~3.5G so the save peak fits
    shutil.rmtree(os.path.join(hf_home, "hub"), ignore_errors=True)

    output_dir = "processed_data/pretrain_1B"
    os.makedirs(output_dir, exist_ok=True)
    ds.save_to_disk(output_dir)
    print(f"✅ Saved dataset to {output_dir}")

    # training reads only processed_data/ -> drop the throwaway Arrow cache
    shutil.rmtree(os.path.join(hf_home, "datasets"), ignore_errors=True)
    print("✅ Cleared HF cache; processed_data ready")


if __name__ == "__main__":
    main()
