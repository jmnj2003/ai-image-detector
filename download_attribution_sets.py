"""Download the three generator-attribution datasets and save them as plain
image files under one folder per generator, with a metadata.csv.

    Google     <- bitmind/nano-banana          (Hugging Face)
    ByteDance  <- ash12321/seedream-4.5-generated-2k (Hugging Face)
    OpenAI     <- gpt-image-1-pictures          (Kaggle)

Requires:
    pip install datasets huggingface_hub kagglehub pillow pyarrow
    Kaggle API token at ~/.kaggle/kaggle.json (see download_kaggle.py's docstring)

Usage:
    python download_attribution_sets.py
    MAX_GOOGLE=3000 python download_attribution_sets.py   # cap Google to 3000 images
"""
import csv
import glob
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # avoids the CAS/Xet 401 seen earlier

import kagglehub
from datasets import load_dataset

OUT_ROOT = os.environ.get("OUT_ROOT", "downloaded_data/attribution_train")
MAX_GOOGLE = int(os.environ.get("MAX_GOOGLE", "0"))  # 0 = no cap, keep all 9457
EXCLUDE_RECREATED = os.environ.get("INCLUDE_RECREATED", "0") != "1"


def save_hf_dataset_streaming_safe(repo_id, generator, out_dir, max_n=0):
    import glob
    import io

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)

    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id, repo_type="dataset",
            allow_patterns=["*.parquet"], local_files_only=True,
        )
        parquet_files = sorted(glob.glob(os.path.join(snapshot_path, "**", "*.parquet"), recursive=True))
    except Exception:
        parquet_files = []

    if not parquet_files:
        print(f"  {generator}: not cached locally, downloading {repo_id} ...")
        snapshot_path = snapshot_download(
            repo_id=repo_id, repo_type="dataset", allow_patterns=["*.parquet"],
        )
        parquet_files = sorted(glob.glob(os.path.join(snapshot_path, "**", "*.parquet"), recursive=True))

    if not parquet_files:
        raise FileNotFoundError(f"Still no parquet files found for {repo_id} after attempting a download.")
    print(f"  {generator}: found {len(parquet_files)} cached parquet file(s)")

    rows = []
    count = 0
    for pf_path in parquet_files:
        if max_n and count >= max_n:
            break
        pf = pq.ParquetFile(pf_path)
        for batch in pf.iter_batches(batch_size=50, columns=["image"]):
            for rec in batch.to_pylist():
                if max_n and count >= max_n:
                    break
                img_field = rec["image"]  # HF image feature: {"bytes": b"...", "path": ...}
                img_bytes = img_field["bytes"] if isinstance(img_field, dict) else img_field
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                fname = f"{generator}_{count:05d}.jpg"
                img.save(os.path.join(out_dir, fname), "JPEG", quality=95)
                rows.append([fname, generator, repo_id, count])
                count += 1
                if count % 500 == 0:
                    print(f"  {generator}: {count}")
            if max_n and count >= max_n:
                break
    return rows


def save_hf_dataset(repo_id, generator, out_dir, max_n=0):
    os.makedirs(out_dir, exist_ok=True)
    if max_n:
        import itertools
        ds = load_dataset(repo_id, split="train", streaming=True)
        rows_iter = itertools.islice(ds, max_n)
    else:
        ds = load_dataset(repo_id, split="train")
        rows_iter = ds

    rows = []
    for i, row in enumerate(rows_iter):
        img = row["image"]  # PIL Image, per the imagefolder format both repos use
        fname = f"{generator}_{i:05d}.jpg"
        img.convert("RGB").save(os.path.join(out_dir, fname), "JPEG", quality=95)
        rows.append([fname, generator, repo_id, i])
        if (i + 1) % 500 == 0:
            print(f"  {generator}: {i + 1}")
    return rows


def save_kaggle_gpt_image(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cache_path = kagglehub.dataset_download("seifbenayed/gpt-image-1-pictures")
    files = sorted(
        glob.glob(os.path.join(cache_path, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(cache_path, "**", "*.jpg"), recursive=True)
    )
    if EXCLUDE_RECREATED:
        before = len(files)
        files = [f for f in files if "_recreated" not in os.path.basename(f)]
        print(f"  openai: excluded {before - len(files)} '_recreated' files "
              f"(likely image-edits, not pure text-to-image)")

    rows = []
    from PIL import Image
    for i, f in enumerate(files):
        fname = f"openai_{i:05d}.jpg"
        with Image.open(f) as im:
            im.convert("RGB").save(os.path.join(out_dir, fname), "JPEG", quality=95)
        rows.append([fname, "openai", "seifbenayed/gpt-image-1-pictures", os.path.basename(f)])
    return rows


def main():
    all_rows = []

    print("Downloading Google (nano-banana) ...")
    all_rows += save_hf_dataset_streaming_safe(
        "bitmind/nano-banana", "google",
        os.path.join(OUT_ROOT, "google"), max_n=MAX_GOOGLE or 3000,
    )

    print("Downloading ByteDance (seedream-4.5) ...")
    all_rows += save_hf_dataset(
        "ash12321/seedream-4.5-generated-2k", "bytedance",
        os.path.join(OUT_ROOT, "bytedance"),
    )

    print("Downloading OpenAI (gpt-image-1, Kaggle) ...")
    all_rows += save_kaggle_gpt_image(os.path.join(OUT_ROOT, "openai"))

    meta_path = os.path.join(OUT_ROOT, "metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "generator", "source_repo", "source_id"])
        writer.writerows(all_rows)

    print(f"\n{len(all_rows)} images total. Metadata at {meta_path}")
    for gen in ("google", "bytedance", "openai"):
        n = sum(1 for r in all_rows if r[1] == gen)
        print(f"  {gen}: {n}")


if __name__ == "__main__":
    main()