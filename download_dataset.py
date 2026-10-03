"""Download the three Kaggle training datasets with kagglehub.

Requires a Kaggle account + API token first:
    1. https://www.kaggle.com/settings -> "Create New Token" -> downloads kaggle.json
    2. Put it at ~/.kaggle/kaggle.json  (on AutoDL: /root/.kaggle/kaggle.json)
       chmod 600 ~/.kaggle/kaggle.json
    3. pip install kagglehub

Usage:
    python download_dataset.py
"""
import os
import shutil

import kagglehub

# Where to keep everything -- put this on the DATA disk on AutoDL.
DEST_ROOT = os.environ.get("DATA_ROOT", "/root/autodl-tmp/data/kaggle_training")

DATASETS = {
    "human_vs_ai_laxmikanta": "laxmikantaroy/human-vs-ai-generated-image-classification-dataset",
    "real_ai_promy": "fatemaaktherpromy/real-and-ai-generated-image-detect",
    "ai_media_classifier_kotayogi": "kotayogi/ai-media-classifier",
}


def main():
    os.makedirs(DEST_ROOT, exist_ok=True)
    for local_name, kaggle_id in DATASETS.items():
        dest = os.path.join(DEST_ROOT, local_name)
        if os.path.isdir(dest) and os.listdir(dest):
            print(f"[skip] {local_name} already downloaded at {dest}")
            continue

        print(f"[downloading] {kaggle_id} ...")
        # kagglehub caches under ~/.cache/kagglehub and returns that path.
        cache_path = kagglehub.dataset_download(kaggle_id)
        shutil.copytree(cache_path, dest, dirs_exist_ok=True)
        print(f"[done] {local_name} -> {dest}")

    print("\nAll requested datasets are under:", DEST_ROOT)
    print("Next: inspect each folder's structure before writing per-dataset loaders --")
    print("Kaggle datasets don't follow one fixed layout, so check e.g.:")
    for local_name in DATASETS:
        print(f"  find {os.path.join(DEST_ROOT, local_name)} -maxdepth 2 | head -20")


if __name__ == "__main__":
    main()