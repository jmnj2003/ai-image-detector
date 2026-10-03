"""Add PROMPT-MATCHED training images from Rapidata/Seedream-3_t2i_human_preference.

  - streams the dataset (no full download / Arrow conversion, avoiding the
    OOM seen before)
  - keeps only the image/model/prompt columns; the voter info columns are
    never read or stored
  - each Seedream image appears in many comparison rows, so images are
    de-duplicated by a hash of their bytes
  - maps model names to the 4 classes below; any model not listed is skipped
  - writes files as rapidata_<hash>.jpg into
    downloaded_data/attribution_train/<class>/ plus a metadata CSV

Usage:
    pip install datasets pillow
    python download_rapidata_matched.py
    MAX_PER_CLASS=800 python download_rapidata_matched.py
"""
import csv
import hashlib
import io
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image

REPO = "Rapidata/Seedream-3_t2i_human_preference"
OUT_ROOT = os.environ.get("OUT_ROOT", "downloaded_data/attribution_train")
MAX_PER_CLASS = int(os.environ.get("MAX_PER_CLASS", "1500"))
JPEG_QUALITY = 95

# Substring (lowercase) -> class. First match wins, so order matters.
# Deliberately NOT mapped: midjourney, ideogram, recraft, aurora, frames,
# halfmoon, janus, lumina, hidream -- they aren't one of our 4 classes, and
# guessing them into one would put wrong labels into training.
MODEL_TO_CLASS = [
    ("seedream", "bytedance"),
    ("dalle", "openai"),
    ("dall-e", "openai"),
    ("gpt", "openai"),
    ("imagen", "google"),
    ("flux", "opensource"),
    ("stable-diffusion", "opensource"),
    ("stable_diffusion", "opensource"),
    ("sd3", "opensource"),
    ("sd-3", "opensource"),
]
EXACT_TO_CLASS = {"4o": "openai"}


def model_to_class(model_name):
    m = (model_name or "").strip().lower()
    if m in EXACT_TO_CLASS:
        return EXACT_TO_CLASS[m]
    for key, cls in MODEL_TO_CLASS:
        if key in m:
            return cls
    return None


def image_bytes(field):
    """With decode=False the image column is {"bytes": ..., "path": ...}."""
    if isinstance(field, dict):
        return field.get("bytes")
    if isinstance(field, (bytes, bytearray)):
        return bytes(field)
    if isinstance(field, Image.Image):  # fallback if decoding couldn't be turned off
        buf = io.BytesIO()
        field.save(buf, "PNG")
        return buf.getvalue()
    return None


def main():
    ds = load_dataset(REPO, split="train", streaming=True)
    try:
        ds = ds.select_columns(["prompt", "image1", "image2", "model1", "model2"])
    except Exception:
        pass  # older datasets versions; the extra columns are simply ignored below
    for col in ("image1", "image2"):
        try:
            ds = ds.cast_column(col, HFImage(decode=False))  # hash raw bytes, decode only kept ones
        except Exception:
            pass

    seen_hashes = set()
    counts = {}
    for cls in {c for _, c in MODEL_TO_CLASS} | set(EXACT_TO_CLASS.values()):
        os.makedirs(os.path.join(OUT_ROOT, cls), exist_ok=True)
        existing = [f for f in os.listdir(os.path.join(OUT_ROOT, cls)) if f.startswith("rapidata_")]
        counts[cls] = len(existing)
        seen_hashes.update(f[len("rapidata_"):-4] for f in existing)

    unmapped = {}
    meta_path = os.path.join(OUT_ROOT, "rapidata_metadata.csv")
    new_meta = not os.path.exists(meta_path)
    rows_read = 0

    with open(meta_path, "a", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        if new_meta:
            writer.writerow(["filename", "class", "model", "prompt"])

        for row in ds:
            rows_read += 1
            for img_col, model_col in (("image1", "model1"), ("image2", "model2")):
                model = row.get(model_col)
                cls = model_to_class(model)
                if cls is None:
                    unmapped[model] = unmapped.get(model, 0) + 1
                    continue
                if counts[cls] >= MAX_PER_CLASS:
                    continue
                data = image_bytes(row.get(img_col))
                if not data:
                    continue
                h = hashlib.md5(data).hexdigest()[:16]
                if h in seen_hashes:
                    continue  # same image already saved (Seedream repeats across rows)
                seen_hashes.add(h)
                try:
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    continue
                fname = f"rapidata_{h}.jpg"
                img.save(os.path.join(OUT_ROOT, cls, fname), "JPEG", quality=JPEG_QUALITY)
                writer.writerow([fname, cls, model, row.get("prompt", "")])
                counts[cls] += 1

            if rows_read % 500 == 0:
                mf.flush()
                print(f"  rows read {rows_read}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
            if all(v >= MAX_PER_CLASS for v in counts.values()):
                break

    print(f"\nDone after {rows_read} rows. rapidata_ images per class:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    if unmapped:
        print("\nSkipped models (not one of the 4 classes):")
        for m, n in sorted(unmapped.items(), key=lambda kv: -kv[1]):
            print(f"  {m}: {n}")
    print("\nIf a model you expected to be used shows up in the skipped list "
          "(e.g. a differently-spelled Stable Diffusion name), add it to MODEL_TO_CLASS and re-run.")


if __name__ == "__main__":
    main()