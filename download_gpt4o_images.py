"""Pull N GPT-4o text-to-image samples from WINDop/OpenGPT-4o-Image into the
'openai' attribution-training folder, WITHOUT storing the 10.7GB archive.

How: the text-to-image images live in split archives gen.tar.gz.00 ... .05
(~10.7GB each). Part .00 is the start of one gzip+tar stream, so it can be
read on its own, front to back. This script streams it over HTTP straight
into a tar reader, converts each PNG to JPEG, and stops once it has N images.
Only the JPEGs you keep touch the disk. (Parts .01-.05 can't be read on
their own -- they start mid-stream -- so this only ever uses part .00.)

Only the gen/ (text-to-image) images are used. The editing/ images are
skipped on purpose: an edited photo is part real, part AI, which would muddy
the 'openai' label.

Usage:
    pip install requests pillow
    python download_gpt4o_images.py                 # 1500 images (default)
    N_IMAGES=3000 STRIDE=2 python download_gpt4o_images.py
"""
import io
import os
import tarfile

import requests
from PIL import Image

REPO = "WINDop/OpenGPT-4o-Image"
PART = "gen.tar.gz.00"
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
OUT_DIR = os.environ.get("OUT_DIR", "downloaded_data/attribution_train/openai")
N_IMAGES = int(os.environ.get("N_IMAGES", "1500"))
# Keep every STRIDE-th image. The archive is probably ordered by task type,
# so taking every 3rd image over a longer stretch spreads the sample across
# more task types than taking the first 1500 in a row would.
STRIDE = int(os.environ.get("STRIDE", "3"))
JPEG_QUALITY = 95


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    already = {f for f in os.listdir(OUT_DIR) if f.startswith("gpt4o_")}
    if len(already) >= N_IMAGES:
        print(f"Already have {len(already)} gpt4o_ images in {OUT_DIR}; nothing to do.")
        return

    url = f"{ENDPOINT}/datasets/{REPO}/resolve/main/{PART}"
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    print(f"Streaming {url}")
    print(f"Keeping every {STRIDE}th image until {N_IMAGES} saved -> {OUT_DIR}")

    resp = requests.get(url, stream=True, headers=headers, timeout=60, allow_redirects=True)
    resp.raise_for_status()
    resp.raw.decode_content = False  # it's a .tar.gz; let tarfile do the gunzip

    saved, seen = len(already), 0
    try:
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name.replace("\\", "/")
                if "/gen/" not in f"/{name}" or not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    continue
                seen += 1
                if seen % STRIDE != 0:
                    continue  # tar streams are sequential, skipping just reads past it

                out_name = f"gpt4o_{os.path.splitext(os.path.basename(name))[0]}.jpg"
                out_path = os.path.join(OUT_DIR, out_name)
                if out_name in already:
                    continue

                data = tar.extractfile(member).read()
                try:
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    img.save(out_path, "JPEG", quality=JPEG_QUALITY)
                except Exception as e:
                    print(f"  [skip] couldn't decode {name}: {e}")
                    continue

                saved += 1
                if saved % 100 == 0:
                    print(f"  saved {saved}/{N_IMAGES} (scanned {seen} images so far)")
                if saved >= N_IMAGES:
                    break
    except (tarfile.ReadError, EOFError, requests.exceptions.ChunkedEncodingError) as e:
        # A dropped connection ends the stream early. Everything saved so far
        # is valid; just re-run to continue (existing files are skipped).
        print(f"Stream ended early ({type(e).__name__}: {e}). Re-run to continue.")
    finally:
        resp.close()

    print(f"Done. {saved} gpt4o_ images in {OUT_DIR}")


if __name__ == "__main__":
    main()