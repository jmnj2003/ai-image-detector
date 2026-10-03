"""Preprocess the self-made attribution test set (SDXL / ChatGPT / Gemini / Dola).

  1. Crop off the bottom 10% of every image (removes Dola's bottom-right
     watermark; harmless for the others).
  2. Resize the shorter side to TARGET, then center-crop to a TARGET x TARGET
     square (unifies resolution and aspect ratio across sources).
  3. Re-encode everything as JPEG at the same quality (unifies file format --
     SDXL/Dola were PNG, ChatGPT/Gemini were PNG/JPEG at different qualities).
  4. Writes metadata.csv: id, prompt_id, category, prompt, generator,
     orig_width, orig_height, orig_format.

Usage:
    pip install pillow
    python preprocess_attribution_set.py
"""
import csv
import os
import re

from PIL import Image

SRC_ROOT = os.environ.get("SRC_ROOT", "data")
OUT_ROOT = os.environ.get("OUT_ROOT", "data_processed/attribution")
TARGET = 768          # output is TARGET x TARGET
BOTTOM_CROP_FRAC = 0.10  # crop off the bottom 10% of every image, all sources
JPEG_QUALITY = 92

# folder name -> (file prefix, generator label)
GENERATORS = {
    "sdxl": ("p", "sdxl"),
    "chatgpt": ("c", "chatgpt"),
    "gemini": ("g", "gemini"),
    "dola": ("d", "dola"),
}

# Same 50 prompts / categories used for every generator, in generation order.
PROMPTS = [
    ("p01", "portrait", "A middle-aged woman with short gray hair smiling in a sunlit kitchen."),
    ("p02", "portrait", "A young man wearing a denim jacket standing at a bus stop on a rainy evening."),
    ("p03", "portrait", "An elderly fisherman mending a net on a wooden dock."),
    ("p04", "portrait", "A little girl in a yellow raincoat jumping in a puddle."),
    ("p05", "portrait", "A teenage boy playing a guitar on a bedroom floor."),
    ("p06", "group", "Three friends laughing around a table at an outdoor café."),
    ("p07", "group", "A group of children playing football on a dusty field."),
    ("p08", "group", "Commuters waiting on a crowded subway platform."),
    ("p09", "group", "A family having a picnic in a park on a sunny afternoon."),
    ("p10", "group", "Two cooks working side by side in a busy restaurant kitchen."),
    ("p11", "landscape", "A misty mountain valley at sunrise with a small river."),
    ("p12", "landscape", "A tropical beach with palm trees and a wooden boat."),
    ("p13", "landscape", "A snowy pine forest with a cabin and smoke rising from the chimney."),
    ("p14", "landscape", "A rice terrace on a hillside after the rain."),
    ("p15", "landscape", "A desert road stretching toward distant mountains at dusk."),
    ("p16", "animal", "A tabby cat sleeping on a windowsill."),
    ("p17", "animal", "A golden retriever running along a shoreline."),
    ("p18", "animal", "A flock of birds resting on a power line at sunset."),
    ("p19", "animal", "A brown horse standing in a grassy field."),
    ("p20", "animal", "A butterfly landing on a purple flower."),
    ("p21", "food", "A bowl of ramen with a soft-boiled egg and green onions."),
    ("p22", "food", "A slice of chocolate cake on a white plate."),
    ("p23", "food", "A street vendor grilling skewers at night."),
    ("p24", "food", "A fresh fruit platter with watermelon, mango, and grapes."),
    ("p25", "food", "A plate of nasi lemak with sambal, egg, and fried anchovies."),
    ("p26", "indoor", "A cozy living room with a sofa, bookshelf, and a floor lamp."),
    ("p27", "indoor", "A modern office with rows of desks and large windows."),
    ("p28", "indoor", "A cluttered workshop with tools hanging on the wall."),
    ("p29", "indoor", "A small bedroom with a single bed and morning light."),
    ("p30", "indoor", "A library reading room with tall shelves."),
    ("p31", "street", "A narrow alley in an old town with hanging lanterns."),
    ("p32", "street", "A busy intersection in a big city at night."),
    ("p33", "street", "A quiet suburban street lined with trees in autumn."),
    ("p34", "street", "A market street full of stalls and shoppers."),
    ("p35", "street", "A train station platform in the early morning."),
    ("p36", "object", "A wristwatch lying on a wooden table."),
    ("p37", "object", "A pair of worn running shoes on a concrete floor."),
    ("p38", "object", "A cup of coffee next to an open notebook and a pen."),
    ("p39", "object", "A bicycle leaning against a brick wall."),
    ("p40", "object", "A vase of sunflowers on a kitchen counter."),
    ("p41", "illustration", "A watercolor painting of a lighthouse on a cliff."),
    ("p42", "illustration", "A flat vector illustration of a city skyline at sunset."),
    ("p43", "illustration", "A pencil sketch of an old man reading a newspaper."),
    ("p44", "illustration", "An anime-style illustration of a girl walking through a cherry blossom lane."),
    ("p45", "illustration", "A children's book illustration of a bear having tea in a forest."),
    ("p46", "text", "A storefront sign that reads \"OPEN 24 HOURS\" above a small shop door."),
    ("p47", "text", "A chalkboard menu listing three coffee drinks and their prices."),
    ("p48", "text", "A poster on a wall that says \"Welcome to the Science Fair\"."),
    ("p49", "text", "A street name sign reading \"Maple Street\" on a pole."),
    ("p50", "text", "A birthday card with the words \"Happy Birthday\" on the front."),
]


def find_generator_files(folder, prefix):
    """Map prompt index (1..50) -> file path, for files named like c1.png, c10.png, ..."""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.(png|jpe?g)$", re.IGNORECASE)
    out = {}
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            out[int(m.group(1))] = os.path.join(folder, fname)
    return out


def process_image(path):
    """Bottom-crop -> resize shorter side -> center-crop square. Returns a PIL Image."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        orig_w, orig_h, orig_fmt = w, h, (Image.open(path).format or "unknown")

        # 1. crop off the bottom strip (same fraction for every source)
        keep_h = int(h * (1 - BOTTOM_CROP_FRAC))
        im = im.crop((0, 0, w, keep_h))
        w, h = im.size

        # 2. resize shorter side to TARGET
        scale = TARGET / min(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        im = im.resize((new_w, new_h), Image.LANCZOS)

        # 3. center-crop to TARGET x TARGET
        left = (new_w - TARGET) // 2
        top = (new_h - TARGET) // 2
        im = im.crop((left, top, left + TARGET, top + TARGET))

        return im, orig_w, orig_h, orig_fmt


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    meta_path = os.path.join(OUT_ROOT, "metadata.csv")

    rows = []
    for folder_name, (prefix, generator) in GENERATORS.items():
        folder = os.path.join(SRC_ROOT, folder_name)
        if not os.path.isdir(folder):
            print(f"[skip] {folder} not found")
            continue

        files_by_idx = find_generator_files(folder, prefix)
        out_dir = os.path.join(OUT_ROOT, generator)
        os.makedirs(out_dir, exist_ok=True)

        missing = []
        for idx, (prompt_id, category, prompt) in enumerate(PROMPTS, 1):
            src_path = files_by_idx.get(idx)
            if src_path is None:
                missing.append(idx)
                continue

            img, ow, oh, ofmt = process_image(src_path)
            out_name = f"{generator}_{prompt_id}.jpg"
            out_path = os.path.join(out_dir, out_name)
            img.save(out_path, "JPEG", quality=JPEG_QUALITY)

            rows.append([
                out_name, prompt_id, category, prompt, generator,
                ow, oh, ofmt,
            ])

        if missing:
            print(f"[warn] {generator}: no file found for prompt index(es) {missing}")
        else:
            print(f"[ok] {generator}: all 50 processed -> {out_dir}")

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "prompt_id", "category", "prompt", "generator",
            "orig_width", "orig_height", "orig_format",
        ])
        writer.writerows(rows)

    print(f"\n{len(rows)} images processed. Metadata written to {meta_path}")


if __name__ == "__main__":
    main()