"""Build unified manifests from all the data you've collected so far.

Produces (under MANIFEST_DIR, default 'manifests/'):
    phase1_train.csv, phase1_val.csv, phase1_test.csv
        columns: filepath,label       (label: 0=real, 1=ai)
    phase2_train.csv, phase2_val.csv, phase2_test.csv
        columns: filepath,label       (label: google/openai/bytedance/opensource)
    phase1_crossdomain_test.csv, phase2_crossdomain_test.csv
        your 200 self-made images (sdxl/chatgpt/gemini/dola) -- NEVER trained on,
        used only to check generalisation to a domain the model has never seen.
    dola_probe.csv
        your 50 Dola images -- phase-1 label is known (they ARE ai-generated) but
        phase-2 has no ground-truth generator label. Used only to see what the
        attribution model *guesses* Dola is built on, not to score accuracy.

Usage:
    python prepare_manifest.py
"""
import csv
import glob
import os
import random

random.seed(42)

ROOT = os.environ.get("PROJECT_ROOT", ".")
SELF_MADE_DIR = os.path.join(ROOT, "data_processed", "attribution")
ATTR_TRAIN_DIR = os.path.join(ROOT, "downloaded_data", "attribution_train")
KAGGLE_DIR = os.path.join(ROOT, "downloaded_data", "kaggle_training")
MANIFEST_DIR = os.path.join(ROOT, "manifests")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# self-made folder name -> phase2 (attribution) class
SELF_MADE_TO_CLASS = {
    "sdxl": "opensource",
    "chatgpt": "openai",
    "gemini": "google",
    "dola": None,  # unknown ground truth -- probe only, not a training label
}

# downloaded_data/attribution_train folder name -> phase2 class (identity map, kept
# explicit so it's obvious where to edit if you rename folders later)
ATTR_TRAIN_TO_CLASS = {"google": "google", "openai": "openai", "bytedance": "bytedance",
                       "opensource": "opensource"}

MIN_CLASS_SIZE_FOR_TRAIN = 30  # below this, fall back to self-made images for training
CAP_PER_SOURCE = int(os.environ.get("CAP_PER_SOURCE", "8000"))  # keep runtime bounded
# Phase 2 needs a MUCH smaller, separate cap: with google at 3000 vs e.g.
# bytedance at 200, a shared cap doesn't help -- google still dominates by
# 15x and the model learns to default to it whenever it's unsure. Capping
# every attribution class to roughly the same size (near the smallest
# class's real size) prevents that bias, at the cost of not using 100% of
# the larger classes' data.
CAP_PER_ATTR_CLASS = int(os.environ.get("CAP_PER_ATTR_CLASS", "400"))  # (no longer used for phase 2)

# Phase 2 now trains on SPECIFIC MODEL labels (nano-banana, gpt-4o, sdxl, ...)
# and only merges them into companies at prediction time. Old and new
# versions of one company's model look different; forcing them into one
# label made the model learn a blurred average that matched neither.
# The cap is per specific model, so no single model swamps its company.
CAP_PER_FINE_LABEL = int(os.environ.get("CAP_PER_FINE_LABEL", "800"))
MIN_FINE_LABEL = 20  # models with fewer images get merged into "<company>-other"

# Self-made set split: one half picks the winning model, the other half is
# only used to report the final score, so the reported number isn't one we
# tuned against. Split by ODD/EVEN prompt number, not first/second half:
# the 50 prompts are ordered in blocks of 5 by type (portrait, group, ...,
# text), so a first/second-half split gave the two halves completely
# different content. Odd/even puts every content type in both halves.


def load_rapidata_models(attr_dir):
    """filename -> original model name, from download_rapidata_matched.py's CSV."""
    path = os.path.join(attr_dir, "rapidata_metadata.csv")
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[row["filename"]] = row.get("model", "")
    return out


def rapidata_fine(model_name):
    m = (model_name or "").lower()
    if m == "4o" or "gpt" in m:
        return "gpt-4o"
    if "dall" in m:
        return "dalle-3"
    if "imagen" in m:
        return "imagen"
    if "seedream" in m:
        return "seedream-3"
    if "flux" in m:
        return "flux"
    if "stable" in m or "sd3" in m or "sd-3" in m:
        return "sd3"
    return None


def fine_label(path, company, rapidata_models):
    """Which specific model made this training image, from its filename."""
    name = os.path.basename(path)
    if name.startswith("rapidata_"):
        fine = rapidata_fine(rapidata_models.get(name))
        return fine or f"{company}-other"
    prefix_rules = [
        ("gpt4o_", "gpt-4o"),          # download_gpt4o_images.py (OpenGPT-4o-Image)
        ("openai_", "gpt-image-1"),    # Kaggle gpt-image-1 set
        ("google_", "nano-banana"),    # bitmind/nano-banana
        ("bytedance_", "seedream-4.5"),  # ash12321/seedream-4.5-generated-2k
        ("sdxl_bulk_", "sdxl"),        # gen_sdxl_bulk.py
        ("sdxl_p", "sdxl"),            # self-made fallback
        ("chatgpt_p", "chatgpt-app"),  # self-made fallback
        ("gemini_p", "gemini-app"),    # self-made fallback
    ]
    for prefix, fine in prefix_rules:
        if name.startswith(prefix):
            return fine
    return f"{company}-other"


def prompt_number(path):
    """'sdxl_p07.jpg' -> 7; None if the name has no pNN part."""
    import re
    m = re.search(r"_p(\d+)\.", os.path.basename(path))
    return int(m.group(1)) if m else None

VAL_FRAC, TEST_FRAC = 0.15, 0.15


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


def split_list(items, val_frac=VAL_FRAC, test_frac=TEST_FRAC):
    items = items[:]
    random.shuffle(items)
    n = len(items)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val = items[:n_val]
    test = items[n_val:n_val + n_test]
    train = items[n_val + n_test:]
    return train, val, test


def cap(items, n):
    if n and len(items) > n:
        random.shuffle(items)
        return items[:n]
    return items


PRIORITY_PREFIX = "rapidata_"  # prompt-matched images; keep these first when capping


def cap_with_priority(items, n, prefix=PRIORITY_PREFIX):
    """Like cap(), but prompt-matched images are always kept before anything
    else -- they're the ones that stop the model from using content as a
    shortcut, so they shouldn't be the ones a random cap throws away."""
    if not n or len(items) <= n:
        return items
    pri = [f for f in items if os.path.basename(f).startswith(prefix)]
    rest = [f for f in items if not os.path.basename(f).startswith(prefix)]
    random.shuffle(pri)
    random.shuffle(rest)
    return (pri + rest)[:n]


def write_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {len(rows):6d} rows -> {path}")


def main():
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    # ---------- 1. gather phase-1 (real vs AI) pools, by source ----------
    real_pools = {
        "ai_media_classifier": list_images(os.path.join(
            KAGGLE_DIR, "ai_media_classifier_kotayogi", "AIMediaClassifier", "RealImages")),
        "human_vs_ai": list_images(os.path.join(
            KAGGLE_DIR, "human_vs_ai_laxmikanta", "RealArt")),
    }
    ai_unlabeled_pools = {  # phase-1 AI, but no known generator -> phase2_label = NA
        "ai_media_classifier": list_images(os.path.join(
            KAGGLE_DIR, "ai_media_classifier_kotayogi", "AIMediaClassifier", "AIGeneratedImages")),
        "human_vs_ai": list_images(os.path.join(
            KAGGLE_DIR, "human_vs_ai_laxmikanta", "AiArtData")),
    }

    print("Real-image pools found:")
    for k, v in real_pools.items():
        print(f"  {k}: {len(v)}")
    print("AI-image (generator unknown) pools found:")
    for k, v in ai_unlabeled_pools.items():
        print(f"  {k}: {len(v)}")

    # ---------- 2. gather phase-2 (attribution) pools, by class ----------
    # every class that should exist -- from downloaded pools AND self-made classes
    # (e.g. "opensource" has no downloaded_data pool at all yet, only self-made SDXL)
    all_classes = set(ATTR_TRAIN_TO_CLASS.values()) | {c for c in SELF_MADE_TO_CLASS.values() if c}
    attr_pools = {cls: [] for cls in all_classes}
    for folder, cls in ATTR_TRAIN_TO_CLASS.items():
        attr_pools[cls].extend(list_images(os.path.join(ATTR_TRAIN_DIR, folder)))

    print("\nDownloaded attribution-training pools found:")
    fallback_classes = []
    for cls, files in attr_pools.items():
        print(f"  {cls}: {len(files)}")
        if len(files) < MIN_CLASS_SIZE_FOR_TRAIN:
            fallback_classes.append(cls)

    # self-made (processed) pool, split by generator -> class
    self_made_by_class = {}
    for folder, cls in SELF_MADE_TO_CLASS.items():
        files = list_images(os.path.join(SELF_MADE_DIR, folder))
        if cls is None:
            continue
        self_made_by_class.setdefault(cls, []).extend(files)
    dola_files = list_images(os.path.join(SELF_MADE_DIR, "dola"))

    if fallback_classes:
        print(f"\n[WARNING] these attribution classes have < {MIN_CLASS_SIZE_FOR_TRAIN} "
              f"downloaded images: {fallback_classes}")
        print("  Falling back to using the self-made (50-image) set for training on "
              "these classes too.")
        print("  This means they WON'T have a genuine cross-domain test yet -- finish "
              "downloading them and re-run this script when you can.")

    # ---------- 3. build phase-1 splits ----------
    p1_train, p1_val, p1_test = [], [], []
    for source, files in real_pools.items():
        files = cap(files, CAP_PER_SOURCE)
        tr, va, te = split_list(files)
        p1_train += [(f, "real") for f in tr]
        p1_val += [(f, "real") for f in va]
        p1_test += [(f, "real") for f in te]

    for source, files in ai_unlabeled_pools.items():
        files = cap(files, CAP_PER_SOURCE)
        tr, va, te = split_list(files)
        p1_train += [(f, "ai") for f in tr]
        p1_val += [(f, "ai") for f in va]
        p1_test += [(f, "ai") for f in te]

    for cls, files in attr_pools.items():
        files = cap(files, CAP_PER_SOURCE)
        tr, va, te = split_list(files)
        p1_train += [(f, "ai") for f in tr]
        p1_val += [(f, "ai") for f in va]
        p1_test += [(f, "ai") for f in te]

    # if a class fell back, split ITS self-made images into train/val/test too
    # (a smaller, less rigorous split -- but this keeps phase-1 runnable today
    # even before every attribution class has finished downloading)
    for cls in fallback_classes:
        tr, va, te = split_list(self_made_by_class.get(cls, []))
        p1_train += [(f, "ai") for f in tr]
        p1_val += [(f, "ai") for f in va]
        p1_test += [(f, "ai") for f in te]

    print("\nPhase 1 (real vs AI):")
    hdr = ["filepath", "label", "group"]
    for name, rows in (("train", p1_train), ("val", p1_val), ("test", p1_test)):
        write_csv(os.path.join(MANIFEST_DIR, f"phase1_{name}.csv"),
                  [(f, lab, lab) for f, lab in rows], hdr)

    # ---------- 4. build phase-2 splits (per specific model) ----------
    from collections import Counter, defaultdict
    rapidata_models = load_rapidata_models(ATTR_TRAIN_DIR)

    by_fine = defaultdict(list)  # (company, fine) -> files
    for cls, files in attr_pools.items():
        for f in files:
            by_fine[(cls, fine_label(f, cls, rapidata_models))].append(f)
    for cls in fallback_classes:
        for f in self_made_by_class.get(cls, []):
            by_fine[(cls, fine_label(f, cls, rapidata_models))].append(f)

    # merge models with too few images into "<company>-other"
    merged = defaultdict(list)
    for (cls, fine), files in by_fine.items():
        key = (cls, fine) if len(files) >= MIN_FINE_LABEL else (cls, f"{cls}-other")
        merged[key].extend(files)

    p2_train, p2_val, p2_test = [], [], []
    for (cls, fine), files in sorted(merged.items()):
        files = cap(files, CAP_PER_FINE_LABEL)
        tr, va, te = split_list(files)
        p2_train += [(f, fine, cls) for f in tr]
        p2_val += [(f, fine, cls) for f in va]
        p2_test += [(f, fine, cls) for f in te]

    print(f"\nPhase 2 training counts per specific model (each capped at {CAP_PER_FINE_LABEL}):")
    fine_counts = Counter((cls, fine) for _, fine, cls in p2_train)
    for cls in sorted({c for c, _ in fine_counts}):
        parts = ", ".join(f"{fine}={n}" for (c, fine), n in sorted(fine_counts.items()) if c == cls)
        total = sum(n for (c, _), n in fine_counts.items() if c == cls)
        print(f"  {cls:11s} total={total:5d}  [{parts}]")

    print("\nPhase 2 (generator attribution):")
    hdr = ["filepath", "label", "group"]
    write_csv(os.path.join(MANIFEST_DIR, "phase2_train.csv"), p2_train, hdr)
    write_csv(os.path.join(MANIFEST_DIR, "phase2_val.csv"), p2_val, hdr)
    write_csv(os.path.join(MANIFEST_DIR, "phase2_test.csv"), p2_test, hdr)

    # ---------- 5. cross-domain sets (self-made, never trained on) ----------
    # odd prompt numbers  -> *_crossdomain_val (used to pick the winning model)
    # even prompt numbers -> *_crossdomain_test (only used to report the final score)
    xd = {"phase1": {"val": [], "test": []}, "phase2": {"val": [], "test": []}}
    for folder, cls in SELF_MADE_TO_CLASS.items():
        if cls in fallback_classes:
            continue  # these were moved into TRAIN above -- can't also be held out
        for f in list_images(os.path.join(SELF_MADE_DIR, folder)):
            n = prompt_number(f)
            part = "val" if (n is not None and n % 2 == 1) else "test"
            xd["phase1"][part].append((f, "ai", "ai"))  # every self-made image is AI-generated
            if cls is not None:
                # specific model isn't known for these, only the company -> label = company
                xd["phase2"][part].append((f, cls, cls))

    print("\nCross-domain sets (self-made, never trained on):")
    for phase in ("phase1", "phase2"):
        for part in ("val", "test"):
            write_csv(os.path.join(MANIFEST_DIR, f"{phase}_crossdomain_{part}.csv"),
                      xd[phase][part], hdr)

    # ---------- 6. Dola probe (no ground-truth attribution label) ----------
    dola_rows = [(f,) for f in dola_files]
    print("\nDola probe (attribution ground truth unknown):")
    write_csv(os.path.join(MANIFEST_DIR, "dola_probe.csv"), dola_rows, ["filepath"])

    print("\nDone. Manifests are under:", MANIFEST_DIR)


if __name__ == "__main__":
    main()