"""Train & compare 3 models for phase 1 (real vs AI) and phase 2 (generator
attribution), pick the best of each, and save everything the web app needs.

Models compared, for EACH phase:
    A. CLIP (frozen) + Logistic Regression, with a small C search
    B. CLIP (frozen) + MLP head, early stopping
    C. ResNet50 fine-tuned on NATIVE-RESOLUTION 224px crops (no downscaling,
       which erases the fine pixel-level traces generators leave), trained
       with random JPEG re-compression / rescaling for robustness, and scored
       at test time by averaging 5 crops

Phase 2 trains on SPECIFIC MODEL labels (nano-banana, gpt-4o, sdxl, flux,
...). Predictions are merged into companies (google / openai / bytedance /
opensource) by summing probabilities. All reported phase-2 scores are at
the company level, since that's what the website shows.

Model selection uses ONLY the cross-domain VAL set (your own images,
odd prompt numbers). The cross-domain TEST set (even numbers) is never used for any
decision, so its score is the honest one to report.

Usage:
    pip install transformers torch torchvision scikit-learn pillow joblib
    python train_models.py
Tunable via env vars: BATCH_SIZE, RESNET_EPOCHS, RESNET_PATIENCE,
MAX_TRAIN_FOR_RESNET, CLIP_MODEL_ID, RESNET_ARCH, MLP_EPOCHS, MLP_PATIENCE,
NUM_WORKERS
"""
import csv
import io
import json
import os
import random
import time
from collections import Counter

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tvmodels
from torchvision import transforms as T

MANIFEST_DIR = os.environ.get("MANIFEST_DIR", "manifests")
OUT_DIR = os.environ.get("MODEL_OUT_DIR", "model_out")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "6"))
RESNET_EPOCHS = int(os.environ.get("RESNET_EPOCHS", "30"))
RESNET_PATIENCE = int(os.environ.get("RESNET_PATIENCE", "6"))
MAX_TRAIN_FOR_RESNET = int(os.environ.get("MAX_TRAIN_FOR_RESNET", "20000"))
MLP_EPOCHS = int(os.environ.get("MLP_EPOCHS", "80"))
MLP_PATIENCE = int(os.environ.get("MLP_PATIENCE", "12"))
CLIP_MODEL_ID = os.environ.get("CLIP_MODEL_ID", "openai/clip-vit-large-patch14")
RESNET_ARCH = os.environ.get("RESNET_ARCH", "resnet50")  # "resnet18" or "resnet50"
PREPROCESS_VERSION = "v2"  # bump whenever normalize_image() changes (invalidates CLIP cache)
CROP = 224
RESNET_INPUT = "native_multicrop"  # recorded in summary.json so serving matches training

RESNET_REGISTRY = {
    "resnet18": (tvmodels.resnet18, tvmodels.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (tvmodels.resnet50, tvmodels.ResNet50_Weights.IMAGENET1K_V2),
}

os.makedirs(OUT_DIR, exist_ok=True)


def normalize_image(pil_image, jpeg_quality=90):
    """Applied to EVERY image, from EVERY source, before it reaches any model:
    one shared JPEG re-encode, so compression/format differences between
    datasets can't be used as a shortcut. Must match webapp/model_utils.py."""
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, "JPEG", quality=jpeg_quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_manifest(name):
    path = os.path.join(MANIFEST_DIR, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:  # older manifests have no group column: group = label
        r.setdefault("group", r["label"])
        if not r["group"]:
            r["group"] = r["label"]
    return rows


# ---------------------------------------------------------------------------
# CLIP feature extraction (shared by models A and B, both phases)
# ---------------------------------------------------------------------------

def extract_clip_features(filepaths, cache_path):
    """Returns {filepath: np.array[512]}. Cached to disk so re-runs are free."""
    cache = {}
    if os.path.exists(cache_path):
        cache = dict(np.load(cache_path, allow_pickle=True).item())

    todo = [p for p in filepaths if p not in cache]
    if not todo:
        log(f"CLIP features: all {len(filepaths)} already cached.")
        return cache

    log(f"CLIP features ({CLIP_MODEL_ID}): extracting {len(todo)} new images "
        f"({len(filepaths) - len(todo)} already cached)...")

    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

    class ImgDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            p = self.paths[i]
            try:
                img = Image.open(p).convert("RGB")
                img = normalize_image(img)
            except Exception as e:
                log(f"  [warn] failed to open {p}: {e}; using a blank image instead")
                img = Image.new("RGB", (224, 224))
            return p, img

    def collate(batch):
        paths, imgs = zip(*batch)
        return list(paths), list(imgs)

    loader = DataLoader(ImgDataset(todo), batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, collate_fn=collate)

    done = 0
    with torch.no_grad():
        for paths, imgs in loader:
            inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
            feats = model.get_image_features(**inputs)
            if not isinstance(feats, torch.Tensor):
                # some transformers versions return a ModelOutput here instead
                # of a plain tensor -- pull the embedding out of it either way
                raw = feats
                feats = getattr(raw, "image_embeds", None)
                if feats is None:
                    feats = getattr(raw, "pooler_output", None)
                if feats is None:
                    raise TypeError(
                        f"Unexpected return type from get_image_features(): {type(raw)}"
                    )
            feats = feats.cpu().numpy()
            for p, f in zip(paths, feats):
                cache[p] = f
            done += len(paths)
            if done % (BATCH_SIZE * 10) < BATCH_SIZE:
                log(f"  {done}/{len(todo)}")

    np.save(cache_path, cache)
    del model
    torch.cuda.empty_cache()
    return cache


# ---------------------------------------------------------------------------
# Class weights: every COMPANY gets equal total weight, and inside a company
# every specific model gets an equal share. Without this, a company that
# happens to have more specific-model labels (e.g. opensource = sdxl + flux
# + sd3) would collect more probability when they're summed up.
# ---------------------------------------------------------------------------

def compute_class_weights(y, label_names, label_to_group):
    n = len(label_names)
    counts = np.bincount(y, minlength=n).astype(float)
    present = [i for i in range(n) if counts[i] > 0]
    groups = sorted({label_to_group[label_names[i]] for i in present})
    per_group = Counter(label_to_group[label_names[i]] for i in present)
    w = np.ones(n)
    for i in present:
        g = label_to_group[label_names[i]]
        w[i] = len(y) / (len(groups) * per_group[g] * counts[i])
    return w


# ---------------------------------------------------------------------------
# Model A: CLIP + Logistic Regression
# ---------------------------------------------------------------------------

def train_logreg(X_train, y_train, X_val, y_val, class_w):
    cw = {i: float(class_w[i]) for i in np.unique(y_train)}
    best = (None, -1, -1)
    for C in (0.01, 0.1, 1.0, 10.0, 100.0):
        clf = LogisticRegression(max_iter=3000, class_weight=cw, C=C)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        f1 = f1_score(y_val, pred, average="macro")
        if f1 > best[2]:
            best = (clf, accuracy_score(y_val, pred), f1)
    return best


# ---------------------------------------------------------------------------
# Model B: CLIP + MLP head
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """Must match webapp/model_utils.py's MLPHead exactly."""

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 384), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(384, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, n_classes, class_w):
    Xt = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)

    model = MLPHead(Xt.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32).to(DEVICE))

    best_state, best_f1, best_acc, bad = None, -1, -1, 0
    for _ in range(MLP_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xv).argmax(1).cpu().numpy()
        f1 = f1_score(y_val, pred, average="macro")
        if f1 > best_f1:
            best_f1, best_acc, bad = f1, accuracy_score(y_val, pred), 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= MLP_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc, best_f1


# ---------------------------------------------------------------------------
# Model C: ResNet on native-resolution crops
# ---------------------------------------------------------------------------

IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TO_TENSOR = T.Compose([T.ToTensor(), IMAGENET_NORM])
TRAIN_CROP = T.Compose([
    T.RandomCrop(CROP, pad_if_needed=True),
    T.RandomHorizontalFlip(),
])


def ensure_min_side(img, side=CROP):
    """Upscale only if an image is smaller than one crop (rare)."""
    w, h = img.size
    if min(w, h) >= side:
        return img
    s = side / min(w, h)
    return img.resize((max(side, round(w * s)), max(side, round(h * s))), Image.BICUBIC)


def robustness_augment(img, rng):
    """Mimic what happens to images in the wild: resizing and re-saving with
    different JPEG quality. Teaches the model traces that survive this."""
    if rng.random() < 0.5:
        w, h = img.size
        s = max(rng.uniform(0.5, 1.0), CROP / min(w, h))
        if s < 1.0:
            img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    if rng.random() < 0.5:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(60, 95))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def five_crops(img):
    """4 corners + centre at native resolution -> tensor [5, 3, CROP, CROP]."""
    img = ensure_min_side(img)
    return torch.stack([TO_TENSOR(c) for c in T.FiveCrop(CROP)(img)])


def load_image(path):
    try:
        return normalize_image(Image.open(path).convert("RGB"))
    except Exception:
        return Image.new("RGB", (CROP, CROP))


class TrainCropDataset(Dataset):
    def __init__(self, rows, label2idx):
        self.rows, self.label2idx = rows, label2idx
        # use the module-level RNG: _worker_seed reseeds it per worker, whereas a
        # Random() object created here would be copied identically into every worker
        self.rng = random

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = robustness_augment(load_image(r["filepath"]), self.rng)
        return TO_TENSOR(TRAIN_CROP(ensure_min_side(img))), self.label2idx[r["label"]]


class EvalCropDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return five_crops(load_image(self.rows[i]["filepath"]))


def _worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def resnet_predict_proba(model, rows):
    loader = DataLoader(EvalCropDataset(rows), batch_size=max(1, BATCH_SIZE // 4),
                        shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    out = []
    with torch.no_grad():
        for crops in loader:  # [B, 5, 3, H, W]
            b = crops.shape[0]
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                logits = model(crops.view(-1, 3, CROP, CROP).to(DEVICE))
            probs = torch.softmax(logits.float(), dim=1).view(b, 5, -1).mean(1)
            out.append(probs.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.fc.out_features))


def train_resnet(train_rows, val_rows, label_names, label2idx, class_w):
    ctor, weights = RESNET_REGISTRY[RESNET_ARCH]
    model = ctor(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(label_names))
    model = model.to(DEVICE)

    ds = TrainCropDataset(train_rows, label2idx)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, worker_init_fn=_worker_seed, drop_last=False)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32).to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=RESNET_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    y_val = [label2idx[r["label"]] for r in val_rows]

    best_state, best_f1, best_acc, bad = None, -1, -1, 0
    for epoch in range(RESNET_EPOCHS):
        model.train()
        t0 = time.time()
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                loss = loss_fn(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        pred = resnet_predict_proba(model, val_rows).argmax(1)
        f1 = f1_score(y_val, pred, average="macro")
        acc = accuracy_score(y_val, pred)
        log(f"  {RESNET_ARCH} epoch {epoch + 1}/{RESNET_EPOCHS}: val_acc={acc:.3f} "
            f"val_f1={f1:.3f} ({time.time() - t0:.0f}s)")
        if f1 > best_f1:
            best_f1, best_acc, bad = f1, acc, 0
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= RESNET_PATIENCE:
                log(f"  no improvement for {RESNET_PATIENCE} epochs, stopping early")
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc, best_f1


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def run_phase(phase_name, clip_cache_path, label_order=None):
    log(f"\n===== {phase_name} =====")
    train_rows = read_manifest(f"{phase_name}_train.csv")
    val_rows = read_manifest(f"{phase_name}_val.csv")
    test_rows = read_manifest(f"{phase_name}_test.csv")
    xd_val_rows = read_manifest(f"{phase_name}_crossdomain_val.csv")
    xd_test_rows = read_manifest(f"{phase_name}_crossdomain_test.csv")

    if not train_rows or not val_rows:
        log(f"  [skip] {phase_name}: not enough data (train={len(train_rows)}, val={len(val_rows)}).")
        return None

    labelled = train_rows + val_rows + test_rows
    label_names = label_order or sorted({r["label"] for r in labelled})
    label2idx = {n: i for i, n in enumerate(label_names)}
    label_to_group = {}
    for r in labelled:
        label_to_group[r["label"]] = r["group"]
    for n in label_names:
        label_to_group.setdefault(n, n)
    groups = set(label_to_group.values())
    group_names = [g for g in label_order if g in groups] if label_order else sorted(groups)
    G = np.zeros((len(label_names), len(group_names)))
    for i, n in enumerate(label_names):
        G[i, group_names.index(label_to_group[n])] = 1.0

    all_paths = sorted({r["filepath"] for r in labelled + xd_val_rows + xd_test_rows})
    feats = extract_clip_features(all_paths, clip_cache_path)

    def X_of(rows):
        return np.stack([feats[r["filepath"]] for r in rows])

    Xtr = X_of(train_rows)
    ytr = np.array([label2idx[r["label"]] for r in train_rows])
    Xval = X_of(val_rows)
    yval = np.array([label2idx[r["label"]] for r in val_rows])
    class_w = compute_class_weights(ytr, label_names, label_to_group)

    log(f"  train={len(train_rows)} val={len(val_rows)}  groups={group_names}")
    if len(label_names) != len(group_names):
        per = Counter(r["label"] for r in train_rows)
        log("  specific-model labels: " + ", ".join(f"{n}({per.get(n, 0)})" for n in label_names))

    results = {}
    log("  Training Model A: CLIP + Logistic Regression ...")
    clfA, accA, f1A = train_logreg(Xtr, ytr, Xval, yval, class_w)
    results["clip_logreg"] = {"val_acc": accA, "val_f1": f1A}
    log(f"    val_acc={accA:.3f} val_f1={f1A:.3f}")

    log("  Training Model B: CLIP + MLP ...")
    clfB, accB, f1B = train_mlp(Xtr, ytr, Xval, yval, len(label_names), class_w)
    results["clip_mlp"] = {"val_acc": accB, "val_f1": f1B}
    log(f"    val_acc={accB:.3f} val_f1={f1B:.3f}")

    log(f"  Training Model C: {RESNET_ARCH} on native-resolution crops ...")
    rn_rows = train_rows[:]
    random.Random(42).shuffle(rn_rows)
    clfC, accC, f1C = train_resnet(rn_rows[:MAX_TRAIN_FOR_RESNET], val_rows,
                                   label_names, label2idx, class_w)
    results["resnet_finetune"] = {"val_acc": accC, "val_f1": f1C}
    log(f"    val_acc={accC:.3f} val_f1={f1C:.3f}")

    def proba(model_name, rows):
        """[n, n_specific_labels] probabilities for any of the 3 models."""
        if not rows:
            return np.zeros((0, len(label_names)))
        if model_name == "resnet_finetune":
            return resnet_predict_proba(clfC, rows)
        X = X_of(rows)
        if model_name == "clip_mlp":
            with torch.no_grad():
                logits = clfB(torch.tensor(X, dtype=torch.float32).to(DEVICE))
            return torch.softmax(logits, 1).cpu().numpy()
        p = np.zeros((len(rows), len(label_names)))
        p[:, clfA.classes_] = clfA.predict_proba(X)  # classes_ are label indices
        return p

    def evaluate(rows, tag, model_name, detail=True):
        if not rows:
            return None
        p = proba(model_name, rows)
        gp = p @ G
        pred_g = [group_names[i] for i in gp.argmax(1)]
        true_g = [r["group"] for r in rows]
        acc = accuracy_score(true_g, pred_g)
        f1 = f1_score(true_g, pred_g, average="macro", labels=sorted(set(true_g)))
        res = {"n": len(rows), "acc": acc, "macro_f1_over_true_groups": f1}
        line = f"    [{tag}] {model_name}: n={len(rows)} acc={acc:.3f} macro_f1={f1:.3f}"
        # specific-model accuracy, where the rows actually have a specific label
        fine_rows = [i for i, r in enumerate(rows) if r["label"] in label2idx and r["label"] != r["group"]]
        if fine_rows:
            fa = np.mean([p[i].argmax() == label2idx[rows[i]["label"]] for i in fine_rows])
            res["specific_model_acc"] = float(fa)
            line += f" (specific-model acc={fa:.3f})"
        log(line)
        if detail and len(group_names) > 2:
            conf = {}
            pred_fine = [label_names[i] for i in p.argmax(1)]
            for g in sorted(set(true_g)):
                idx = [i for i, t in enumerate(true_g) if t == g]
                pc = Counter(pred_g[i] for i in idx)
                fc = Counter(pred_fine[i] for i in idx)
                conf[g] = dict(pc)
                log(f"      true={g:11s} (n={len(idx)}): "
                    + ", ".join(f"{k}={v}" for k, v in pc.most_common())
                    + "   | closest specific model: "
                    + ", ".join(f"{k}={v}" for k, v in fc.most_common(4)))
            res["confusion"] = conf
        return res

    if xd_val_rows:
        log("  Cross-domain VAL (odd prompts), all 3 candidates -- this picks the winner:")
        for name in results:
            r = evaluate(xd_val_rows, "xd-val", name, detail=False)
            results[name]["xdomain_val_acc"] = r["acc"]
        best_name = max(results, key=lambda k: (results[k]["xdomain_val_acc"], results[k]["val_f1"]))
        log(f"  >>> Best model for {phase_name}: {best_name} "
            f"(xd-val acc={results[best_name]['xdomain_val_acc']:.3f})")
    else:
        best_name = max(results, key=lambda k: results[k]["val_f1"])
        log(f"  >>> Best model for {phase_name}: {best_name} (by in-domain val_f1, "
            f"no cross-domain val set found)")

    test_metrics = evaluate(test_rows, "held-out test (same datasets)", best_name)
    xd_val_metrics = evaluate(xd_val_rows, "cross-domain VAL (odd prompts)", best_name)
    xd_test_metrics = evaluate(xd_test_rows, "cross-domain TEST (even prompts, report this)", best_name)

    phase_out = os.path.join(OUT_DIR, phase_name)
    os.makedirs(phase_out, exist_ok=True)
    if best_name == "resnet_finetune":
        torch.save(clfC.state_dict(), os.path.join(phase_out, "resnet.pt"))
    elif best_name == "clip_mlp":
        torch.save(clfB.state_dict(), os.path.join(phase_out, "mlp.pt"))
    else:
        import joblib
        joblib.dump(clfA, os.path.join(phase_out, "logreg.joblib"))

    summary = {
        "best_model": best_name,
        "label_names": label_names,
        "label_to_group": label_to_group,
        "group_names": group_names,
        "feature_dim": int(Xtr.shape[1]),
        "clip_model_id": CLIP_MODEL_ID,
        "resnet_arch": RESNET_ARCH,
        "resnet_input": RESNET_INPUT,
        "val": results,
        "test": test_metrics,
        "cross_domain_val": xd_val_metrics,
        "cross_domain_test": xd_test_metrics,
    }
    with open(os.path.join(phase_out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"  Saved winning model + summary.json -> {phase_out}")
    del clfC
    torch.cuda.empty_cache()
    return summary


def main():
    t0 = time.time()
    tag = CLIP_MODEL_ID.replace("/", "_") + "_" + PREPROCESS_VERSION
    phases = os.environ.get("PHASES", "phase1,phase2").split(",")  # e.g. PHASES=phase2
    if "phase1" in phases:
        run_phase("phase1", os.path.join(OUT_DIR, f"clip_feats_phase1_{tag}.npy"), label_order=["real", "ai"])
    if "phase2" in phases:
        run_phase("phase2", os.path.join(OUT_DIR, f"clip_feats_phase2_{tag}.npy"))
    log(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()