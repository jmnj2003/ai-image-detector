"""Loads whichever model won each phase (from train_models.py's summary.json)
and runs inference on a PIL image. Handles all 3 possible model types so the
serving code doesn't care which one actually won -- it reads the backbone
(CLIP model id, feature dim, ResNet architecture) straight from summary.json,
so it always matches whatever train_models.py actually trained with.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models as tvmodels
from torchvision import transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESNET_TRANSFORM_EVAL = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def normalize_image(pil_image, jpeg_quality=90):
    """MUST match train_models.py's normalize_image exactly -- this removes
    a format/compression shortcut the model was trained without, so skipping
    it here would mean live predictions see a distribution the model never
    saw during training (train/serve skew)."""
    import io
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, "JPEG", quality=jpeg_quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

RESNET_REGISTRY = {
    "resnet18": tvmodels.resnet18,
    "resnet50": tvmodels.resnet50,
}


class MLPHead(nn.Module):
    """Must match train_models.py's MLPHead exactly -- same layer sizes --
    or loading the saved state_dict will fail."""

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 384), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(384, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class _SharedClip:
    """CLIP is loaded at most once PER MODEL ID and shared across phases --
    if phase 1 and phase 2 happen to use different CLIP backbones, both get
    cached separately rather than silently reusing the wrong one."""
    _models = {}  # model_id -> (model, processor)

    @classmethod
    def get(cls, model_id):
        if model_id not in cls._models:
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained(model_id).to(DEVICE).eval()
            processor = CLIPProcessor.from_pretrained(model_id)
            cls._models[model_id] = (model, processor)
        return cls._models[model_id]

    @classmethod
    def embed(cls, pil_image, model_id):
        model, processor = cls.get(model_id)
        with torch.no_grad():
            inputs = processor(images=[pil_image], return_tensors="pt").to(DEVICE)
            feat = model.get_image_features(**inputs)
            if not isinstance(feat, torch.Tensor):
                raw = feat
                feat = getattr(raw, "image_embeds", None)
                if feat is None:
                    feat = getattr(raw, "pooler_output", None)
                if feat is None:
                    raise TypeError(
                        f"Unexpected return type from get_image_features(): {type(raw)}"
                    )
        return feat.cpu().numpy()[0]


CROP = 224
IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TO_TENSOR = T.Compose([T.ToTensor(), IMAGENET_NORM])


def five_crops(img):
    """Must match train_models.py: 4 corners + centre at NATIVE resolution
    (no downscaling), so the model sees the same kind of input it trained on."""
    w, h = img.size
    if min(w, h) < CROP:
        s = CROP / min(w, h)
        img = img.resize((max(CROP, round(w * s)), max(CROP, round(h * s))), Image.BICUBIC)
    return torch.stack([TO_TENSOR(c) for c in T.FiveCrop(CROP)(img)])


class PhaseModel:
    """Loads one phase's winning model (from model_out_dir/<phase>/).

    .predict_proba(img)       -> {specific_label: probability}
    .predict_group_proba(img) -> {group: probability}, specific-model
                                 probabilities summed per company. For phase 1
                                 the groups are just "real"/"ai".
    """

    def __init__(self, phase_dir):
        with open(os.path.join(phase_dir, "summary.json")) as f:
            summary = json.load(f)
        self.best_model = summary["best_model"]
        self.label_names = summary["label_names"]
        self.label_to_group = summary.get("label_to_group") or {n: n for n in self.label_names}
        self.group_names = summary.get("group_names") or list(dict.fromkeys(
            self.label_to_group[n] for n in self.label_names))
        self.clip_model_id = summary.get("clip_model_id", "openai/clip-vit-large-patch14")
        self.feature_dim = summary.get("feature_dim", 768)
        self.resnet_arch = summary.get("resnet_arch", "resnet50")
        self.resnet_input = summary.get("resnet_input", "resize_center")  # older runs
        self.summary = summary

        if self.best_model == "clip_logreg":
            import joblib
            self.clf = joblib.load(os.path.join(phase_dir, "logreg.joblib"))
        elif self.best_model == "clip_mlp":
            self.clf = MLPHead(self.feature_dim, len(self.label_names)).to(DEVICE)
            self.clf.load_state_dict(torch.load(
                os.path.join(phase_dir, "mlp.pt"), map_location=DEVICE))
            self.clf.eval()
        elif self.best_model == "resnet_finetune":
            ctor = RESNET_REGISTRY[self.resnet_arch]
            self.clf = ctor(weights=None)
            self.clf.fc = nn.Linear(self.clf.fc.in_features, len(self.label_names))
            self.clf.load_state_dict(torch.load(
                os.path.join(phase_dir, "resnet.pt"), map_location=DEVICE))
            self.clf = self.clf.to(DEVICE).eval()
        else:
            raise ValueError(f"Unknown model type in summary.json: {self.best_model}")

    def predict_proba(self, pil_image):
        pil_image = normalize_image(pil_image.convert("RGB"))

        if self.best_model == "resnet_finetune":
            if self.resnet_input == "native_multicrop":
                x = five_crops(pil_image).to(DEVICE)  # [5, 3, 224, 224]
                with torch.no_grad():
                    probs = torch.softmax(self.clf(x), dim=1).mean(0).cpu().numpy()
            else:
                x = RESNET_TRANSFORM_EVAL(pil_image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    probs = torch.softmax(self.clf(x), dim=1).cpu().numpy()[0]

        elif self.best_model == "clip_mlp":
            feat = _SharedClip.embed(pil_image, self.clip_model_id)
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(self.clf(x), dim=1).cpu().numpy()[0]

        else:  # clip_logreg -- its classes_ are label indices (or names, in old runs)
            feat = _SharedClip.embed(pil_image, self.clip_model_id)
            raw = self.clf.predict_proba(feat.reshape(1, -1))[0]
            probs = np.zeros(len(self.label_names))
            for c, pr in zip(self.clf.classes_, raw):
                idx = int(c) if not isinstance(c, str) else self.label_names.index(c)
                probs[idx] = pr

        return {name: float(p) for name, p in zip(self.label_names, probs)}

    def predict_group_proba(self, pil_image, fine=None):
        fine = fine if fine is not None else self.predict_proba(pil_image)
        out = {g: 0.0 for g in self.group_names}
        for name, p in fine.items():
            out[self.label_to_group.get(name, name)] += p
        return out