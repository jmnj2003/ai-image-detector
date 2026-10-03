"""FastAPI backend: upload an image, get back P(AI) and, if AI, the
attribution breakdown across generators.

Usage:
    pip install fastapi uvicorn python-multipart pillow torch torchvision transformers joblib scikit-learn
    export MODEL_DIR=/path/to/model_out   # the folder train_models.py wrote to
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
import io
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model_utils import PhaseModel

MODEL_DIR = os.environ.get("MODEL_DIR", "model_out")
# Below this P(AI), we don't even bother running the attribution model --
# it's not meaningful for an image the model is confident is real.
AI_THRESHOLD = float(os.environ.get("AI_THRESHOLD", "0.5"))
# Below this confidence in the top attribution class, report "unknown" instead
# of a specific generator -- an untrained guess isn't worth showing as fact.
ATTRIBUTION_CONFIDENCE_FLOOR = float(os.environ.get("ATTRIBUTION_CONFIDENCE_FLOOR", "0.35"))

app = FastAPI(title="AI Content Detector")

phase1 = None
phase2 = None


@app.on_event("startup")
def load_models():
    global phase1, phase2
    p1_dir = os.path.join(MODEL_DIR, "phase1")
    p2_dir = os.path.join(MODEL_DIR, "phase2")
    if not os.path.exists(os.path.join(p1_dir, "summary.json")):
        raise RuntimeError(
            f"No trained phase-1 model found at {p1_dir}. Run train_models.py first."
        )
    phase1 = PhaseModel(p1_dir)
    if os.path.exists(os.path.join(p2_dir, "summary.json")):
        phase2 = PhaseModel(p2_dir)
    else:
        print(f"[warn] no phase-2 (attribution) model at {p2_dir} -- "
              f"/predict will only report real-vs-AI, not source attribution.")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read that as an image.")

    p1_probs = phase1.predict_proba(img)
    p_ai = p1_probs.get("ai", 0.0)

    result = {
        "p_ai": round(p_ai, 4),
        "is_ai": p_ai >= AI_THRESHOLD,
        "attribution": None,
    }

    if p_ai >= AI_THRESHOLD and phase2 is not None:
        fine = phase2.predict_proba(img)                    # per specific model
        p2_probs = phase2.predict_group_proba(img, fine)    # summed per company
        top_class = max(p2_probs, key=p2_probs.get)
        top_conf = p2_probs[top_class]
        top_model = max(fine, key=fine.get)
        result["attribution"] = {
            "probabilities": {k: round(v, 4) for k, v in p2_probs.items()},
            "top_guess": top_class if top_conf >= ATTRIBUTION_CONFIDENCE_FLOOR else "unknown",
            "top_confidence": round(top_conf, 4),
            "closest_model": top_model,
            "closest_model_confidence": round(fine[top_model], 4),
        }

    return result


@app.get("/health")
def health():
    return {
        "phase1_model": phase1.best_model if phase1 else None,
        "phase2_model": phase2.best_model if phase2 else None,
    }


# serve the single-page frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(static_dir, "index.html"))