import io
import os
import pickle
import urllib.error
import urllib.request
import warnings
import jwt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torchvision import models, transforms
from PIL import Image
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Auth (shared JWT secret with the Node/Express server) ───────────────────
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable is not set. Refusing to start.")


def require_auth(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.split(" ", 1)[1]
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ─── Load XGBoost Model ───────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, "..", "alzheimer_xgboost_model.pkl"), "rb") as f:
    xgb_artifacts = pickle.load(f)

xgb_model     = xgb_artifacts["model"]
xgb_scaler    = xgb_artifacts["scaler"]
feature_names = xgb_artifacts["feature_names"]
threshold     = xgb_artifacts["best_threshold"]

# ─── Load reference dataset (for "fetch by record number" autofill) ──────────
_PATIENT_DF = pd.read_csv(os.path.join(BASE_DIR, "..", "alzheimers_disease_data.csv"))

# ─── Load CNN Model ───────────────────────────────────────────────────────────
# This file is ~94MB — too large for GitHub's web-upload UI (25MB limit) and
# excluded from git via .gitignore. Host it as a GitHub Release asset (or any
# direct-download URL) and set CNN_MODEL_URL so the server fetches it on first
# boot; otherwise just place the file at this path manually.
_CNN_MODEL_PATH = os.environ.get("CNN_MODEL_PATH") or os.path.join(BASE_DIR, "alzheimer_cnn_model.pkl")
_CNN_MODEL_URL = os.environ.get("CNN_MODEL_URL")
# Opt-in: defer CNN loading to the first /predict-mri call instead of failing
# the whole app at startup if the model isn't resolved yet. Off by default —
# Render/HF deployments should keep failing fast on misconfiguration.
_LAZY_CNN_LOAD = os.environ.get("LAZY_CNN_LOAD", "false").lower() == "true"

_cnn_cache = {}


def _build_resnet50(num_classes: int) -> nn.Module:
    net = models.resnet50(weights=None)
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return net


def _load_cnn():
    if not os.path.exists(_CNN_MODEL_PATH):
        if not _CNN_MODEL_URL:
            raise RuntimeError(
                "alzheimer_cnn_model.pkl not found and CNN_MODEL_URL is not set. "
                "Either place the model file at alzimer_backend/alzheimer_cnn_model.pkl, "
                "or set CNN_MODEL_URL to a direct-download link (e.g. a GitHub Release asset)."
            )
        print(f"alzheimer_cnn_model.pkl not found locally — downloading from: {_CNN_MODEL_URL}")
        try:
            req = urllib.request.Request(_CNN_MODEL_URL, headers={"User-Agent": "mindbridge-deploy/1.0"})
            with urllib.request.urlopen(req) as resp, open(_CNN_MODEL_PATH, "wb") as out:
                out.write(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"Failed to download alzheimer_cnn_model.pkl from CNN_MODEL_URL "
                f"(HTTP {e.code} {e.reason}) — URL was: {_CNN_MODEL_URL}. "
                "Check that this is the asset's direct download link (not the release "
                "page), the tag/filename are exact, and the repo/asset is public."
            ) from e
        downloaded_size = os.path.getsize(_CNN_MODEL_PATH)
        print(f"Download complete ({downloaded_size} bytes).")

    with open(_CNN_MODEL_PATH, "rb") as f:
        cnn_payload = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # Keep thread-pool overhead (and its memory footprint) low on small/shared
        # CPU-only instances — we only ever run one inference at a time per request.
        torch.set_num_threads(1)

    model = _build_resnet50(cnn_payload["num_classes"])
    model.load_state_dict(cnn_payload["model_state_dict"])
    model.to(device)
    model.eval()

    # The state dict's values are now copied into the model's own parameters —
    # drop this ~100MB duplicate instead of holding it for the process lifetime.
    del cnn_payload["model_state_dict"]

    img_size = cnn_payload["img_size"]
    classes = cnn_payload["classes"]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(cnn_payload["normalize_mean"], cnn_payload["normalize_std"]),
    ])

    return {
        "model": model,
        "device": device,
        "transform": transform,
        "classes": classes,
        "img_size": img_size,
        "val_acc": cnn_payload.get("best_val_acc"),
    }


def _get_cnn(force: bool = True):
    """Returns the cached CNN bundle, loading it on first call. If `force` is
    False and lazy-loading is enabled, returns None instead of loading when
    it hasn't been loaded yet (used by /health to avoid triggering a load)."""
    if "data" not in _cnn_cache:
        if _LAZY_CNN_LOAD and not force:
            return None
        _cnn_cache["data"] = _load_cnn()
    return _cnn_cache["data"]


if not _LAZY_CNN_LOAD:
    _get_cnn()

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Alzheimer Prediction API", version="2.0.0")

_cors_origins = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:5000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Input Schema (text/clinical features) ────────────────────────────────────
class PatientInput(BaseModel):
    Age: int
    Gender: int
    Ethnicity: int
    EducationLevel: int
    BMI: float
    Smoking: int
    AlcoholConsumption: float
    PhysicalActivity: float
    DietQuality: float
    SleepQuality: float
    FamilyHistoryAlzheimers: int
    CardiovascularDisease: int
    Diabetes: int
    Depression: int
    HeadInjury: int
    Hypertension: int
    SystolicBP: int
    DiastolicBP: int
    CholesterolTotal: float
    CholesterolLDL: float
    CholesterolHDL: float
    CholesterolTriglycerides: float
    MMSE: float
    FunctionalAssessment: float
    MemoryComplaints: int
    BehavioralProblems: int
    ADL: float
    Confusion: int
    Disorientation: int
    PersonalityChanges: int
    DifficultyCompletingTasks: int
    Forgetfulness: int

# ─── Text/Clinical Prediction Endpoint ────────────────────────────────────────
@app.post("/predict")
def predict(patient: PatientInput, _user: dict = Depends(require_auth)):
    try:
        data = patient.model_dump()
        df = pd.DataFrame([data])

        df["CognitiveScore"] = (
            df["MMSE"] + df["FunctionalAssessment"] + df["ADL"]
        ) / 3

        behavioral_cols = [
            "MemoryComplaints", "BehavioralProblems", "Confusion",
            "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
            "Forgetfulness",
        ]
        df["BehavioralSymptomCount"] = df[behavioral_cols].sum(axis=1)

        df["CardioRiskScore"] = (
            df["CardiovascularDisease"] + df["Diabetes"] +
            df["Hypertension"] + df["Smoking"]
        )

        df["CholesterolRatio"] = df["CholesterolLDL"] / (df["CholesterolHDL"] + 1e-5)

        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[59, 65, 70, 75, 80, 91],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

        df = df[feature_names]
        X_scaled = xgb_scaler.transform(df)
        prob = float(xgb_model.predict_proba(X_scaled)[0][1])
        prediction = int(prob >= threshold)

        if prob < 0.3:
            risk_level = "Low Risk"
        elif prob < 0.6:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return {
            "prediction": prediction,
            "probability": round(prob, 4),
            "risk_level": risk_level,
            "diagnosis": "Alzheimer's Positive" if prediction == 1 else "No Alzheimer's Detected",
            "threshold_used": threshold,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Dataset Record Lookup Endpoint ───────────────────────────────────────────
@app.get("/patient-record/{record_number}")
def get_patient_record(record_number: int, _user: dict = Depends(require_auth)):
    """Look up a clinical record from the reference dataset by its PatientID,
    or (if no PatientID matches) by its 1-indexed row position in the CSV."""
    match = _PATIENT_DF[_PATIENT_DF["PatientID"] == record_number]
    if not match.empty:
        row = match.iloc[0]
    elif 1 <= record_number <= len(_PATIENT_DF):
        row = _PATIENT_DF.iloc[record_number - 1]
    else:
        min_id, max_id = int(_PATIENT_DF["PatientID"].min()), int(_PATIENT_DF["PatientID"].max())
        raise HTTPException(
            status_code=404,
            detail=f"No record found for {record_number}. Use a PatientID ({min_id}-{max_id}) "
                   f"or a row number (1-{len(_PATIENT_DF)}).",
        )

    data = {}
    for field in PatientInput.model_fields:
        value = row[field]
        data[field] = value.item() if hasattr(value, "item") else value

    return {
        "patientId": int(row["PatientID"]),
        "diagnosis": int(row["Diagnosis"]),
        "data": data,
    }


# ─── MRI Image Classification Endpoint ───────────────────────────────────────
MAX_MRI_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@app.post("/predict-mri")
async def predict_mri(file: UploadFile = File(...), _user: dict = Depends(require_auth)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        if len(contents) > MAX_MRI_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image too large (max 10MB).")
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Send a valid JPEG/PNG.")

    try:
        cnn = _get_cnn()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"MRI model unavailable: {e}")

    tensor = cnn["transform"](img).unsqueeze(0).to(cnn["device"])

    with torch.no_grad():
        logits = cnn["model"](tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    predicted_class = cnn["classes"][pred_idx]
    confidence = float(probs[pred_idx])

    class_probabilities = {
        cls: round(float(probs[i]), 4)
        for i, cls in enumerate(cnn["classes"])
    }

    if predicted_class == "NonDemented":
        severity = "None"
    elif predicted_class == "VeryMildDemented":
        severity = "Very Mild"
    elif predicted_class == "MildDemented":
        severity = "Mild"
    else:
        severity = "Moderate"

    return {
        "predicted_class": predicted_class,
        "severity": severity,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probabilities,
        "model": "ResNet50 CNN",
    }


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    # Doesn't force-load the CNN model when lazy-loading is enabled — a health
    # check shouldn't trigger a 94MB download + model build on its own.
    cnn = _get_cnn(force=not _LAZY_CNN_LOAD)
    cnn_info = (
        {
            "type": "ResNet50 CNN",
            "classes": cnn["classes"],
            "img_size": cnn["img_size"],
            "device": str(cnn["device"]),
            "val_accuracy": cnn["val_acc"],
        }
        if cnn
        else {"type": "ResNet50 CNN", "status": "not loaded yet (lazy mode)"}
    )
    return {
        "status": "ok",
        "models": {
            "xgboost": {
                "type": "XGBoost (Tuned)",
                "features": len(feature_names),
                "threshold": threshold,
            },
            "cnn": cnn_info,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("RELOAD", "false").lower() == "true",
    )
