import os
import math
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
from torch import nn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# -----------------------------------------------------------------------------
# 1. Device Setup & File Paths
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Running on device: {device}")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# Locate data.csv (checks parent directory first, then current)
DATA_PATH = os.path.join(PARENT_DIR, "data.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(CURRENT_DIR, "data.csv")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"data.csv not found in {PARENT_DIR} or {CURRENT_DIR}")

# Locate model_weights.pth (checks parent directory first, then current)
MODEL_PATH = os.path.join(PARENT_DIR, "model_weights.pth")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(CURRENT_DIR, "model_weights.pth")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"model_weights.pth not found in {PARENT_DIR} or {CURRENT_DIR}")

# Read data.csv and extract exact feature names in original order
df_data = pd.read_csv(DATA_PATH)
if "Bankrupt?" not in df_data.columns:
    raise ValueError("Target column 'Bankrupt?' not found in data.csv")

# Feature names exactly as ordered in data.csv
FEATURE_NAMES = df_data.drop("Bankrupt?", axis=1).columns.tolist()
NUM_FEATURES = len(FEATURE_NAMES)
print(f"[*] Loaded {NUM_FEATURES} features from: {DATA_PATH}")

# -----------------------------------------------------------------------------
# 2. Model Architecture
# -----------------------------------------------------------------------------
class MultiCell(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        k1 = self.layer1(x)
        k2 = torch.relu(k1)
        return k2

model = nn.Sequential(
    MultiCell(95, 30),
    MultiCell(30, 20),
    nn.Linear(20, 1)
)

# -----------------------------------------------------------------------------
# 3. Model Weight Loading (Loaded once at startup)
# -----------------------------------------------------------------------------
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print(f"[*] Successfully loaded model weights from: {MODEL_PATH}")

# -----------------------------------------------------------------------------
# 4. FastAPI App & Templates
# -----------------------------------------------------------------------------
app = FastAPI(title="AI Bankruptcy Risk Prediction")

STATIC_DIR = os.path.join(CURRENT_DIR, "static")
TEMPLATES_DIR = os.path.join(CURRENT_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Precalculate sample records for quick demo/testing
sample_bankrupt = df_data[df_data["Bankrupt?"] == 1].iloc[0][FEATURE_NAMES].to_dict()
sample_non_bankrupt = df_data[df_data["Bankrupt?"] == 0].iloc[0][FEATURE_NAMES].to_dict()
sample_mean = df_data[FEATURE_NAMES].mean().to_dict()

# -----------------------------------------------------------------------------
# 5. Routes
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Renders the main dashboard page with all 95 features."""
    feature_items = [
        {"id": f"feat_{idx}", "name": name, "display_name": name.strip(), "index": idx + 1}
        for idx, name in enumerate(FEATURE_NAMES)
    ]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "features": feature_items,
            "num_features": NUM_FEATURES,
            "device_name": str(device).upper()
        }
    )

@app.get("/api/features")
async def get_features():
    """Returns the ordered list of 95 feature names."""
    return {"count": len(FEATURE_NAMES), "features": FEATURE_NAMES}

@app.get("/api/sample")
async def get_sample(type: str = "non_bankrupt"):
    """
    Returns sample data to easily populate the form for testing:
    - 'bankrupt': A real bankrupt company sample from data.csv
    - 'non_bankrupt': A real healthy company sample from data.csv
    - 'mean': Dataset mean values
    """
    if type == "bankrupt":
        return {"type": "bankrupt", "features": sample_bankrupt}
    elif type == "mean":
        return {"type": "mean", "features": sample_mean}
    else:
        return {"type": "non_bankrupt", "features": sample_non_bankrupt}

@app.post("/predict")
async def predict(request: Request):
    """
    Receives JSON payload with all 95 features, validates inputs,
    builds a float32 tensor of shape [1, 95] on device, evaluates model,
    and returns prediction and probability.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload provided.")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object.")

    # Support either {"features": {...}} or direct dictionary {...}
    features_dict = payload.get("features", payload) if "features" in payload and isinstance(payload["features"], dict) else payload

    # Check for missing features
    missing_features = []
    ordered_values = []

    for name in FEATURE_NAMES:
        if name not in features_dict:
            missing_features.append(name.strip())
            continue

        val = features_dict[name]

        # Check for empty string or None
        if val is None or val == "":
            missing_features.append(f"{name.strip()} (empty)")
            continue

        try:
            val_float = float(val)
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{name.strip()}' has invalid non-numeric value: {val}"
            )

        # Check for NaN or Inf
        if math.isnan(val_float) or math.isinf(val_float):
            raise HTTPException(
                status_code=400,
                detail=f"Feature '{name.strip()}' contains invalid NaN or Infinite value."
            )

        ordered_values.append(val_float)

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing or empty required features ({len(missing_features)}): {', '.join(missing_features[:5])}..."
            if len(missing_features) > 5
            else f"Missing or empty required features: {', '.join(missing_features)}"
        )

    # Convert to PyTorch float32 tensor with shape [1, 95] on device
    tensor_input = torch.tensor([ordered_values], dtype=torch.float32, device=device)

    # Perform inference with torch.no_grad()
    with torch.no_grad():
        logit = model(tensor_input)
        probability_tensor = torch.sigmoid(logit)
        probability = float(probability_tensor.item())

    # Karar:
    # probability >= 0.5 -> Bankrupt = 1
    # probability < 0.5  -> Bankrupt = 0
    prediction = 1 if probability >= 0.5 else 0
    percentage = round(probability * 100, 2)

    return JSONResponse(
        content={
            "prediction": prediction,
            "probability": round(probability, 4),
            "percentage": percentage
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
