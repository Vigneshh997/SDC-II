# backend/main.py
"""FastAPI server for X-Fusion inference.

Usage:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict  – Upload an X-ray image + clinical note + metadata JSON
                     → returns pneumonia probability, type, and heatmap.
    GET  /health   – Health check.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.models.multimodal import MultimodalModel

app = FastAPI(
    title="X-Fusion: Multimodal Pneumonia Detection",
    description="Chest X-ray analysis using BiomedCLIP + CXR-BERT fusion",
    version="1.0.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("XFUSION_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded once at startup)
_model: Optional[MultimodalModel] = None
_model_device: Optional[torch.device] = None
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "backend" / "checkpoints"


@app.on_event("startup")
async def load_model():
    """Load the trained model at server startup."""
    global _model, _model_device
    requested_device = os.getenv("XFUSION_DEVICE", "auto").lower()
    if requested_device == "cpu":
        device = torch.device("cpu")
    elif requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _model = MultimodalModel(meta_input_dim=6)
    _model.to(device)

    # Load the specified checkpoint
    preferred_ckpt = CHECKPOINT_DIR / "laptop-advanced-best-epoch=14-val_loss=0.00.ckpt"

    if preferred_ckpt.exists():
        ckpt_path = preferred_ckpt
    else:
        # Fallback: pick the latest *best* checkpoint available
        best_ckpts = list(CHECKPOINT_DIR.glob("*best*.ckpt"))
        best_ckpts.sort(key=os.path.getmtime)
        ckpt_path = best_ckpts[-1] if best_ckpts else None

    if not ckpt_path:
        raise RuntimeError(
            f"No model checkpoint found in {CHECKPOINT_DIR}. "
            "Refusing to serve predictions with random weights."
        )

    print(f"Loading trained weights from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not model_state:
        model_state = state_dict
    incompatible = _model.load_state_dict(model_state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} is incompatible; missing keys: "
            f"{incompatible.missing_keys[:5]}"
        )
    print(f"Loaded checkpoint from {ckpt_path}")

    _model.eval()
    _model_device = device
    print(f"Model loaded on {device}")


@app.get("/health")
async def health():
    loaded = _model is not None and _model_device is not None
    return {
        "status": "ok" if loaded else "not_ready",
        "device": str(_model_device) if loaded else "not loaded",
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    clinical_note: str = Form("No clinical notes provided."),
    metadata: str = Form("{}"),
):
    """Run inference on a single chest X-ray.

    - image: JPEG/PNG chest X-ray file.
    - clinical_note: Free-text clinical note (optional).
    - metadata: JSON string with keys: age, sex, temperature, heart_rate, resp_rate, spo2.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if not image.filename or image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP images are supported.")

    # Save uploaded image to temp file
    suffix = Path(image.filename).suffix or ".jpg"
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The uploaded image must be smaller than 15 MB.")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        try:
            meta_dict = json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="metadata must be valid JSON") from exc
        if not isinstance(meta_dict, dict):
            raise HTTPException(status_code=400, detail="metadata must be a JSON object")
        
        # Get predictions
        results = _model.infer(tmp_path, clinical_note, meta_dict)
        
        # Get XAI Heatmap (ViT-CAM)
        heatmap_base64 = _model.get_xai_heatmap(tmp_path, clinical_note, meta_dict)
        results["heatmap"] = heatmap_base64

        # Get SHAP feature attributions for metadata
        try:
            shap_values = _model.get_shap_values(tmp_path, clinical_note, meta_dict)
            results["metadata_attributions"] = shap_values
        except Exception as e:
            print(f"SHAP Error: {e}")
            results["metadata_attributions"] = {}
            
        return results
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
