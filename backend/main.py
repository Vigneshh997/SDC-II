"""Research-only API. Readiness remains visible when weights are unavailable."""

import io
import json
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.checkpoint import load_model
from backend.config import DISCLAIMER, FEATURES
from backend.data.validation import MetadataTransform, validate_note
from backend.inference import InferenceService

MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 20_000_000


@asynccontextmanager
async def lifespan(app):
    app.state.service = None
    app.state.readiness_reason = (
        "No compatible trained checkpoint configured; genuine paired training is required."
    )
    checkpoint_path = os.getenv("XFUSION_CHECKPOINT")
    if checkpoint_path:
        try:
            requested = os.getenv("XFUSION_DEVICE", "auto")
            if requested not in {"auto", "cpu", "cuda"}:
                raise ValueError("XFUSION_DEVICE must be auto, cpu, or cuda")
            if requested == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA requested but unavailable in this Python environment")
            device = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
            model, checkpoint = load_model(checkpoint_path, device, os.getenv("HF_HUB_OFFLINE") == "1")
            app.state.service = InferenceService(model, checkpoint)
            app.state.readiness_reason = None
        except Exception as exc:
            app.state.readiness_reason = f"Model initialization failed ({type(exc).__name__}). Check checkpoint version, provenance and runtime dependencies."
    yield
    app.state.service = None


app = FastAPI(
    title="X-Fusion tri-modal research prototype", description=DISCLAIMER, version="2.0.0", lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        v.strip()
        for v in os.getenv("XFUSION_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if v.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    service = getattr(app.state, "service", None)
    return {
        "status": "ok" if service else "not_ready",
        "device": str(next(service.model.parameters()).device) if service else None,
        "reason": getattr(app.state, "readiness_reason", "Model not initialized"),
        "disclaimer": DISCLAIMER,
    }


@app.get("/schema")
def schema():
    service = getattr(app.state, "service", None)
    fields = service.model.metadata_transform.fields if service else ("age", "sex")
    return {
        "metadata_fields": {f: FEATURES[f] for f in fields},
        "required": ["image", "clinical_note", "metadata", "examination_id", "paired_inputs_confirmed"],
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict")
def predict(
    image: UploadFile = File(...),
    clinical_note: str = Form(...),
    metadata: str = Form(...),
    examination_id: str = Form(...),
    paired_inputs_confirmed: bool = Form(...),
):
    try:
        note = validate_note(clinical_note)
        values = json.loads(metadata)
        service = getattr(app.state, "service", None)
        validator = service.model.metadata_transform if service else MetadataTransform()
        validator.encode(values)
        if not paired_inputs_confirmed:
            raise ValueError("Confirm all inputs belong to the same examination and were observed")
        if not examination_id.strip() or len(examination_id) > 128:
            raise ValueError("A de-identified examination_id is required (maximum 128 characters)")
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, detail="Only JPEG, PNG and WebP images are supported")
    data = image.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, detail="Image exceeds 15 MB")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if min(probe.size) < 32 or max(probe.size) > 8192:
                raise HTTPException(422, detail="Image dimensions must each be between 32 and 8192 pixels")
            if probe.width * probe.height > MAX_PIXELS:
                raise HTTPException(413, detail="Image exceeds 20 million pixels")
            if probe.format not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(415, detail="Unsupported image encoding")
            expected = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[probe.format]
            if image.content_type != expected:
                raise HTTPException(415, detail="Image content does not match its declared MIME type")
            probe.verify()
        with Image.open(io.BytesIO(data)) as decoded:
            decoded.load()
            original = decoded.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, detail="Image is empty, corrupt, or unsafe to decode") from exc
    if service is None:
        raise HTTPException(503, detail=getattr(app.state, "readiness_reason", "No compatible trained model"))
    try:
        return service.predict(original, note, values, examination_id)
    except BlockingIOError as exc:
        raise HTTPException(429, detail=str(exc), headers={"Retry-After": "5"}) from exc
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise HTTPException(503, detail="GPU memory exhausted; reduce workload before retrying") from exc
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, detail="Inference failed; no prediction was substituted") from exc
