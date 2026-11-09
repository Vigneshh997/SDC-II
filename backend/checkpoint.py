"""Safe, explicit v2 checkpoint contract. Legacy leakage checkpoints are rejected."""

from pathlib import Path

import torch

from backend.config import ARCHITECTURE_VERSION, DISEASE_LABELS


def read_checkpoint(path):
    if not path or not Path(path).is_file():
        raise ValueError(
            "No compatible trained checkpoint configured. Supply XFUSION_CHECKPOINT after valid paired training."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture_version") != ARCHITECTURE_VERSION:
        raise ValueError(
            "Legacy/incompatible checkpoint rejected. Retraining on genuine paired modalities is required."
        )
    if checkpoint.get("labels") != DISEASE_LABELS:
        raise ValueError("Checkpoint label ordering is incompatible")
    if checkpoint.get("training_steps", 0) < 1 or checkpoint.get("provenance_reviewed") is not True:
        raise ValueError("Checkpoint lacks reviewed paired training provenance")
    if not checkpoint.get("manifest_sha256") or not checkpoint.get("provenance_review_sha256"):
        raise ValueError("Checkpoint missing dataset/review fingerprints")
    thresholds = torch.as_tensor(checkpoint.get("thresholds", []))
    if (
        thresholds.shape != (14,)
        or not torch.isfinite(thresholds).all()
        or not ((thresholds > 0) & (thresholds < 1)).all()
    ):
        raise ValueError("Invalid checkpoint thresholds")
    return checkpoint


def load_model(path, device="cpu", local_files_only=False):
    from backend.models.multimodal import MultimodalModel

    checkpoint = read_checkpoint(path)
    model = MultimodalModel(**checkpoint["model_config"], local_files_only=local_files_only)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if not model.metadata_transform.fitted.item():
        raise ValueError("Checkpoint has no fitted training normalization")
    model.to(device).eval()
    return model, checkpoint


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    torch.save(payload, temporary)
    temporary.replace(path)
