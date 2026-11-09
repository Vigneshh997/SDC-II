"""Single orchestrator for strict inputs, predictions and evidence."""

import threading
import time

import torch

from backend.config import DISEASE_LABELS
from backend.data.validation import validate_note
from backend.explainability import explain
from backend.reporting import build_report


class InferenceService:
    def __init__(self, model, checkpoint):
        self.model, self.checkpoint = model, checkpoint
        self.lock = threading.Lock()

    def predict(self, image, note, metadata, examination_id):
        note = validate_note(note)
        encoded = self.model.metadata_transform.encode(metadata)
        if not examination_id or not examination_id.strip() or len(examination_id) > 128:
            raise ValueError("A de-identified examination_id is required (maximum 128 characters)")
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("Inference is busy. Retry after the active examination finishes.")
        try:
            start = time.perf_counter()
            device = next(self.model.parameters()).device
            tensor = self.model.vision_preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = self.model(tensor, [note], encoded.unsqueeze(0).to(device))
                probabilities = output.logits.sigmoid()[0].cpu().tolist()
            if not all(0 <= p <= 1 for p in probabilities):
                raise RuntimeError("Model returned invalid probabilities")
            thresholds = self.checkpoint["thresholds"]
            predictions = dict(zip(DISEASE_LABELS, probabilities))
            selected = [label for label, p, t in zip(DISEASE_LABELS, probabilities, thresholds) if p >= t]
            target = max(range(len(probabilities)), key=probabilities.__getitem__)
            try:
                explanations = explain(
                    self.model,
                    tensor,
                    image,
                    note,
                    metadata,
                    self.checkpoint.get("metadata_background"),
                    target,
                )
            except Exception as exc:
                explanations = {
                    "status": {"visual": f"unavailable: {type(exc).__name__}"},
                    "limitation": "Prediction succeeded, but explanations failed. No substitute evidence was generated.",
                }
            result = {
                "predictions": predictions,
                "thresholds": dict(zip(DISEASE_LABELS, thresholds)),
                "selected_conditions": selected,
                "has_findings": bool(selected),
                "explanations": explanations,
                "heatmap": explanations.get("heatmap"),
                "metadata_attributions": explanations.get("metadata", {}).get("values", {}),
                "model": {
                    "architecture_version": self.checkpoint["architecture_version"],
                    "manifest_sha256": self.checkpoint["manifest_sha256"],
                    "training_steps": self.checkpoint["training_steps"],
                },
                "elapsed_seconds": time.perf_counter() - start,
            }
            result["report"] = build_report(
                {
                    "examination_id": examination_id,
                    "clinical_note": note,
                    "metadata": metadata,
                    "image_size": list(image.size),
                },
                result,
            )
            return result
        finally:
            self.lock.release()
