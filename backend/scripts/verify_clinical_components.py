"""Use observed demo clinical components without diagnostic inference/training."""

import argparse
import json
from pathlib import Path

import torch

from backend.data.validation import MetadataTransform, validate_note
from backend.models.multimodal import MetadataEncoder, load_pretrained_encoders
from backend.scripts.prepare_mimic_demo import FIELDS


def verify(root, output, device="cuda"):
    path = root / "clinical_inputs_staging.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(row.get("trimodal_training_ready") is not False for row in rows):
        raise ValueError("Expected explicitly partial observed clinical staging records")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    patients = {
        split: {r["subject_id"] for r in rows if r["component_split"] == split}
        for split in ("train", "val", "test")
    }
    if any(patients[a] & patients[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Clinical component patient leakage")
    normalizer = MetadataTransform(FIELDS).fit(
        [r["metadata"] for r in rows if r["component_split"] == "train"]
    )
    held_out = [r for r in rows if r["component_split"] == "test"][:8]
    if not held_out:
        raise ValueError("No held-out observed clinical records")
    notes = [validate_note(row["clinical_text"]) for row in held_out]
    visual, bert, tokenizer, _, _ = load_pretrained_encoders(local_files_only=True)
    del visual
    bert.requires_grad_(False).to(device).eval()
    encoded = tokenizer(notes, padding=True, truncation=True, max_length=256, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    normalized = normalizer(torch.stack([normalizer.encode(row["metadata"]) for row in held_out])).to(device)
    torch.manual_seed(42)
    mlp = MetadataEncoder(len(normalizer.names)).to(device).eval()
    with torch.no_grad():
        text_features = bert(**encoded).last_hidden_state
        meta_features = mlp(normalized)
    if not torch.isfinite(text_features).all() or not torch.isfinite(meta_features).all():
        raise ValueError("Nonfinite component features")
    report = {
        "source": "Observed MIMIC-IV-ED Demo 2.2 clinical records",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "observed_records_encoded": len(held_out),
        "metadata_fields": list(FIELDS),
        "encoded_metadata_width": len(normalizer.names),
        "cxr_bert_token_feature_shape": list(text_features.shape),
        "metadata_mlp_feature_shape": list(meta_features.shape),
        "features_finite": True,
        "patient_disjoint": True,
        "normalizer_fit_scope": "Training component patients only",
        "mlp_status": "Random initialization; forward validation only; no optimization or checkpoint saved",
        "diagnostic_predictions_produced": False,
        "trimodal_training_performed": False,
        "missing": ["Matching authorized X-rays", "Reviewed study-time linkage", "Compatible disease labels"],
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("backend/data/external/mimic-iv-ed-demo-2.2"))
    parser.add_argument("--output", type=Path, default=Path("audit/clinical_component_verification.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    verify(args.root, args.output, args.device)


if __name__ == "__main__":
    main()
