"""Explicit held-out evaluation, never called automatically during model selection."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from backend.checkpoint import load_model, read_checkpoint
from backend.data.dataset import load_manifest, paired_loader
from backend.data.validation import MetadataTransform
from backend.pipelines.metrics import compute_metrics
from backend.pipelines.train import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", default="outputs/xfusion_v2/test_metrics.json")
    args = parser.parse_args()
    checkpoint = read_checkpoint(args.checkpoint)
    digest = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    if digest != checkpoint["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from recorded dataset/splits")
    metadata = MetadataTransform(checkpoint["model_config"].get("metadata_fields", ["age", "sex"]))
    rows = load_manifest(args.manifest, metadata)
    model, checkpoint = load_model(args.checkpoint, args.device)
    loader = paired_loader([r for r in rows if r["split"] == "test"], model.vision_preprocess, metadata, 4)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    targets, probabilities, loss = collect(model, loader, args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = compute_metrics(targets, probabilities, checkpoint["thresholds"])
    result.update(
        {
            "test_asl": loss,
            "manifest_sha256": digest,
            "pipeline_seconds": elapsed,
            "pipeline_seconds_per_sample": elapsed / len(targets),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
            "timing_scope": "Data loading, preprocessing and prediction; no explanations; no warmup.",
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
