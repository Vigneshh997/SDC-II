"""Reproducible paired-data training. Fails preflight before encoder loading."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import yaml

from backend.checkpoint import read_checkpoint, save_checkpoint
from backend.config import ARCHITECTURE_VERSION, DISEASE_LABELS
from backend.data.dataset import DatasetReadinessError, load_manifest, paired_loader
from backend.data.validation import MetadataTransform
from backend.models.losses import AsymmetricLoss, contrastive_loss
from backend.models.multimodal import MultimodalModel
from backend.pipelines.metrics import select_thresholds


def preflight(config):
    metadata = MetadataTransform(config.get("model", {}).get("metadata_fields", ["age", "sex"]))
    rows = load_manifest(config["manifest"], metadata)
    review = Path(config.get("provenance_review", ""))
    if not review.is_file() or not review.read_text(encoding="utf-8").strip():
        raise DatasetReadinessError("A human provenance review file is required before training")
    if config.get("batch_size", 4) < 2:
        raise DatasetReadinessError("InfoNCE needs batch_size >= 2")
    train = [row for row in rows if row["split"] == "train"]
    if len(train) < config.get("batch_size", 4):
        raise DatasetReadinessError("Not enough distinct training patients for a contrastive batch")
    return rows, metadata


def collect(model, loader, device):
    model.eval()
    targets, probabilities = [], []
    total_loss, count = 0.0, 0
    criterion = AsymmetricLoss()
    with torch.no_grad():
        for images, notes, meta, labels in loader:
            logits = model(images.to(device), notes, meta.to(device)).logits
            total_loss += criterion(logits, labels.to(device)).item() * len(labels)
            count += len(labels)
            targets.append(labels)
            probabilities.append(logits.sigmoid().cpu())
    return torch.cat(targets).numpy(), torch.cat(probabilities).numpy(), total_loss / count


def train(config, resume=None):
    # Protect user artifacts before loading data or expensive pretrained encoders.
    output = Path(config.get("output_dir", "outputs/xfusion_v2"))
    existing = [output / name for name in ("best.pt", "last.pt", "history.json") if (output / name).exists()]
    if existing and (resume is None or Path(resume).resolve().parent != output.resolve()):
        raise FileExistsError(
            "Training output already contains run artifacts. Choose a new output_dir or explicitly "
            "resume a checkpoint from this directory; existing checkpoints were not changed."
        )
    rows, metadata = preflight(config)
    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = config.get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; choose cpu or auto explicitly")
    model_config = config.get("model", {})
    model = MultimodalModel(**model_config, local_files_only=config.get("local_files_only", False))
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    model.metadata_transform.fit([row["metadata"] for row in train_rows])
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    epochs = config.get("max_epochs", 12)
    if epochs < 1 or config.get("contrastive_weight", 0.1) <= 0:
        raise ValueError("Positive epochs and contrastive_weight required for final X-Fusion")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    amp = device == "cuda" and config.get("mixed_precision", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    digest = hashlib.sha256(Path(config["manifest"]).read_bytes()).hexdigest()
    review_digest = hashlib.sha256(Path(config["provenance_review"]).read_bytes()).hexdigest()
    start, steps, best, stale, history = 0, 0, float("inf"), 0, []
    if resume:
        restored = read_checkpoint(resume)
        if (
            restored["manifest_sha256"] != digest
            or restored["config"] != config
            or restored["provenance_review_sha256"] != review_digest
        ):
            raise ValueError("Resume configuration/dataset/review mismatch")
        model.load_state_dict(restored["model_state"], strict=True)
        optimizer.load_state_dict(restored["optimizer_state"])
        scheduler.load_state_dict(restored["scheduler_state"])
        scaler.load_state_dict(restored["scaler_state"])
        torch.set_rng_state(restored["rng_state"])
        if device == "cuda" and restored.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(restored["cuda_rng_state"])
        start, steps, best, stale, history = (
            restored[k] for k in ("epoch", "training_steps", "best_val_loss", "stale_epochs", "history")
        )
    output.mkdir(parents=True, exist_ok=True)
    val_loader = paired_loader(val_rows, model.vision_preprocess, metadata, config.get("batch_size", 4))
    criterion = AsymmetricLoss()
    for epoch in range(start, epochs):
        # Epoch-specific seed makes loader order and augmentation reproducible on resume.
        random.seed(seed + epoch)
        torch.manual_seed(seed + epoch)
        loader = paired_loader(
            train_rows,
            model.train_preprocess,
            metadata,
            config.get("batch_size", 4),
            train=True,
            num_workers=config.get("num_workers", 0),
        )
        model.train()
        asl_total, nce_total, batches = 0.0, 0.0, 0
        for images, notes, meta, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=amp):
                prediction = model(images.to(device), notes, meta.to(device))
                asl = criterion(prediction.logits, labels.to(device))
                nce = contrastive_loss(
                    prediction.image, prediction.text, prediction.metadata, config.get("temperature", 0.07)
                )
                loss = asl + config.get("contrastive_weight", 0.1) * nce
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training objective; checkpoint was not updated")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.get("gradient_clip", 1.0), error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            batches += 1
            asl_total += asl.item()
            nce_total += nce.item()
        y, p, val_loss = collect(model, val_loader, device)
        scheduler.step()
        improved = val_loss < best
        best, stale = (val_loss, 0) if improved else (best, stale + 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_asl": asl_total / batches,
                "train_infonce": nce_total / batches,
                "val_asl": val_loss,
            }
        )
        payload = {
            "architecture_version": ARCHITECTURE_VERSION,
            "labels": DISEASE_LABELS,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "config": config,
            "manifest_sha256": digest,
            "provenance_review_sha256": review_digest,
            "provenance_reviewed": True,
            "training_steps": steps,
            "epoch": epoch + 1,
            "best_val_loss": best,
            "stale_epochs": stale,
            "history": history,
            "thresholds": select_thresholds(y, p),
            "threshold_source": "validation_only",
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device == "cuda" else None,
            "metadata_background": [row["metadata"] for row in train_rows[:32]],
        }
        if improved:
            save_checkpoint(output / "best.pt", payload)
        save_checkpoint(output / "last.pt", payload)
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(history[-1]), flush=True)
        if stale >= config.get("patience", 5):
            break
    return output / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.preflight_only:
        preflight(config)
        print("Structural preflight passed. Human provenance review remains authoritative.")
    else:
        print(train(config, args.resume))


if __name__ == "__main__":
    main()
