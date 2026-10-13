# backend/pipelines/eval.py
"""Evaluation script for the X-Fusion multimodal pneumonia model.

Usage:
    python -m backend.pipelines.eval --config config.yaml

Loads the best checkpoint and runs full test-set evaluation,
reporting accuracy, AUROC, confusion matrix, and per-class metrics.
"""

import argparse
import yaml
import torch
import pytorch_lightning as pl
from pathlib import Path

from backend.models.multimodal_lit import MultimodalLitModel
from backend.data.dataset import NIHChestXrayDataModule


def main():
    parser = argparse.ArgumentParser(description="Evaluate X-Fusion model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path from config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Data
    dm = NIHChestXrayDataModule(
        data_root=cfg.get("data_root", "./backend/data/nih_cxr14"),
        batch_size=cfg.get("batch_size", 16),
        num_workers=cfg.get("num_workers", 4),
    )

    # Load checkpoint
    ckpt = args.checkpoint or cfg.get("checkpoint_path", "./checkpoints/last.ckpt")
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}")
        print("  Run training first: python -m backend.pipelines.train --config config.yaml")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    lit_model = MultimodalLitModel.load_from_checkpoint(str(ckpt_path))

    # Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    configured_precision = cfg.get("precision", "16-mixed")
    precision = configured_precision if accelerator == "gpu" else "32-true"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision,
    )

    print("\n--- Test Evaluation ---")
    results = trainer.test(lit_model, dm)
    print("\nResults:")
    for k, v in results[0].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
