# backend/pipelines/train.py
"""Training script for the X-Fusion multimodal pneumonia detection model.

Usage:
    python -m backend.pipelines.train --config config.yaml

Reads hyper-parameters from config.yaml, initialises the DataModule and
LightningModule, then launches a Trainer with mixed-precision (fp16) to
reduce GPU memory on an RTX 4060.
"""

import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from backend.models.multimodal_lit import MultimodalLitModel
from backend.data.dataset import NIHChestXrayDataModule


def main():
    parser = argparse.ArgumentParser(description="Train X-Fusion multimodal model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    pl.seed_everything(cfg.get("seed", 42))

    # Data module
    dm = NIHChestXrayDataModule(
        data_root=cfg.get("data_root", "./backend/data/nih_cxr14"),
        batch_size=cfg.get("batch_size", 16),
        num_workers=cfg.get("num_workers", 4),
        image_size=cfg.get("image_size", 224),
    )

    # Lightning model
    lit_model = MultimodalLitModel(
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-5),
        meta_input_dim=cfg.get("meta_input_dim", 6),
        freeze_vision=cfg.get("freeze_vision", True),
        freeze_text=cfg.get("freeze_text", True),
    )

    # Logger
    logger = TensorBoardLogger(
        save_dir=cfg.get("output_dir", "./outputs"),
        name="xfusion",
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.get("checkpoint_dir", "./checkpoints"),
        filename="best-{epoch:02d}-{val_auroc:.4f}",
        save_top_k=1,
        monitor="val_auroc",
        mode="max",
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=cfg.get("patience", 5),
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    configured_precision = cfg.get("precision", "16-mixed")
    precision = configured_precision if accelerator == "gpu" else "32-true"
    trainer = pl.Trainer(
        max_epochs=cfg.get("max_epochs", 12),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=precision,
        accumulate_grad_batches=cfg.get("accumulate_grad_batches", 2),
        default_root_dir=cfg.get("output_dir", "./outputs"),
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    print("=" * 60)
    print("  X-Fusion Multimodal Training")
    print(f"  Epochs     : {cfg.get('max_epochs', 12)}")
    print(f"  Batch size : {cfg.get('batch_size', 16)}")
    print(f"  LR         : {cfg.get('lr', 1e-4)}")
    print(f"  Precision  : {precision}")
    print(f"  Device     : {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # Train
    trainer.fit(lit_model, dm)

    # Test on best checkpoint
    print("\n--- Running test evaluation on best checkpoint ---")
    trainer.test(lit_model, dm, ckpt_path="best")

    print(f"\nBest checkpoint saved at: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
