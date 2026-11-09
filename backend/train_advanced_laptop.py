import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.data.dataset import NIHChestXrayDataModule
from backend.models.multimodal_lit import MultimodalLitModel


def train_laptop_advanced():
    print("=" * 50)
    print("Starting Advanced Laptop-Optimized Training")
    print("=" * 50)

    # 1. Dataset Loading
    # Small batch size to fit in 8GB VRAM
    data_module = NIHChestXrayDataModule(
        data_root="./backend/data/nih_cxr14",
        batch_size=4,
        num_workers=0,  # Keep 0 for Windows compatibility and stability
    )

    # 2. Initialize the Model
    # Partial Fine-tuning: We unfreeze the top layers of the vision model
    model = MultimodalLitModel(
        lr=2e-4,  # Slightly lower learning rate for fine-tuning
        weight_decay=1e-4,
        freeze_vision=False,
        freeze_text=True,
    )

    # Manually freeze all vision layers except the last 2 Transformer blocks
    for name, param in model.model.vision_model.visual.named_parameters():
        if "transformer.resblocks.10" not in name and "transformer.resblocks.11" not in name:
            param.requires_grad = False

    print("\n[INFO] Partial Fine-Tuning Enabled: Last 2 layers of Vision Encoder are trainable.")

    # 3. Setup Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="./backend/checkpoints",
        filename="laptop-advanced-best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 4. Initialize Trainer
    # accumulate_grad_batches=8 means effective batch size is 4*8 = 32
    # precision="16-mixed" saves massive VRAM and speeds up RTX GPUs
    try:
        trainer = pl.Trainer(
            max_epochs=15,
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
            log_every_n_steps=10,
            accumulate_grad_batches=8,
            enable_progress_bar=True,
        )

        # 5. Train!
        print("\n[INFO] Resuming high-performance training from Epoch 1...")
        trainer.fit(
            model,
            datamodule=data_module,
            ckpt_path="backend/checkpoints/laptop-advanced-best-epoch=00-val_loss=0.02.ckpt",
        )

        print("\n" + "=" * 50)
        print("Training Complete! The best weights have been saved in backend/checkpoints/")
        print("=" * 50)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\n" + "!" * 50)
            print("OOM ERROR DETECTED: The GPU ran out of memory.")
            print("Please reduce the batch size in this script to 2 or 1.")
            print("!" * 50)
            sys.exit(1)
        else:
            raise e


if __name__ == "__main__":
    train_laptop_advanced()
