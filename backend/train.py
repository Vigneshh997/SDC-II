import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Add the project root to the python path so imports work
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.data.dataset import NIHChestXrayDataModule
from backend.models.multimodal_lit import MultimodalLitModel


def train_model():
    print("=" * 50)
    print("Starting Laptop-Optimized Tri-Modal Training")
    print("=" * 50)

    # 1. Initialize Dataset Module
    data_module = NIHChestXrayDataModule(
        data_root="./backend/data/nih_cxr14",
        batch_size=8,  # Small batch size for laptop memory
        num_workers=0,  # 0 for windows laptop compatibility
    )

    # 2. Initialize the Model with frozen heavy encoders!
    # By freezing vision and text, we only train the new multi_disease_head and fusion layer.
    model = MultimodalLitModel(
        lr=1e-3,
        weight_decay=1e-4,
        freeze_vision=True,  # CRITICAL: Keeps memory low
        freeze_text=True,  # CRITICAL: Keeps memory low
    )

    # 3. Setup Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="./backend/checkpoints",
        filename="best-multimodal-v2-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=2,
        mode="min",
    )

    # 4. Initialize Trainer
    # We use a max of 3 epochs for a quick laptop training session.
    # The accelerator="auto" will use CPU or GPU automatically.
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    # 5. Train!
    print("\n[INFO] Starting training loop...")
    trainer.fit(model, datamodule=data_module)

    print("\n" + "=" * 50)
    print("Training Complete! The best weights have been saved in backend/checkpoints/")
    print("=" * 50)


if __name__ == "__main__":
    train_model()
