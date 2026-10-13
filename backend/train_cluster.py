import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.data.dataset import NIHChestXrayDataModule
from backend.models.multimodal_lit import MultimodalLitModel

def train_cluster():
    print("="*50)
    print("Starting A100/H100 Optimized Tri-Modal Training")
    print("="*50)
    
    # 1. High-Performance Dataset Loading
    # We increase batch size significantly since we have 80GB of VRAM
    # We also use multiple workers to prevent dataloader bottlenecks
    data_module = NIHChestXrayDataModule(
        data_root="./backend/data/nih_cxr14",
        batch_size=64, # Increase to 128 if OOM doesn't occur on A100
        num_workers=8, # Pre-fetching
    )
    
    # 2. Initialize the Model
    # We unfreeze the top layers of the vision model to allow it to adapt
    # to the specific X-ray domain (fine-tuning) while keeping the base frozen.
    model = MultimodalLitModel(
        lr=5e-4, # Slightly lower LR since batch size is large and we are fine-tuning
        weight_decay=1e-4,
        freeze_vision=False,  # We will manually freeze all but last 2 layers below
        freeze_text=True      # Keep text frozen as we only have short notes
    )
    
    # Manually freeze all vision layers except the last 2 Transformer blocks
    # This is called Partial Fine-Tuning and gives the best accuracy without catastrophic forgetting
    for name, param in model.model.vision_model.visual.named_parameters():
        if "transformer.resblocks.10" not in name and "transformer.resblocks.11" not in name:
            param.requires_grad = False
            
    print("\n[INFO] Partial Fine-Tuning Enabled: Last 2 layers of Vision Encoder are trainable.")

    # 3. Setup Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="./backend/checkpoints",
        filename="cluster-best-v2-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        save_weights_only=True # Saves disk space since checkpoint files get huge
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # 4. Initialize Trainer (A100/H100 OPTIMIZED)
    # bf16-mixed uses the Hopper/Ampere Tensor Cores to speed up matrix multiplication 3x
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=-1, # Use all available GPUs on the node
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto", # Distributed Data Parallel
        precision="bf16-mixed", # MASSIVE SPEEDUP on A100/H100
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=50,
        accumulate_grad_batches=2 # Effectively doubles batch size without increasing VRAM
    )
    
    # Optional: Enable PyTorch 2.0 compiler for even more speed if supported
    # model = torch.compile(model)
    
    # 5. Train!
    print("\n[INFO] Starting high-performance training loop...")
    torch.set_float32_matmul_precision('high') # Leverage Tensor Cores
    trainer.fit(model, datamodule=data_module)
    
    print("\n" + "="*50)
    print("Training Complete! The world-class weights have been saved in backend/checkpoints/")
    print("="*50)

if __name__ == "__main__":
    train_cluster()
