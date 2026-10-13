# backend/models/multimodal_lit.py
"""PyTorch Lightning wrapper for the multimodal pneumonia model.

Handles:
- Training / validation / test step logic with BCEWithLogitsLoss.
- Metric logging (loss, accuracy, AUROC).
- Optimizer + cosine-annealing LR scheduler.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC

from .multimodal import MultimodalModel

class AsymmetricLoss(torch.nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        # x: logits, y: targets (multi-hot)
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos * (1 - xs_pos) ** self.gamma_pos + los_neg * (1 - xs_neg) ** self.gamma_neg
        return -loss.sum() / x.size(0)


class MultimodalLitModel(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        meta_input_dim: int = 6,
        freeze_vision: bool = True,
        freeze_text: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = MultimodalModel(
            meta_input_dim=meta_input_dim,
            freeze_vision=freeze_vision,
            freeze_text=freeze_text,
        )
        self.criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        self.temperature = 0.07  # InfoNCE temperature

        # Metrics for 14 classes
        self.train_acc = MultilabelAccuracy(num_labels=14)
        self.val_acc = MultilabelAccuracy(num_labels=14)
        self.val_auroc = MultilabelAUROC(num_labels=14)
        self.test_acc = MultilabelAccuracy(num_labels=14)
        self.test_auroc = MultilabelAUROC(num_labels=14)

    def forward(self, image, clinical_note, metadata):
        return self.model(image, clinical_note, metadata)

    # ---------- steps ----------

    def training_step(self, batch, batch_idx):
        images, notes, meta, labels = batch
        labels_f = labels.float()
        
        logits, vision_feat, text_feat = self(images, notes, meta)
        
        # 1. Asymmetric Loss for Multi-Disease Classification
        asl_loss = self.criterion(logits, labels_f)
        
        # 2. InfoNCE Contrastive Loss (Vision <-> Text)
        # Flatten sequence dim: (B, 1, D) -> (B, D)
        v = F.normalize(vision_feat[:, 0, :], p=2, dim=-1)
        t = F.normalize(text_feat[:, 0, :], p=2, dim=-1)
        
        # Cosine similarity matrix scaled by temperature
        sim = torch.matmul(v, t.T) / self.temperature
        
        # Contrastive targets (diagonal should be 1)
        b_size = v.size(0)
        contrast_labels = torch.arange(b_size, device=self.device)
        
        loss_v2t = F.cross_entropy(sim, contrast_labels)
        loss_t2v = F.cross_entropy(sim.T, contrast_labels)
        info_nce_loss = (loss_v2t + loss_t2v) / 2.0
        
        # Total Loss
        loss = asl_loss + 0.1 * info_nce_loss

        preds = (torch.sigmoid(logits) > 0.5).long()
        self.train_acc(preds, labels.long())

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_asl", asl_loss, on_step=True)
        self.log("train_nce", info_nce_loss, on_step=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, notes, meta, labels = batch
        labels_f = labels.float()
        
        logits, _, _ = self(images, notes, meta)
        loss = self.criterion(logits, labels_f)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        labels_long = labels.long()
        
        self.val_acc(preds, labels_long)
        self.val_auroc(probs, labels_long)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("val_auroc", self.val_auroc, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, notes, meta, labels = batch
        logits, _, _ = self(images, notes, meta)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        labels_long = labels.long()
        
        self.test_acc(preds, labels_long)
        self.test_auroc(probs, labels_long)

        self.log("test_acc", self.test_acc, on_epoch=True)
        self.log("test_auroc", self.test_auroc, on_epoch=True)

    # ---------- optimizer ----------

    def configure_optimizers(self):
        # Only train unfrozen params (fusion + heads + projections)
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs if self.trainer else 10
        )
        return [optimizer], [scheduler]
