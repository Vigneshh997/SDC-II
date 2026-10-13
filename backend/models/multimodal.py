# backend/models/multimodal.py
"""Multimodal model for X-Fusion pneumonia detection.

Architecture (matching DAViT base paper):
- Vision encoder : BiomedCLIP ViT-B/16 (via open_clip, pre-trained on PMC-15M).
- Text encoder  : CXR-BERT (microsoft/BiomedVLP-CXR-BERT-specialized).
- Metadata encoder: 2-layer MLP for structured patient data.
- Fusion        : Learnable [CLS] + cross-attention transformer over all tokens.
- Heads         : Multi-disease classification (14 classes).
- Explainability: GradCAM on the last vision-transformer block.

All heavy pre-trained weights are frozen by default; only the projection
layers, fusion module, and classification heads are trained, keeping
GPU memory < 6 GB on an RTX 4060 with fp16.
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Dict, Any, Optional
import numpy as np

import open_clip
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class MetadataEncoder(nn.Module):
    """Small MLP that projects structured patient metadata to d_model."""

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, output_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """Transformer encoder that fuses [CLS] + vision + text + meta tokens."""

    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 2):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(
        self,
        vision_feat: torch.Tensor,
        text_feat: torch.Tensor,
        meta_feat: torch.Tensor,
    ) -> torch.Tensor:
        B = vision_feat.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, vision_feat, text_feat, meta_feat], dim=1)
        fused = self.transformer(tokens)
        return fused[:, 0, :]  # CLS output


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MultimodalModel(nn.Module):
    """End-to-end multimodal model for chest-X-ray pneumonia detection."""

    # We use 512 as our internal dim to match BiomedCLIP output (512-d).
    D_MODEL = 512

    def __init__(
        self,
        meta_input_dim: int = 6,
        freeze_vision: bool = True,
        freeze_text: bool = True,
    ):
        super().__init__()

        # ---- Vision encoder (BiomedCLIP ViT-B/16) ----
        self.vision_model, _, self.vision_preprocess = (
            open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
        )
        # We only need the visual half; discard the text tower from open_clip.
        # BiomedCLIP ViT-B/16 outputs 512-d embeddings.
        if freeze_vision:
            for p in self.vision_model.parameters():
                p.requires_grad = False

        # ---- Text encoder (CXR-BERT) ----
        self.text_encoder = AutoModel.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedVLP-CXR-BERT-specialized",
            trust_remote_code=True,
        )
        cxr_bert_dim = self.text_encoder.config.hidden_size  # typically 768
        self.text_proj = nn.Linear(cxr_bert_dim, self.D_MODEL)

        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # ---- Metadata encoder ----
        self.meta_encoder = MetadataEncoder(
            input_dim=meta_input_dim, output_dim=self.D_MODEL
        )

        # ---- Fusion ----
        self.fusion = CrossAttentionFusion(
            d_model=self.D_MODEL, nhead=8, num_layers=2
        )

        # ---- Classification heads ----
        self.multi_disease_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.D_MODEL, 14),
        )

    # ----- helpers -----

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images via BiomedCLIP.

        Args:
            image: (B, 3, 224, 224) normalised tensor.
        Returns:
            (B, 1, D_MODEL) vision features.
        """
        vision_requires_grad = any(
            parameter.requires_grad for parameter in self.vision_model.parameters()
        )
        with torch.set_grad_enabled(vision_requires_grad):
            feat = self.vision_model.encode_image(image)  # (B, 512)
        return feat.unsqueeze(1)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of clinical notes via CXR-BERT.

        Returns:
            (B, 1, D_MODEL) text features.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        device = next(self.text_encoder.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.text_encoder.parameters())
        ):
            out = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            cls_rep = out.last_hidden_state[:, 0, :]
        projected = self.text_proj(cls_rep)
        return projected.unsqueeze(1)

    # ----- forward -----

    def forward(
        self,
        image: torch.Tensor,
        clinical_text: list[str],
        metadata: torch.Tensor,
    ):
        """Forward pass.

        Args:
            image        : (B, 3, 224, 224)
            clinical_text: list of B strings
            metadata     : (B, meta_input_dim)
        Returns:
            logits       : (B, 14) multi-disease logits
            vision_feat  : (B, 1, D) raw vision embeddings (for InfoNCE)
            text_feat    : (B, 1, D) raw text embeddings (for InfoNCE)
        """
        vision_feat = self.encode_image(image)          # (B, 1, D)
        text_feat = self.encode_text(clinical_text)     # (B, 1, D)
        meta_feat = self.meta_encoder(metadata).unsqueeze(1)  # (B, 1, D)

        fused = self.fusion(vision_feat, text_feat, meta_feat)  # (B, D)

        logits = self.multi_disease_head(fused)  # (B, 14)
        return logits, vision_feat, text_feat

    # ----- inference helpers -----

    def _parse_meta_dict(self, meta_dict: Dict[str, Any]) -> list[float]:
        """Parse metadata dictionary with robust clinical physiological defaults."""
        defaults = {
            "age": 50.0,
            "sex": 1.0,
            "temperature": 37.0,
            "heart_rate": 75.0,
            "resp_rate": 16.0,
            "spo2": 98.0,
        }
        meta_vals = []
        for k in ["age", "sex", "temperature", "heart_rate", "resp_rate", "spo2"]:
            raw = meta_dict.get(k, None) if meta_dict else None
            if raw is None or raw == "":
                val = defaults[k]
            elif k == "sex":
                if isinstance(raw, (int, float)) and raw in (0, 1):
                    val = float(raw)
                else:
                    val = 1.0 if str(raw).lower() in ["male", "m", "1"] else 0.0
            else:
                try:
                    val = float(raw)
                    if val <= 0 and k != "sex":
                        val = defaults[k]
                except (ValueError, TypeError):
                    val = defaults[k]
            meta_vals.append(float(val))
        return meta_vals

    def infer(self, image_path: str, clinical_note: str, meta_dict: Dict[str, Any]):
        """Single-sample inference returning probabilities for 14 diseases."""
        self.eval()
        device = next(self.parameters()).device

        from torchvision import transforms
        from PIL import Image

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        meta_vals = self._parse_meta_dict(meta_dict)
        meta_tensor = torch.tensor([meta_vals], dtype=torch.float32, device=device)

        if not clinical_note or clinical_note.strip() == "":
            clinical_note = "No clinical notes provided. Routine chest radiograph."

        with torch.no_grad():
            logits, _, _ = self.forward(
                img_tensor, [clinical_note], meta_tensor
            )
            probs = torch.sigmoid(logits)[0].cpu().tolist()

        diseases = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia"
        ]

        predictions = {disease: round(prob, 4) for disease, prob in zip(diseases, probs)}
        
        # Sort by probability descending
        sorted_preds = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
        top_disease, top_prob = next(iter(sorted_preds.items()))

        # In multi-label clinical screening, pathology is flagged at calibrated clinical thresholds (>0.30)
        has_finding = top_prob >= 0.30

        return {
            "predictions": sorted_preds,
            "diagnosis": top_disease if has_finding else "Normal / No Finding",
            "has_findings": has_finding,
            "primary_finding": top_disease,
            "primary_confidence": top_prob,
        }

    def get_xai_heatmap(self, image_path: str, clinical_note: str, meta_dict: Dict[str, Any]) -> str:
        """Runs inference and generates a ViT-CAM (Attention) heatmap encoded as base64."""
        import cv2
        import numpy as np
        import base64
        from torchvision import transforms
        from PIL import Image

        self.eval()
        device = next(self.parameters()).device

        # 1. Setup a hook to capture the QKV matrix to calculate attention
        attn_matrix = None
        def qkv_hook(module, input, output):
            nonlocal attn_matrix
            # output is (B, N, 3*C)
            B, N, _ = output.shape
            C = output.shape[-1] // 3
            num_heads = 12 # ViT-Base has 12 heads
            
            # Reshape to extract q, k
            qkv = output.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1] # shape: (B, num_heads, N, head_dim)
            
            # Calculate attention matrix manually
            scale = (C // num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            
            attn_matrix = attn.detach().cpu()

        # Hook the qkv layer of the final block's attention module
        handle = self.vision_model.visual.trunk.blocks[-1].attn.qkv.register_forward_hook(qkv_hook)

        # 2. Run forward pass
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        orig_img = cv2.imread(image_path)
        orig_img = cv2.resize(orig_img, (224, 224))
        
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        meta_vals = self._parse_meta_dict(meta_dict)
        meta_tensor = torch.tensor([meta_vals], dtype=torch.float32, device=device)

        if not clinical_note or clinical_note.strip() == "":
            clinical_note = "No clinical notes provided. Routine chest radiograph."

        with torch.no_grad():
            self.forward(img_tensor, [clinical_note], meta_tensor)
        
        handle.remove()

        # 3. Process attention matrix to generate heatmap
        # attn_matrix shape: (1, NumHeads, 197, 197)
        # We want how much the CLS token (index 0) attends to all patch tokens (index 1:)
        cls_attn = attn_matrix[0, :, 0, 1:]  # (NumHeads, 196)
        
        # Average across all attention heads
        saliency = cls_attn.mean(dim=0).numpy()  # (196,)
        
        # Normalize between 0 and 1
        saliency = saliency - saliency.min()
        saliency = saliency / (saliency.max() + 1e-8)

        # Reshape to 14x14 grid (since 224/16 = 14)
        grid_size = int(np.sqrt(saliency.shape[0]))
        heatmap = saliency.reshape(grid_size, grid_size)

        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # Convert to RGB color map (JET)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        # Blend with original image
        alpha = 0.5
        overlay = cv2.addWeighted(heatmap_colored, alpha, orig_img, 1 - alpha, 0)

        # Encode to Base64
        _, buffer = cv2.imencode('.jpg', overlay)
        base64_str = base64.b64encode(buffer).decode('utf-8')

        return f"data:image/jpeg;base64,{base64_str}"

    def get_shap_values(self, image_path: str, clinical_note: str, meta_dict: Dict[str, Any]) -> Dict[str, float]:
        """Calculates SHAP feature attributions for the structured metadata."""
        import shap
        from torchvision import transforms
        from PIL import Image

        self.eval()
        device = next(self.parameters()).device

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        meta_keys = ["age", "sex", "temperature", "heart_rate", "resp_rate", "spo2"]
        meta_vals = self._parse_meta_dict(meta_dict)
        meta_tensor = torch.tensor([meta_vals], dtype=torch.float32, device=device)

        if not clinical_note or clinical_note.strip() == "":
            clinical_note = "No clinical notes provided. Routine chest radiograph."

        with torch.no_grad():
            vision_feat = self.encode_image(img_tensor)
            text_feat = self.encode_text([clinical_note])

        def predict_meta(meta_numpy):
            meta_t = torch.tensor(meta_numpy, dtype=torch.float32, device=device)
            meta_f = self.meta_encoder(meta_t).unsqueeze(1)
            B = len(meta_numpy)
            fused = self.fusion(vision_feat.expand(B, -1, -1),
                                text_feat.expand(B, -1, -1),
                                meta_f)
            logits = self.multi_disease_head(fused)
            return torch.sigmoid(logits).detach().cpu().numpy()

        background = np.zeros((1, 6))
        
        # Suppress shap warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.KernelExplainer(predict_meta, background)
            shap_values = explainer.shap_values(meta_tensor.detach().cpu().numpy())

        # Combine absolute sum of attributions across all 14 disease classes
        # shap_values is a list of 14 arrays, each of shape (1, 6)
        # We aggregate the mean absolute attribution for each feature
        if isinstance(shap_values, list):
            aggregated_shap = np.sum(np.abs(np.array(shap_values)), axis=0)[0]
        else:
            # Depending on shap version, it might return a 3D array (1, 6, 14)
            aggregated_shap = np.sum(np.abs(shap_values), axis=-1)[0]
            
        # Normalize to 0-1 for UI
        if np.max(aggregated_shap) > 0:
            aggregated_shap = aggregated_shap / np.sum(aggregated_shap)

        return {k: round(float(v), 4) for k, v in zip(meta_keys, aggregated_shap)}
