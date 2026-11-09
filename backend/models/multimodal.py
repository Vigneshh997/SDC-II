"""Strict tri-modal X-Fusion. No fallback encoders or inferred patient inputs.

The official BiomedCLIP spatial patch tokens feed two 3x3 convolutions.
CXR-BERT's actual BERT backbone supplies token features (its unused MLM and
sentence projection heads are discarded). New shared projections are trained.
"""

import json
import math
from dataclasses import dataclass

import torch
from torch import nn

from backend.config import BIOMEDCLIP_ID, BIOMEDCLIP_REVISION, CXR_BERT_ID, CXR_BERT_REVISION
from backend.data.validation import MetadataTransform, validate_note
from backend.data.preprocessing import ImagePreprocessing


class MetadataEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, values):
        return self.net(values)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dimension, heads):
        super().__init__()
        self.attention = nn.ModuleList(
            [nn.MultiheadAttention(dimension, heads, batch_first=True) for _ in range(3)]
        )
        self.norm1 = nn.ModuleList([nn.LayerNorm(dimension) for _ in range(3)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(dimension) for _ in range(3)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dimension, dimension * 4), nn.GELU(), nn.Linear(dimension * 4, dimension)
                )
                for _ in range(3)
            ]
        )

    def forward(self, tokens, masks, explain=False):
        updated, weights = [], {}
        for index in range(3):
            others = [j for j in range(3) if j != index]
            # Q is exclusively the receiving modality. K/V exclusively the other two.
            context = torch.cat([tokens[j] for j in others], dim=1)
            context_mask = torch.cat([masks[j] for j in others], dim=1)
            attended, weight = self.attention[index](
                tokens[index],
                context,
                context,
                key_padding_mask=context_mask,
                need_weights=explain,
                average_attn_weights=False,
            )
            value = self.norm1[index](tokens[index] + attended)
            updated.append(self.norm2[index](value + self.ffn[index](value)))
            if explain:
                weights[str(index)] = weight
        return updated, weights


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2):
        super().__init__()
        self.modality_ids = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.layers = nn.ModuleList([CrossAttentionBlock(d_model, nhead) for _ in range(num_layers)])
        if num_layers < 1:
            raise ValueError("At least one cross-attention layer is required")

    def forward(self, vision, text, metadata, text_mask=None, explain=False):
        tokens = [vision, text, metadata]
        if any(
            t.ndim != 3 or t.shape[0] != vision.shape[0] or t.shape[-1] != vision.shape[-1] for t in tokens
        ):
            raise ValueError("Three compatible token sequences required")
        masks = [torch.zeros(t.shape[:2], dtype=torch.bool, device=t.device) for t in tokens]
        if text_mask is not None:
            masks[1] = ~text_mask.bool()
        if any(m.all(1).any() for m in masks):
            raise ValueError("Each modality must contain an unmasked token")
        tokens = [t + self.modality_ids[i] for i, t in enumerate(tokens)]
        weights = {}
        for layer in self.layers:
            tokens, weights = layer(tokens, masks, explain)
        pooled = [masked_mean(t, ~m) for t, m in zip(tokens, masks)]
        return torch.stack(pooled).mean(0), weights


def masked_mean(tokens, mask):
    return (tokens * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def load_pretrained_encoders(local_files_only=False):
    """Pinned authentic weights; reject missing files instead of substituting."""
    import open_clip
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    from safetensors.torch import load_file
    from transformers import BertConfig, BertModel, BertTokenizerFast

    config_file = hf_hub_download(
        BIOMEDCLIP_ID,
        "open_clip_config.json",
        revision=BIOMEDCLIP_REVISION,
        local_files_only=local_files_only,
    )
    hf_hub_download(
        BIOMEDCLIP_ID,
        "open_clip_pytorch_model.bin",
        revision=BIOMEDCLIP_REVISION,
        local_files_only=local_files_only,
    )
    clip, train_preprocess, preprocess = open_clip.create_model_and_transforms(
        f"local-dir:{Path(config_file).parent}"
    )
    # Only the visual tower is retained. Its pretrained projection is not used by the patch adapter.
    visual = clip.visual
    del clip
    text_config_path = hf_hub_download(
        CXR_BERT_ID, "config.json", revision=CXR_BERT_REVISION, local_files_only=local_files_only
    )
    text_weights = hf_hub_download(
        CXR_BERT_ID, "model.safetensors", revision=CXR_BERT_REVISION, local_files_only=local_files_only
    )
    config = BertConfig.from_dict(json.loads(Path(text_config_path).read_text()))
    text = BertModel(config, add_pooling_layer=False)
    state = load_file(text_weights)
    backbone = {k.removeprefix("bert."): v for k, v in state.items() if k.startswith("bert.")}
    # Older Transformers persisted this deterministic buffer; current BERT does not.
    positions = backbone.pop("embeddings.position_ids", None)
    if positions is not None and not torch.equal(
        positions, torch.arange(config.max_position_embeddings).unsqueeze(0)
    ):
        raise ValueError("Unexpected CXR-BERT position buffer")
    text.load_state_dict(backbone, strict=True)
    tokenizer = BertTokenizerFast.from_pretrained(
        CXR_BERT_ID, revision=CXR_BERT_REVISION, local_files_only=local_files_only
    )
    return visual, text, tokenizer, train_preprocess, preprocess


@dataclass
class FusionOutput:
    logits: torch.Tensor
    image: torch.Tensor
    text: torch.Tensor
    metadata: torch.Tensor
    spatial: torch.Tensor
    attention: dict


class MultimodalModel(nn.Module):
    def __init__(
        self,
        metadata_fields=("age", "sex"),
        d_model=512,
        nhead=8,
        num_layers=2,
        freeze_vision=True,
        freeze_text=True,
        local_files_only=False,
        encoders=None,
        clahe=False,
    ):
        super().__init__()
        # Encoder injection is for isolated software tests only, never a runtime fallback.
        if encoders is None:
            encoders = load_pretrained_encoders(local_files_only)
        (
            self.vision_model,
            self.text_encoder,
            self.tokenizer,
            self.train_preprocess,
            self.vision_preprocess,
        ) = encoders
        self.train_preprocess = ImagePreprocessing(self.train_preprocess, clahe)
        self.vision_preprocess = ImagePreprocessing(self.vision_preprocess, clahe)
        self.metadata_transform = MetadataTransform(metadata_fields)
        self.freeze_vision, self.freeze_text = freeze_vision, freeze_text
        for module, frozen in ((self.vision_model, freeze_vision), (self.text_encoder, freeze_text)):
            if frozen:
                module.requires_grad_(False)
                module.eval()
        channels = self.vision_model.trunk.num_features
        self.cnn_refinement = nn.Sequential(
            nn.Conv2d(channels, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GELU(),
        )
        self.image_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, d_model)
        self.meta_encoder = MetadataEncoder(len(self.metadata_transform.names), output_dim=d_model)
        self.fusion = CrossAttentionFusion(d_model, nhead, num_layers)
        self.multi_disease_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_model, 14)
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_vision:
            self.vision_model.eval()
        if self.freeze_text:
            self.text_encoder.eval()
        return self

    def encode_image(self, images):
        features = self.vision_model.trunk.forward_features(images)
        prefix = self.vision_model.trunk.num_prefix_tokens
        patches = features[:, prefix:, :]
        side = math.isqrt(patches.shape[1])
        if side * side != patches.shape[1]:
            raise ValueError("BiomedCLIP patch grid must be square")
        spatial = patches.transpose(1, 2).reshape(images.shape[0], -1, side, side)
        spatial = self.cnn_refinement(spatial)
        return self.image_proj(spatial.flatten(2).transpose(1, 2)), spatial

    def encode_text(self, texts):
        notes = [validate_note(t) for t in texts]
        encoded = self.tokenizer(notes, padding=True, truncation=True, max_length=256, return_tensors="pt")
        device = next(self.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        result = self.text_encoder(**encoded).last_hidden_state
        return self.text_proj(result), encoded["attention_mask"]

    def forward(self, image, clinical_text, metadata, explain=False):
        if (
            image is None
            or metadata is None
            or not clinical_text
            or image.ndim != 4
            or image.shape[1] != 3
            or metadata.ndim != 2
            or image.shape[0] != len(clinical_text)
            or image.shape[0] != metadata.shape[0]
        ):
            raise ValueError("Image, clinical notes and metadata from every examination are mandatory")
        if not torch.isfinite(image).all():
            raise ValueError("Image contains nonfinite values")
        vision, spatial = self.encode_image(image)
        text, mask = self.encode_text(clinical_text)
        meta = self.meta_encoder(self.metadata_transform(metadata)).unsqueeze(1)
        fused, attention = self.fusion(vision, text, meta, mask, explain)
        return FusionOutput(
            self.multi_disease_head(fused),
            vision.mean(1),
            masked_mean(text, mask),
            meta[:, 0],
            spatial,
            attention,
        )
