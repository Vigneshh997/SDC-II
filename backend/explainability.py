"""Class-specific attribution with explicit method names and limitations."""

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.config import DISEASE_LABELS


def layercam_map(spatial, gradient):
    heatmap = (gradient.clamp_min(0) * spatial).sum(1).clamp_min(0)
    heatmap = heatmap - heatmap.amin((-2, -1), keepdim=True)
    return heatmap / heatmap.amax((-2, -1), keepdim=True).clamp_min(1e-8)


def overlay_original(original, heatmap):
    """Invert official Resize(224)+CenterCrop(224) geometry; leave cropped margins unchanged."""
    original = original.convert("RGB")
    width, height = original.size
    scale = 224 / min(width, height)
    rw, rh = int(width * scale), int(height * scale)
    resized = original.resize((rw, rh), Image.Resampling.BICUBIC)
    left, top = round((rw - 224) / 2), round((rh - 224) / 2)
    heat = (
        F.interpolate(heatmap[:, None].float(), (224, 224), mode="bilinear", align_corners=False)[0, 0]
        .detach()
        .cpu()
        .numpy()
    )
    crop = np.asarray(resized.crop((left, top, left + 224, top + 224))).astype(float)
    # Red overlay with opacity proportional to saliency. Zero maps preserve the image.
    alpha = (0.6 * heat)[..., None]
    color = np.zeros_like(crop)
    color[..., 0] = 255
    blended = np.uint8(crop * (1 - alpha) + color * alpha)
    resized.paste(Image.fromarray(blended), (left, top))
    result = resized.resize((width, height), Image.Resampling.BICUBIC)
    output = io.BytesIO()
    result.save(output, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def text_attribution(model, image_tokens, meta_tokens, note, target):
    encoded = model.tokenizer([note], padding=True, truncation=True, max_length=256, return_tensors="pt")
    device = image_tokens.device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    embedding = model.text_encoder.get_input_embeddings()(encoded["input_ids"]).detach().requires_grad_(True)
    hidden = model.text_encoder(
        inputs_embeds=embedding, attention_mask=encoded["attention_mask"]
    ).last_hidden_state
    text = model.text_proj(hidden)
    fused, _ = model.fusion(image_tokens.detach(), text, meta_tokens.detach(), encoded["attention_mask"])
    logit = model.multi_disease_head(fused)[0, target]
    (gradients,) = torch.autograd.grad(logit, embedding)
    scores = (gradients * embedding).sum(-1)[0].detach().cpu().tolist()
    ids = encoded["input_ids"][0].tolist()
    tokens = model.tokenizer.convert_ids_to_tokens(ids)
    special = set(model.tokenizer.all_special_ids)
    return {
        "method": "gradient_times_input_embedding",
        "target": DISEASE_LABELS[target],
        "unit": "local logit attribution",
        "tokens": [
            {"token": t, "attribution": float(s)} for t, s, i in zip(tokens, scores, ids) if i not in special
        ],
        "limitation": "Local first-order sensitivity, not a causal explanation. Only the first 256 tokens are modeled.",
    }


def metadata_shap(model, image_tokens, text_tokens, mask, metadata, background, target):
    import shap

    fields = model.metadata_transform.fields
    categories = ["female", "male", "other", "unknown"]

    def raw_vector(row):
        model.metadata_transform.encode(row)
        return [categories.index(row[f]) if f == "sex" else float(row[f]) for f in fields]

    if not background:
        raise ValueError("Observed training background unavailable; SHAP cannot be computed")
    bg = np.array([raw_vector(row) for row in background], dtype=float)

    def predict(values):
        predictions = []
        with torch.no_grad():
            for value in values:
                row = {
                    f: categories[int(round(v))] if f == "sex" else float(v) for f, v in zip(fields, value)
                }
                encoded = model.metadata_transform.encode(row).unsqueeze(0).to(image_tokens.device)
                meta = model.meta_encoder(model.metadata_transform(encoded)).unsqueeze(1)
                fused, _ = model.fusion(image_tokens, text_tokens, meta, mask)
                predictions.append(model.multi_disease_head(fused).sigmoid()[0, target].item())
        return np.array(predictions)

    explainer = shap.KernelExplainer(predict, bg)
    values = explainer.shap_values(
        np.array([raw_vector(metadata)]), nsamples=2 ** len(fields), l1_reg=0, silent=True
    )
    return {
        "method": "KernelSHAP",
        "target": DISEASE_LABELS[target],
        "unit": "probability difference",
        "background_source": "Observed training patients saved in checkpoint",
        "expected_value": float(np.asarray(explainer.expected_value).item()),
        "values": {f: float(v) for f, v in zip(fields, np.asarray(values).reshape(-1))},
        "limitation": "Conditional on the supplied image and note. Correlated features and background selection affect attribution.",
    }


def explain(model, tensor, original, note, metadata, background, target):
    device = next(model.parameters()).device
    encoded_meta = model.metadata_transform.encode(metadata).unsqueeze(0).to(device)
    with torch.enable_grad():
        output = model(tensor, [note], encoded_meta, explain=True)
        (gradient,) = torch.autograd.grad(output.logits[0, target], output.spatial)
        heatmap = layercam_map(output.spatial, gradient)
    evidence = {
        "target": DISEASE_LABELS[target],
        "status": {},
        "heatmap": overlay_original(original, heatmap),
        "visual_method": "LayerCAM on shallow CNN output",
        "visual_limitation": "Class-specific model sensitivity. Central crop only; cropped margins have no saliency evidence.",
    }
    with torch.no_grad():
        image_tokens, _ = model.encode_image(tensor)
        text_tokens, mask = model.encode_text([note])
        meta_tokens = model.meta_encoder(model.metadata_transform(encoded_meta)).unsqueeze(1)
    for name, operation in (
        ("text", lambda: text_attribution(model, image_tokens, meta_tokens, note, target)),
        (
            "metadata",
            lambda: metadata_shap(model, image_tokens, text_tokens, mask, metadata, background, target),
        ),
    ):
        try:
            evidence[name] = operation()
            evidence["status"][name] = "available"
        except Exception as exc:
            # Do not emit note contents or raw patient data to logs/errors.
            evidence["status"][name] = f"unavailable: {type(exc).__name__}"
    weights = output.attention["0"][0].detach().mean(0)
    length = text_tokens.shape[1]
    token_ids = model.tokenizer([note], truncation=True, max_length=256)["input_ids"][0]
    tokens = model.tokenizer.convert_ids_to_tokens(token_ids)
    links = []
    for flat in torch.topk(
        weights[:, :length].flatten(), min(16, weights[:, :length].numel())
    ).indices.tolist():
        patch, token = divmod(flat, length)
        side = output.spatial.shape[-1]
        links.append(
            {
                "patch_row": patch // side,
                "patch_column": patch % side,
                "token": tokens[token],
                "weight": float(weights[patch, token]),
            }
        )
    evidence["cross_modal"] = {
        "method": "last-layer image-query cross-attention averaged over heads",
        "links": links,
        "limitation": "Attention distribution, not causal or class-specific attribution.",
    }
    return evidence
