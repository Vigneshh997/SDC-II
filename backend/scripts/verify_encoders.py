"""Authentic pretrained encoder software check. Artificial inputs, no clinical claims.

Runs one disposable optimization step, verifies all modality contributions and
explanation execution, then discards the model. Never saves a serving checkpoint.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from backend.config import BIOMEDCLIP_REVISION, CXR_BERT_REVISION
from backend.explainability import explain
from backend.models.losses import AsymmetricLoss, contrastive_loss
from backend.models.multimodal import MultimodalModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("audit/encoder_verification.json"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this Python environment")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    start = time.perf_counter()
    model = MultimodalModel(local_files_only=True, clahe=True).to(args.device).eval()
    load_time = time.perf_counter() - start
    background = [{"age": 20, "sex": "female"}, {"age": 60, "sex": "male"}]
    model.metadata_transform.fit(background)
    image = Image.new("RGB", (320, 256), (128, 128, 128))
    images = torch.stack([model.vision_preprocess(image), model.vision_preprocess(image)]).to(args.device)
    notes = ["Artificial software verification input one.", "Artificial software verification input two."]
    metadata = torch.stack([model.metadata_transform.encode(row) for row in background]).to(args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = model(images, notes, metadata, explain=True)
    if args.device == "cuda":
        torch.cuda.synchronize()
    forward_time = time.perf_counter() - start
    loss = AsymmetricLoss()(output.logits, torch.zeros(2, 14, device=args.device))
    loss += 0.1 * contrastive_loss(output.image, output.text, output.metadata)
    loss.backward()
    gradients = {
        name: float(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None))
        for name, module in [
            ("cnn", model.cnn_refinement),
            ("text_projection", model.text_proj),
            ("metadata_mlp", model.meta_encoder),
            ("cross_attention", model.fusion),
        ]
    }
    assert all(value > 0 for value in gradients.values()), gradients
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        base = model(images, notes, metadata).logits
        variants = [
            model(images + 0.1, notes, metadata).logits,
            model(images, ["Changed artificial note.", "Other artificial note."], metadata).logits,
            model(images, notes, metadata + torch.tensor([10, 0, 0, 0, 0], device=args.device)).logits,
        ]
        differences = {
            name: float((variant - base).abs().max().item())
            for name, variant in zip(["image", "text", "metadata"], variants)
        }
    assert all(value > 1e-7 for value in differences.values()), differences
    start = time.perf_counter()
    evidence = explain(model, images[:1], image, notes[0], background[0], background, 0)
    explanation_time = time.perf_counter() - start
    assert evidence["status"] == {"text": "available", "metadata": "available"}, evidence["status"]
    result = {
        "scope": "Artificial software fixtures only. Disposable untrained fusion. No genuine training or clinical evaluation.",
        "torch": torch.__version__,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "encoder_revisions": {"biomedclip": BIOMEDCLIP_REVISION, "cxr_bert": CXR_BERT_REVISION},
        "logit_shape": list(output.logits.shape),
        "cnn_shape": list(output.spatial.shape),
        "gradient_absolute_sums": gradients,
        "modality_logit_change": differences,
        "explanation_status": evidence["status"],
        "heatmap_generated": evidence["heatmap"].startswith("data:image/jpeg;base64,"),
        "cross_modal_links": len(evidence["cross_modal"]["links"]),
        "load_seconds": load_time,
        "batch2_forward_seconds": forward_time,
        "single_explanation_seconds": explanation_time,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
