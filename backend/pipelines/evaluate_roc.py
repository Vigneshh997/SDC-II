"""Generate exploratory per-label ROC curves for the local NIH test subset.

The current dataset creates report text and several vital signs from labels.
Curves from this script are therefore software diagnostics, not valid clinical
performance estimates.
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve

from backend.data.dataset import DISEASE_LABELS, NIHChestXrayDataModule
from backend.models.multimodal import MultimodalModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "backend"
    / "checkpoints"
    / "laptop-advanced-best-epoch=14-val_loss=0.00.ckpt"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "backend" / "data" / "nih_cxr14",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--figure",
        type=Path,
        default=PROJECT_ROOT / "paper_assets" / "roc_exploratory.png",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "xfusion" / "test_predictions_exploratory.csv",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = NIHChestXrayDataModule(
        data_root=str(args.data_root),
        batch_size=args.batch_size,
        num_workers=0,
    )
    data.setup(stage="test")
    test_loader = data.test_dataloader()

    model = MultimodalModel(meta_input_dim=6)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_state = checkpoint.get("state_dict", checkpoint)
    model_state = {
        key.removeprefix("model."): value
        for key, value in checkpoint_state.items()
        if key.startswith("model.")
    }
    if not model_state:
        model_state = checkpoint_state
    incompatible = model.load_state_dict(model_state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            f"Checkpoint is missing model parameters: {incompatible.missing_keys[:8]}"
        )
    model.to(device).eval()

    all_targets = []
    all_scores = []
    with torch.inference_mode():
        for batch_index, (images, notes, metadata, labels) in enumerate(test_loader):
            images = images.to(device)
            metadata = metadata.to(device)
            logits, _, _ = model(images, notes, metadata)
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(labels.numpy())
            if batch_index % 25 == 0:
                print(f"Processed {batch_index * args.batch_size} test images")

    targets = np.concatenate(all_targets, axis=0).astype(np.uint8)
    scores = np.concatenate(all_scores, axis=0)
    image_names = data.test_dataset.df["Image Index"].tolist()
    if len(image_names) != len(targets):
        raise RuntimeError("Test image identifiers do not match prediction count")

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    prediction_table = {"Image Index": image_names}
    for label_index, label in enumerate(DISEASE_LABELS):
        prediction_table[f"label_{label}"] = targets[:, label_index]
        prediction_table[f"prob_{label}"] = scores[:, label_index]
    import pandas as pd

    pd.DataFrame(prediction_table).to_csv(args.predictions, index=False)

    figure, axes = plt.subplots(4, 4, figsize=(13, 9), constrained_layout=True)
    axes_flat = axes.ravel()
    valid_curves = []
    common_fpr = np.linspace(0, 1, 201)

    for label_index, label in enumerate(DISEASE_LABELS):
        axis = axes_flat[label_index]
        positives = int(targets[:, label_index].sum())
        negatives = len(targets) - positives
        if positives == 0 or negatives == 0:
            axis.text(0.5, 0.5, "ROC unavailable", ha="center", va="center")
            axis.set_title(label.replace("_", " "), fontsize=8)
            continue

        fpr, tpr, _ = roc_curve(targets[:, label_index], scores[:, label_index])
        label_auc = auc(fpr, tpr)
        valid_curves.append(np.interp(common_fpr, fpr, tpr))
        axis.plot(fpr, tpr, color="#176B87", linewidth=1.5)
        axis.plot([0, 1], [0, 1], color="#9A9A9A", linestyle="--", linewidth=0.8)
        axis.set_title(f"{label.replace('_', ' ')} | AUC {label_auc:.3f} | n+ {positives}", fontsize=7)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)

    micro_fpr, micro_tpr, _ = roc_curve(targets.ravel(), scores.ravel())
    micro_auc = auc(micro_fpr, micro_tpr)
    summary_axis = axes_flat[14]
    summary_axis.plot(micro_fpr, micro_tpr, label=f"Micro AUC = {micro_auc:.3f}", color="#176B87")
    if valid_curves:
        macro_tpr = np.mean(valid_curves, axis=0)
        macro_auc = auc(common_fpr, macro_tpr)
        summary_axis.plot(common_fpr, macro_tpr, label=f"Macro AUC = {macro_auc:.3f}", color="#D05A30")
    summary_axis.plot([0, 1], [0, 1], color="#9A9A9A", linestyle="--", linewidth=0.8)
    summary_axis.set_title("Aggregate curves", fontsize=8)
    summary_axis.set_xlim(0, 1)
    summary_axis.set_ylim(0, 1)
    summary_axis.grid(alpha=0.2)
    summary_axis.legend(fontsize=7, loc="lower right")

    note_axis = axes_flat[15]
    note_axis.axis("off")
    note_axis.text(
        0.02,
        0.95,
        "EXPLORATORY ONLY\n\n"
        f"Test images: {len(targets):,}\n"
        f"Seed: {args.seed}\n\n"
        "Training notes and several vitals are generated from disease labels. "
        "These curves are leakage-contaminated diagnostics, not clinical performance.",
        ha="left",
        va="top",
        fontsize=8,
        wrap=True,
    )
    figure.suptitle("X-Fusion ROC Curves on the Available NIH Test Subset", fontsize=12)
    figure.savefig(args.figure, dpi=180)
    plt.close(figure)

    print(f"Saved ROC figure: {args.figure}")
    print(f"Saved predictions: {args.predictions}")
    print(f"Exploratory micro AUC: {micro_auc:.4f}")
    print(f"Test images: {len(targets)}; device: {device}; seed: {args.seed}")


if __name__ == "__main__":
    main()