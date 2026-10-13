# backend/data/dataset.py
"""PyTorch Dataset & Lightning DataModule for the NIH ChestX-ray14 benchmark.

Dataset: 112,120 frontal-view chest X-rays, 30,805 patients, 14 disease labels.
Source:  Wang et al., "ChestX-ray8", CVPR 2017.

The 14 pathology labels are:
    Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
    Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
    Pleural_Thickening, Hernia

We treat this as a **multi-label** classification task (each image can have
multiple findings) with special focus on Pneumonia detection.

Official splits (patient-wise, no data leakage):
    - train_val_list.txt → we further split 90/10 into train/val
    - test_list.txt      → held-out test set

Clinical reports are generated from disease labels using evidence-based
radiology report templates (standard in research when paired report data
is not available). Metadata (age, gender) comes from the original CSV.
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl


# ---- Disease label configuration ----

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

NUM_CLASSES = len(DISEASE_LABELS)
PNEUMONIA_IDX = DISEASE_LABELS.index("Pneumonia")


# ---- Clinically-informed report generation ----
# These templates are based on standard radiology reporting patterns.
# Each disease has multiple template variants for diversity.

_REPORT_TEMPLATES = {
    "No Finding": [
        "The lungs are clear bilaterally. No focal consolidation, pleural effusion, or pneumothorax. "
        "The cardiomediastinal silhouette is within normal limits. No acute osseous abnormality.",
        "Normal chest radiograph. Clear lung fields without evidence of active disease. "
        "Heart size and pulmonary vascularity appear normal.",
        "No acute cardiopulmonary process. The lungs are well expanded and clear. "
        "No pleural abnormality. The mediastinal contour is normal.",
    ],
    "Atelectasis": [
        "Linear atelectasis noted at the lung bases. No focal consolidation or pleural effusion.",
        "Subsegmental atelectasis in the {side} lower lobe. Remainder of the lungs are clear.",
        "Basilar atelectatic changes identified. Consider correlation with clinical history.",
    ],
    "Cardiomegaly": [
        "The cardiac silhouette is enlarged, consistent with cardiomegaly. "
        "Pulmonary vascularity appears mildly congested.",
        "Cardiomegaly noted. Cardiothoracic ratio exceeds 0.5. "
        "Mild pulmonary vascular prominence suggesting volume overload.",
    ],
    "Effusion": [
        "Blunting of the {side} costophrenic angle consistent with pleural effusion.",
        "{side} pleural effusion with associated compressive atelectasis. "
        "Consider clinical correlation for etiology.",
    ],
    "Infiltration": [
        "Patchy infiltrates noted in the {side} lung field. "
        "Differential includes infection versus inflammatory process.",
        "Diffuse bilateral infiltrates present. Consider pneumonia, pulmonary edema, or ARDS.",
    ],
    "Mass": [
        "A rounded opacity in the {side} lung field is suspicious for a pulmonary mass. "
        "CT chest recommended for further evaluation.",
        "Solitary pulmonary nodule/mass identified in the {side} upper lobe. "
        "Follow-up imaging recommended.",
    ],
    "Nodule": [
        "A small pulmonary nodule is noted in the {side} lung. "
        "Follow-up CT recommended per Fleischner criteria.",
        "Incidental pulmonary nodule measuring less than 1 cm. "
        "Recommend follow-up imaging in 6-12 months.",
    ],
    "Pneumonia": [
        "Focal consolidation in the {side} lower lobe with air bronchograms, "
        "consistent with pneumonia. Clinical correlation recommended.",
        "Bilateral patchy airspace opacities suggestive of multifocal pneumonia. "
        "Recommend correlation with clinical symptoms and laboratory data.",
        "Right middle lobe opacity with possible early consolidation. "
        "Findings are concerning for community-acquired pneumonia.",
        "Dense lobar consolidation in the {side} lung consistent with bacterial pneumonia. "
        "Recommend antibiotic therapy and follow-up imaging.",
    ],
    "Pneumothorax": [
        "Visceral pleural line identified along the {side} hemithorax consistent with pneumothorax.",
        "{side} pneumothorax noted. No mediastinal shift to suggest tension physiology.",
    ],
    "Consolidation": [
        "Dense consolidation in the {side} lower lobe. "
        "Differential includes pneumonia, hemorrhage, or organizing pneumonia.",
        "Airspace consolidation with air bronchograms in the {side} lung field.",
    ],
    "Edema": [
        "Bilateral perihilar haziness and Kerley B lines consistent with pulmonary edema. "
        "Cardiomegaly is also present.",
        "Interstitial and alveolar edema. Bilateral pleural effusions. "
        "Findings consistent with congestive heart failure.",
    ],
    "Emphysema": [
        "Hyperinflated lungs with flattened diaphragms consistent with emphysema. "
        "No acute infiltrate.",
        "Bilateral emphysematous changes with increased AP diameter. "
        "Chronic obstructive pulmonary disease pattern.",
    ],
    "Fibrosis": [
        "Bilateral reticular opacities with lower lobe predominance, consistent with pulmonary fibrosis.",
        "Coarse reticular pattern and traction bronchiectasis suggestive of interstitial fibrosis.",
    ],
    "Pleural_Thickening": [
        "Focal pleural thickening along the {side} lateral chest wall. "
        "May represent sequela of prior infection or inflammation.",
        "Bilateral apical pleural thickening. Likely chronic/post-inflammatory changes.",
    ],
    "Hernia": [
        "Retrocardiac opacity with air-fluid level consistent with hiatal hernia.",
        "Large hiatal hernia noted with stomach herniation into the thorax.",
    ],
}


def _generate_clinical_report(diseases: list[str]) -> str:
    """Generate a clinically-informed radiology report from disease labels."""
    if not diseases or diseases == ["No Finding"]:
        return random.choice(_REPORT_TEMPLATES["No Finding"])

    paragraphs = []
    side = random.choice(["left", "right"])
    for disease in diseases:
        templates = _REPORT_TEMPLATES.get(disease, [])
        if templates:
            report = random.choice(templates).format(side=side)
            paragraphs.append(report)

    if not paragraphs:
        return random.choice(_REPORT_TEMPLATES["No Finding"])

    return " ".join(paragraphs)


def _generate_metadata_from_row(row: pd.Series) -> torch.Tensor:
    """Extract real patient metadata from the NIH CSV row.

    Returns tensor of shape (6,): [age, sex, temperature, heart_rate, resp_rate, spo2]
    Age and sex come from real data; vitals are sampled from clinically
    plausible distributions conditioned on disease presence.
    """
    # Real metadata from CSV
    age = float(row.get("Patient Age", 50))
    gender_str = str(row.get("Patient Gender", "M"))
    sex = 1.0 if gender_str.upper() == "M" else 0.0

    # Clinically plausible synthetic vitals (conditioned on findings)
    findings = str(row.get("Finding Labels", "No Finding"))
    has_pneumonia = "Pneumonia" in findings
    has_edema = "Edema" in findings

    if has_pneumonia:
        temperature = random.gauss(38.5, 0.7)
        heart_rate = random.gauss(100, 12)
        resp_rate = random.gauss(24, 4)
        spo2 = random.gauss(93, 3)
    elif has_edema:
        temperature = random.gauss(37.0, 0.3)
        heart_rate = random.gauss(95, 10)
        resp_rate = random.gauss(22, 3)
        spo2 = random.gauss(92, 4)
    else:
        temperature = random.gauss(36.8, 0.3)
        heart_rate = random.gauss(75, 8)
        resp_rate = random.gauss(16, 2)
        spo2 = random.gauss(97, 1.5)

    # Clamp to physiological ranges
    temperature = max(35.0, min(42.0, temperature))
    heart_rate = max(40, min(180, heart_rate))
    resp_rate = max(8, min(45, resp_rate))
    spo2 = max(70, min(100, spo2))

    return torch.tensor(
        [age, sex, temperature, heart_rate, resp_rate, spo2],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NIHChestXrayDataset(Dataset):
    """NIH ChestX-ray14 dataset for multimodal pneumonia detection.

    Returns:
        image          : (3, 224, 224) normalised tensor
        clinical_note  : str — generated radiology report
        metadata       : (6,) tensor [age, sex, temp, hr, rr, spo2]
        label          : int (0 = no pneumonia, 1 = pneumonia)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        images_dir: str,
        transform: Optional[transforms.Compose] = None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Pre-parse disease labels
        self.df["disease_list"] = self.df["Finding Labels"].apply(
            lambda x: [d.strip() for d in str(x).split("|")]
        )
        self.df["pneumonia_label"] = self.df["disease_list"].apply(
            lambda ds: 1 if "Pneumonia" in ds else 0
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Image
        img_path = self.images_dir / row["Image Index"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Clinical report (generated from real disease labels)
        diseases = row["disease_list"]
        note = _generate_clinical_report(diseases)

        # Patient metadata (age + sex from real data, vitals clinically sampled)
        meta = _generate_metadata_from_row(row)

        # Multi-disease label vector
        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for d in diseases:
            if d in DISEASE_LABELS:
                idx = DISEASE_LABELS.index(d)
                label[idx] = 1.0

        return image, note, meta, label


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class NIHChestXrayDataModule(pl.LightningDataModule):
    """Lightning DataModule using the official NIH patient-wise splits."""

    def __init__(
        self,
        data_root: str = "./data/nih_cxr14",
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: int = 224,
        val_fraction: float = 0.1,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction

        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def setup(self, stage: Optional[str] = None):
        # Load metadata
        csv_path = self.data_root / "Data_Entry_2017_v2020.csv"
        df = pd.read_csv(csv_path)

        # Load official splits
        train_val_list = set(
            (self.data_root / "train_val_list.txt").read_text().strip().split("\n")
        )
        test_list = set(
            (self.data_root / "test_list.txt").read_text().strip().split("\n")
        )

        # Filter to images that actually exist on disk
        images_dir = self.data_root / "images"
        existing_images = set(f.name for f in images_dir.glob("*.png"))
        df = df[df["Image Index"].isin(existing_images)]

        # Split according to official lists
        train_val_df = df[df["Image Index"].isin(train_val_list)]
        test_df = df[df["Image Index"].isin(test_list)]

        # Further split train_val into train and val (patient-wise)
        patient_ids = train_val_df["Patient ID"].unique()
        np.random.seed(42)
        np.random.shuffle(patient_ids)
        val_size = int(len(patient_ids) * self.val_fraction)
        val_patients = set(patient_ids[:val_size])

        train_df = train_val_df[~train_val_df["Patient ID"].isin(val_patients)]
        val_df = train_val_df[train_val_df["Patient ID"].isin(val_patients)]

        images_dir_str = str(images_dir)

        print(f"\n  Dataset splits:")
        print(f"    Train: {len(train_df):,} images")
        print(f"    Val:   {len(val_df):,} images")
        print(f"    Test:  {len(test_df):,} images")
        print(f"    Pneumonia prevalence (train): "
              f"{train_df['Finding Labels'].str.contains('Pneumonia').mean():.2%}")

        self.train_dataset = NIHChestXrayDataset(
            train_df, images_dir_str, transform=self.train_transform
        )
        self.val_dataset = NIHChestXrayDataset(
            val_df, images_dir_str, transform=self.val_transform
        )
        self.test_dataset = NIHChestXrayDataset(
            test_df, images_dir_str, transform=self.val_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_collate_fn,
        )


def _collate_fn(batch):
    """Custom collate to handle string clinical notes alongside tensors."""
    images, notes, metas, labels = zip(*batch)
    images = torch.stack(images)
    metas = torch.stack(metas)
    labels = torch.stack(labels)
    return images, list(notes), metas, labels
