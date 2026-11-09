"""Patient-disjoint paired manifest loader. NIH lacks genuine notes/vitals.

Provenance declarations require human review; code cannot establish truth.
One examination per patient is required to avoid false InfoNCE negatives.
"""

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from backend.config import DISEASE_LABELS
from backend.data.validation import validate_note

NUM_CLASSES = len(DISEASE_LABELS)
PNEUMONIA_IDX = DISEASE_LABELS.index("Pneumonia")


class DatasetReadinessError(ValueError):
    pass


def load_manifest(path, metadata_transform):
    path = Path(path)
    if not path.is_file():
        raise DatasetReadinessError(
            "Genuine paired JSONL manifest missing. NIH labels cannot substitute for clinical notes."
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise DatasetReadinessError("Empty paired manifest")
    patients, studies, images, hashes = {}, set(), set(), set()
    for row in rows:
        for key in (
            "patient_id",
            "study_id",
            "image_path",
            "clinical_note",
            "metadata",
            "labels",
            "split",
            "provenance",
        ):
            if key not in row or row[key] is None:
                raise DatasetReadinessError(f"Missing required manifest field: {key}")
        pid, sid, split = str(row["patient_id"]), str(row["study_id"]), row["split"]
        if not pid.strip() or not sid.strip() or split not in {"train", "val", "test"}:
            raise DatasetReadinessError("Invalid patient, study, or split")
        if pid in patients:
            raise DatasetReadinessError("Patient repetition/leakage: select one examination per patient")
        patients[pid] = split
        if sid in studies:
            raise DatasetReadinessError("Duplicate study")
        studies.add(sid)
        prov = row["provenance"]
        if not isinstance(prov, dict) or prov.get("observed") is not True or not prov.get("license"):
            raise DatasetReadinessError("Observed data and permitted-use provenance required")
        if (
            prov.get("note_available_before_prediction") is not True
            or prov.get("labels_derived_from_note") is not False
        ):
            raise DatasetReadinessError(
                "Notes must precede prediction and must not be the source of target labels"
            )
        for modality in ("image", "note", "metadata"):
            link = prov.get(modality, {})
            if (
                str(link.get("patient_id", "")) != pid
                or str(link.get("study_id", "")) != sid
                or not link.get("source")
            ):
                raise DatasetReadinessError(f"Unverified {modality} examination linkage")
        row["clinical_note"] = validate_note(row["clinical_note"])
        metadata_transform.encode(row["metadata"])
        labels = row["labels"]
        if not isinstance(labels, dict) or set(labels) != set(DISEASE_LABELS):
            raise DatasetReadinessError(
                "Labels must define all 14 diseases; No Finding is the all-zero vector"
            )
        if any(type(v) not in (int, float) or v not in (0, 1) for v in labels.values()):
            raise DatasetReadinessError(
                "Unknown or uncertain labels require explicit adjudication before import"
            )
        image = (path.parent / row["image_path"]).resolve()
        if image in images:
            raise DatasetReadinessError("Duplicate image path")
        images.add(image)
        with Image.open(image) as source:
            source.verify()
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in hashes:
            raise DatasetReadinessError("Duplicate image content, possibly across splits")
        hashes.add(digest)
        row["image_path"] = str(image)
    if set(patients.values()) != {"train", "val", "test"}:
        raise DatasetReadinessError("Explicit nonempty train, val and test splits required")
    return rows


class PairedDataset(Dataset):
    def __init__(self, rows, transform, metadata_transform):
        self.rows, self.transform, self.metadata_transform = rows, transform, metadata_transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            image = self.transform(image.convert("RGB"))
        return (
            image,
            row["clinical_note"],
            self.metadata_transform.encode(row["metadata"]),
            torch.tensor([row["labels"][label] for label in DISEASE_LABELS], dtype=torch.float32),
        )


def _collate_fn(batch):
    images, notes, metadata, labels = zip(*batch)
    return torch.stack(images), list(notes), torch.stack(metadata), torch.stack(labels)


def paired_loader(rows, transform, metadata_transform, batch_size, train=False, num_workers=0):
    return DataLoader(
        PairedDataset(rows, transform, metadata_transform),
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )


class NIHChestXrayDataset(Dataset):
    def __init__(self, *args, **kwargs):
        raise DatasetReadinessError(
            "NIH has no observed notes/vitals. Label-generated modalities are prohibited. Use a reviewed paired manifest."
        )


class NIHChestXrayDataModule:
    def __init__(self, *args, **kwargs):
        raise DatasetReadinessError(
            "Legacy NIH tri-modal workflow disabled: target-label leakage. Use backend.pipelines.train with a genuine paired manifest."
        )
