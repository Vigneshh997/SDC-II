"""Read-only inventory of local NIH data. Never certifies clinical pairing."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image


def inspect_image(path):
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            size = image.size
            image.verify()
        return path.name, digest, size, None
    except Exception as exc:
        return path.name, None, None, type(exc).__name__


def audit(root):
    df = pd.read_csv(root / "Data_Entry_2017_v2020.csv")
    paths = sorted((root / "images").glob("*"))
    paths = [p for p in paths if p.is_file()]
    present = {p.name for p in paths}
    local = df[df["Image Index"].isin(present)].copy()
    tv = set((root / "train_val_list.txt").read_text().splitlines())
    test = set((root / "test_list.txt").read_text().splitlines())
    tv_patients = set(local[local["Image Index"].isin(tv)]["Patient ID"])
    test_patients = set(local[local["Image Index"].isin(test)]["Patient ID"])
    hashes, sizes, corrupt = Counter(), Counter(), []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, digest, size, error in pool.map(inspect_image, paths):
            if error:
                corrupt.append({"image": name, "error": error})
            else:
                hashes[digest] += 1
                sizes[str(size)] += 1
    return {
        "csv_rows": len(df),
        "csv_columns": list(df.columns),
        "csv_sha256": hashlib.sha256((root / "Data_Entry_2017_v2020.csv").read_bytes()).hexdigest(),
        "image_files": len(paths),
        "matched_rows": len(local),
        "missing_images": int((~df["Image Index"].isin(present)).sum()),
        "unmatched_files": len(present - set(df["Image Index"])),
        "duplicate_csv_image_ids": int(df["Image Index"].duplicated().sum()),
        "duplicate_image_hash_groups": sum(n > 1 for n in hashes.values()),
        "duplicate_image_excess_files": sum(n - 1 for n in hashes.values()),
        "corrupt_images": corrupt,
        "image_dimensions": dict(sizes),
        "missing_csv_fields": {str(k): int(v) for k, v in local.isna().sum().items()},
        "invalid_ages": int((~pd.to_numeric(local["Patient Age"], errors="coerce").between(0, 120)).sum()),
        "label_counts": dict(Counter(d for s in local["Finding Labels"] for d in s.split("|"))),
        "local_patients": int(local["Patient ID"].nunique()),
        "official_trainval_images_local": int(local["Image Index"].isin(tv).sum()),
        "official_test_images_local": int(local["Image Index"].isin(test).sum()),
        "official_image_overlap": len(tv & test),
        "local_official_patient_overlap": len(tv_patients & test_patients),
        "unassigned_local_images": int((~local["Image Index"].isin(tv | test)).sum()),
        "observed_structured_fields": ["Patient Age", "Patient Gender"],
        "genuine_clinical_notes": 0,
        "genuine_vitals": 0,
        "paired_trimodal_readiness": False,
        "label_correctness": "Weak labels from CSV; no independent radiologist adjudication performed.",
        "provenance": "NIH-format files; download receipts and local license acceptance not supplied.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("backend/data/nih_cxr14"))
    parser.add_argument("--output", type=Path, default=Path("audit/dataset_inventory.json"))
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
