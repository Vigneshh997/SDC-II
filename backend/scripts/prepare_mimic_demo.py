"""Acquire open MIMIC-IV-ED demo data and stage observed clinical inputs only.

No X-rays or disease labels are included. The output cannot pass the tri-modal
training manifest contract. Never join these patients to unrelated NIH images.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from backend.data.validation import MetadataTransform, validate_note

BASE_URL = "https://physionet.org/files/mimic-iv-ed-demo/2.2/"
SOURCE_PAGE = "https://physionet.org/content/mimic-iv-ed-demo/2.2/"
FILES = ("LICENSE.txt", "ed/edstays.csv.gz", "ed/triage.csv.gz", "ed/vitalsign.csv.gz")
FIELDS = ("sex", "temperature", "heart_rate", "resp_rate", "spo2")


def fetch_public(relative):
    if relative not in (*FILES, "SHA256SUMS.txt"):
        raise ValueError("Only the pinned openly accessible demo files are allowed")
    with requests.get(BASE_URL + relative, timeout=(15, 60), stream=True) as response:
        response.raise_for_status()
        content = bytearray()
        for block in response.iter_content(65536):
            content.extend(block)
            if len(content) > 2 * 1024 * 1024:
                raise ValueError("Unexpectedly large demo file; download stopped")
    return bytes(content)


def download(root):
    root.mkdir(parents=True, exist_ok=True)
    sums = fetch_public("SHA256SUMS.txt")
    expected = {}
    for line in sums.decode("utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        expected[relative.lstrip("*")] = digest
    receipt = []
    for relative in FILES:
        path = root / relative
        content = path.read_bytes() if path.exists() else fetch_public(relative)
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected.get(relative):
            raise ValueError(f"Publisher checksum mismatch for {relative}; existing files were not replaced")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_bytes(content)
            temporary.replace(path)
        receipt.append(
            {"file": relative, "url": BASE_URL + relative, "sha256": digest, "bytes": len(content)}
        )
    (root / "SHA256SUMS.txt").write_bytes(sums)
    (root / "download_receipt.json").write_text(
        json.dumps(
            {
                "source": SOURCE_PAGE,
                "access": "Open demo only; no credentialed files requested",
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "files": receipt,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def fahrenheit_to_celsius(value):
    if pd.isna(value):
        raise ValueError("Missing observed temperature")
    return (float(value) - 32) * 5 / 9


def prepare_rows(stays, triage):
    keys = ["subject_id", "stay_id"]
    if stays.duplicated(keys).any() or triage.duplicated(keys).any():
        raise ValueError("Duplicate patient/stay keys; explicit adjudication required")
    merged = triage.merge(stays, on=keys, how="left", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError("Triage record has no matching observed ED stay")
    validator = MetadataTransform(FIELDS)
    staged, excluded = [], Counter()
    for row in merged.to_dict("records"):
        try:
            if pd.isna(row.get("chiefcomplaint")):
                raise ValueError("Missing chief complaint")
            note = validate_note(row["chiefcomplaint"])
            if note.replace("_", "").strip() == "":
                raise ValueError("Chief complaint entirely redacted")
            if row.get("gender") not in {"F", "M"}:
                raise ValueError("Unsupported source sex category; not imputed")
            start, end = pd.Timestamp(row["intime"]), pd.Timestamp(row["outtime"])
            if pd.isna(start) or pd.isna(end) or end <= start:
                raise ValueError("Invalid ED stay interval")
            metadata = {
                "sex": {"F": "female", "M": "male"}[row["gender"]],
                "temperature": fahrenheit_to_celsius(row["temperature"]),
                "heart_rate": float(row["heartrate"]),
                "resp_rate": float(row["resprate"]),
                "spo2": float(row["o2sat"]),
            }
            validator.encode(metadata)
        except (ValueError, TypeError) as exc:
            # Do not write clinical text or original measurements to audit logs.
            excluded[type(exc).__name__] += 1
            continue
        pid, stay = str(int(row["subject_id"])), str(int(row["stay_id"]))
        staged.append(
            {
                "subject_id": pid,
                "stay_id": stay,
                "ed_intime": str(start),
                "ed_outtime": str(end),
                "clinical_text": note,
                "metadata": metadata,
                "text_kind": "Observed ED triage chief complaint, not a full clinical note",
                "source": SOURCE_PAGE,
                "license": "ODbL-1.0; original LICENSE.txt retained",
                "pairing_unit": "ED stay; X-ray study matching not yet established",
                "temperature_conversion": "Recorded degrees Fahrenheit to Celsius: (F-32)*5/9",
                "trimodal_training_ready": False,
            }
        )
    return staged, dict(excluded)


def assign_patient_splits(rows, seed=42):
    patients = sorted(
        {r["subject_id"] for r in rows}, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest()
    )
    if len(patients) < 3:
        raise ValueError("At least three observed patients are required for a component split")
    n_train = max(1, int(len(patients) * 0.8))
    n_val = max(1, int(len(patients) * 0.1))
    if n_train + n_val >= len(patients):
        n_train = len(patients) - n_val - 1
    assignments = {
        p: "train" if i < n_train else "val" if i < n_train + n_val else "test"
        for i, p in enumerate(patients)
    }
    return [{**row, "component_split": assignments[row["subject_id"]]} for row in rows]


def prepare(root, report_path):
    stays = pd.read_csv(root / "ed/edstays.csv.gz")
    triage = pd.read_csv(root / "ed/triage.csv.gz")
    vitals = pd.read_csv(root / "ed/vitalsign.csv.gz")
    rows, excluded = prepare_rows(stays, triage)
    rows = assign_patient_splits(rows)
    staging = root / "clinical_inputs_staging.jsonl"
    staging.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = {
        "source": SOURCE_PAGE,
        "license": "Open Data Commons Open Database License v1.0",
        "ed_stays": len(stays),
        "triage_rows": len(triage),
        "vital_sign_rows": len(vitals),
        "source_patients": int(stays.subject_id.nunique()),
        "staged_complete_rows": len(rows),
        "staged_patients": len({r["subject_id"] for r in rows}),
        "excluded_incomplete_or_invalid_rows": sum(excluded.values()),
        "exclusion_error_types": excluded,
        "metadata_fields": list(FIELDS),
        "seed": 42,
        "split_unit": "Patient; all ED stays of a patient remain together",
        "component_split_rows": dict(Counter(r["component_split"] for r in rows)),
        "component_split_patients": {
            s: len({r["subject_id"] for r in rows if r["component_split"] == s})
            for s in ("train", "val", "test")
        },
        "raw_data_persisted": str(root),
        "staging_sha256": hashlib.sha256(staging.read_bytes()).hexdigest(),
        "diagnostic_targets": "Not acquired or inferred",
        "paired_xrays": 0,
        "trimodal_training_ready": False,
        "limitation": "Observed clinical components only. ED arrival time is not an exact measurement timestamp. Study-time linkage and matching X-rays remain unavailable.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("backend/data/external/mimic-iv-ed-demo-2.2"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("audit/mimic_demo_preparation.json"))
    args = parser.parse_args()
    if args.download:
        download(args.root)
    prepare(args.root, args.report)


if __name__ == "__main__":
    main()
