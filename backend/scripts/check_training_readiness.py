"""Read-only readiness evidence using the existing strict training preflight.

Never loads encoder weights, creates a manifest, certifies clinical provenance,
or starts training. Reports aggregates, not clinical text or patient identifiers.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml


def fingerprint(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assess(config):
    # Importing preflight defines model classes but never calls their constructors.
    from backend.config import DISEASE_LABELS
    from backend.pipelines.train import preflight

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    manifest = Path(config.get("manifest", ""))
    review = Path(config.get("provenance_review", ""))
    output = Path(config.get("output_dir", "outputs/xfusion_v2"))
    report = {
        "report_version": 1,
        "scope": "Existing v2 structural preflight only; clinical validity is not certified.",
        "manifest_present": manifest.is_file(),
        "provenance_review_present": review.is_file(),
        "manifest_sha256": fingerprint(manifest),
        "provenance_review_sha256": fingerprint(review),
        "existing_run_artifacts": [
            name for name in ("best.pt", "last.pt", "history.json") if (output / name).exists()
        ],
        "structural_preflight_passed": False,
        "clinical_validity": "not_certified",
        "diagnostic_checkpoint_created": False,
        "next_actions": [],
    }
    if not report["manifest_present"]:
        report["next_actions"].append(
            "Acquire authorized, temporally aligned paired data; do not synthesize it."
        )
    if not report["provenance_review_present"]:
        report["next_actions"].append("Complete substantive source, timing, label and permitted-use review.")
    if report["existing_run_artifacts"]:
        report["next_actions"].append("Use a new output_dir for a fresh run or explicitly resume that run.")
    try:
        rows, _ = preflight(config)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        # Exceptions from third-party parsers can quote raw records: expose type only.
        report["preflight_error_type"] = type(exc).__name__
        report["status"] = "blocked"
        report["next_actions"].append(
            "Review TRAINING_READINESS_REPORT.md and the strict loader; fix source data without relaxing validation."
        )
    else:
        report["structural_preflight_passed"] = True
        report["status"] = "review_required"
        report["split_counts"] = dict(Counter(row["split"] for row in rows))
        report["class_counts"] = {
            split: {
                label: {
                    "positive": sum(row["labels"][label] == 1 for row in rows if row["split"] == split),
                    "negative": sum(row["labels"][label] == 0 for row in rows if row["split"] == split),
                }
                for label in DISEASE_LABELS
            }
            for split in ("train", "val", "test")
        }
        report["next_actions"].append(
            "Review per-class support, timestamps, label independence, licensing and encoder pretraining overlap. "
            "Passing declarations does not prove their truth."
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--output", type=Path, help="Optional new JSON evidence file; existing files are never replaced"
    )
    args = parser.parse_args()
    report = assess(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    serialized = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        # Exclusive creation also protects configuration, source and historical reports.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    print(serialized)
    raise SystemExit(0 if report["structural_preflight_passed"] else 1)


if __name__ == "__main__":
    main()
