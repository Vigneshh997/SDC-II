"""Readiness reporting cannot substitute declarations for clinical validation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.config import DISEASE_LABELS
from backend.scripts.check_training_readiness import assess, fingerprint, main


class TrainingReadinessTests(unittest.TestCase):
    def test_missing_data_remains_blocked_without_loading_encoders(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "manifest": str(Path(directory) / "absent.jsonl"),
                "provenance_review": str(Path(directory) / "absent.md"),
            }
            with patch("backend.models.multimodal.load_pretrained_encoders") as load:
                result = assess(config)
                load.assert_not_called()
            self.assertEqual(result["status"], "blocked")
            self.assertFalse(result["structural_preflight_passed"])
            self.assertEqual(result["preflight_error_type"], "DatasetReadinessError")
            self.assertIsNone(result["manifest_sha256"])

    def test_parser_errors_do_not_disclose_clinical_content(self):
        secret = "PRIVATE CLINICAL TEXT"
        with patch("backend.pipelines.train.preflight", side_effect=ValueError(secret)):
            result = assess({})
        self.assertNotIn(secret, json.dumps(result))
        self.assertEqual(result["clinical_validity"], "not_certified")

    def test_success_reports_only_aggregates_and_still_requires_review(self):
        rows = [
            {
                "split": split,
                "labels": dict.fromkeys(DISEASE_LABELS, 0),
                "patient_id": "PRIVATE ID",
                "clinical_note": "PRIVATE NOTE",
            }
            for split in ("train", "val", "test")
        ]
        with patch("backend.pipelines.train.preflight", return_value=(rows, None)):
            result = assess({})
        self.assertTrue(result["structural_preflight_passed"])
        self.assertEqual(result["status"], "review_required")
        self.assertEqual(result["split_counts"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual(result["class_counts"]["test"]["Pneumonia"], {"positive": 0, "negative": 1})
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_existing_artifacts_are_reported_and_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            path.write_bytes(b"existing artifact")
            result = assess({"output_dir": directory})
            self.assertEqual(result["existing_run_artifacts"], ["best.pt"])
            self.assertEqual(path.read_bytes(), b"existing artifact")
            self.assertIsNotNone(fingerprint(path))

    def test_nonmapping_config_rejected(self):
        with self.assertRaises(ValueError):
            assess([])

    def test_cli_blocked_exit_code_and_new_evidence_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("{}", encoding="utf-8")
            output = Path(directory) / "readiness.json"
            arguments = ["readiness", "--config", str(config), "--output", str(output)]
            result = {"structural_preflight_passed": False, "status": "blocked"}
            with (
                patch("sys.argv", arguments),
                patch("backend.scripts.check_training_readiness.assess", return_value=result),
                patch("builtins.print"),
            ):
                with self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 1)
            self.assertEqual(json.loads(output.read_text()), result)

    def test_cli_cannot_replace_config_or_other_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.write_text("{}", encoding="utf-8")
            arguments = ["readiness", "--config", str(config), "--output", str(config)]
            with (
                patch("sys.argv", arguments),
                patch(
                    "backend.scripts.check_training_readiness.assess",
                    return_value={"structural_preflight_passed": False},
                ),
            ):
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual(config.read_text(), "{}")


if __name__ == "__main__":
    unittest.main()
