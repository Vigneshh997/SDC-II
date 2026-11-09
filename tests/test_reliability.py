"""Known calibration examples and artifact preservation, using software fixtures only."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.pipelines.metrics import binary_reliability, compute_metrics
from backend.pipelines.train import train


class ReliabilityTests(unittest.TestCase):
    def test_perfect_probabilities_include_endpoints(self):
        result = binary_reliability([0, 1], [0.0, 1.0])
        self.assertEqual(result["brier_score"], 0)
        self.assertEqual(result["positive_event_ece"], 0)
        self.assertEqual(result["reliability_bins"][0]["count"], 1)
        self.assertEqual(result["reliability_bins"][-1]["count"], 1)
        self.assertIsNone(result["reliability_bins"][1]["positive_fraction"])
        json.dumps(result, allow_nan=False)

    def test_calibration_is_distinct_from_discrimination(self):
        # Every score is in the same bin; observed prevalence matches its mean.
        result = binary_reliability([0, 1], [0.5, 0.5])
        self.assertEqual(result["positive_event_ece"], 0)
        self.assertEqual(result["brier_score"], 0.25)
        wrong = binary_reliability([0, 1], [1.0, 0.0])
        self.assertEqual(wrong["positive_event_ece"], 1)
        self.assertEqual(wrong["brier_score"], 1)

    def test_bin_weighting_and_boundaries(self):
        result = binary_reliability([0, 0, 1], [0.1, 0.2, 0.9], bins=2)
        self.assertAlmostEqual(result["positive_event_ece"], (2 * 0.15 + 0.1) / 3)
        self.assertEqual(binary_reliability([1], [0.5], bins=2)["reliability_bins"][1]["count"], 1)

    def test_invalid_inputs_are_rejected(self):
        for y, p, bins in [
            ([], [], 10),
            ([0], [float("nan")], 10),
            ([0], [1.1], 10),
            ([2], [0.5], 10),
            ([0, 1], [0.5], 10),
            ([[0]], [[0.5]], 10),
            ([0], [0.5], 0),
            ([0], [0.5], True),
        ]:
            with self.subTest(y=y, p=p, bins=bins), self.assertRaises(ValueError):
                binary_reliability(y, p, bins)

    def test_multilabel_integration_preserves_existing_metrics(self):
        targets = np.array([[0] * 14, [1] * 14])
        result = compute_metrics(targets, targets * 0.8 + 0.1, [0.5] * 14)
        self.assertEqual(result["micro_f1"], 1)
        self.assertAlmostEqual(result["macro_brier_score"], 0.01)
        self.assertAlmostEqual(result["macro_positive_event_ece"], 0.1)
        self.assertEqual(sum(b["count"] for b in result["per_class"]["Pneumonia"]["reliability_bins"]), 2)
        json.dumps(result, allow_nan=False)


class TrainingArtifactTests(unittest.TestCase):
    def test_fresh_run_cannot_overwrite_existing_artifacts(self):
        for name in ("best.pt", "last.pt", "history.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                artifact = Path(directory) / name
                artifact.write_bytes(b"preserve user artifact")
                with patch("backend.pipelines.train.preflight") as preflight:
                    with self.assertRaises(FileExistsError):
                        train({"output_dir": directory})
                    preflight.assert_not_called()
                self.assertEqual(artifact.read_bytes(), b"preserve user artifact")

    def test_foreign_resume_cannot_replace_another_run(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "last.pt").write_bytes(b"existing run")
            with self.assertRaises(FileExistsError):
                train({"output_dir": directory}, resume=Path(directory) / "other" / "last.pt")


if __name__ == "__main__":
    unittest.main()
