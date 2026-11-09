"""Artificial table fixtures exercise ingestion, not diagnostic performance."""

import unittest

import pandas as pd

from backend.scripts.prepare_mimic_demo import assign_patient_splits, fahrenheit_to_celsius, prepare_rows


class MimicDemoTests(unittest.TestCase):
    def setUp(self):
        self.stays = pd.DataFrame(
            [
                {
                    "subject_id": 10,
                    "stay_id": 20,
                    "gender": "F",
                    "intime": "2100-01-01 10:00:00",
                    "outtime": "2100-01-01 14:00:00",
                }
            ]
        )
        self.triage = pd.DataFrame(
            [
                {
                    "subject_id": 10,
                    "stay_id": 20,
                    "chiefcomplaint": "Artificial fixture",
                    "temperature": 98.6,
                    "heartrate": 80,
                    "resprate": 16,
                    "o2sat": 97,
                }
            ]
        )

    def test_recorded_temperature_is_converted_correctly(self):
        self.assertAlmostEqual(fahrenheit_to_celsius(98.6), 37)
        rows, excluded = prepare_rows(self.stays, self.triage)
        self.assertEqual(excluded, {})
        self.assertAlmostEqual(rows[0]["metadata"]["temperature"], 37)
        self.assertEqual(rows[0]["metadata"]["sex"], "female")
        self.assertIs(rows[0]["trimodal_training_ready"], False)
        self.assertNotIn("image_path", rows[0])
        self.assertNotIn("labels", rows[0])

    def test_missing_or_bad_measurements_are_excluded_not_imputed(self):
        for value in (float("nan"), -1, 500):
            self.triage.loc[0, "heartrate"] = value
            rows, excluded = prepare_rows(self.stays, self.triage)
            self.assertEqual(rows, [])
            self.assertEqual(sum(excluded.values()), 1)

    def test_missing_or_redacted_chief_complaint_is_excluded(self):
        for value in (None, "", "___"):
            self.triage.loc[0, "chiefcomplaint"] = value
            self.assertEqual(prepare_rows(self.stays, self.triage)[0], [])

    def test_unmatched_or_duplicate_stays_are_rejected(self):
        self.triage.loc[0, "stay_id"] = 21
        with self.assertRaises(ValueError):
            prepare_rows(self.stays, self.triage)
        self.triage.loc[0, "stay_id"] = 20
        with self.assertRaises(ValueError):
            prepare_rows(self.stays, pd.concat([self.triage, self.triage]))

    def test_splits_are_reproducible_and_patient_disjoint(self):
        rows = [{"subject_id": str(i), "stay_id": str(j)} for i in range(20) for j in range(2)]
        a = assign_patient_splits(rows)
        b = assign_patient_splits(list(reversed(rows)))
        assignments = {r["subject_id"]: r["component_split"] for r in a}
        self.assertEqual(assignments, {r["subject_id"]: r["component_split"] for r in b})
        for patient in assignments:
            self.assertEqual(len({r["component_split"] for r in a if r["subject_id"] == patient}), 1)
        self.assertEqual(set(assignments.values()), {"train", "val", "test"})


if __name__ == "__main__":
    unittest.main()
