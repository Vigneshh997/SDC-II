"""API tests use artificial bytes and never serve diagnostic predictions."""

import io
import json
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.context = TestClient(app)
        self.client = self.context.__enter__()
        stream = io.BytesIO()
        Image.new("RGB", (32, 32)).save(stream, format="PNG")
        self.image = stream.getvalue()
        self.data = {
            "clinical_note": "Artificial software test input",
            "metadata": json.dumps({"age": 30, "sex": "male"}),
            "examination_id": "TEST",
            "paired_inputs_confirmed": "true",
        }

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def post(self, data=None, image=None, mime="image/png"):
        return self.client.post(
            "/predict",
            data=self.data if data is None else data,
            files={"image": ("test.png", self.image if image is None else image, mime)},
        )

    def test_missing_checkpoint_reports_not_ready(self):
        self.assertEqual(self.client.get("/health").json()["status"], "not_ready")
        self.assertEqual(self.post().status_code, 503)

    def test_missing_each_required_input(self):
        for field in self.data:
            data = {k: v for k, v in self.data.items() if k != field}
            with self.subTest(field=field):
                self.assertEqual(self.post(data=data).status_code, 422)
        self.assertEqual(self.client.post("/predict", data=self.data).status_code, 422)

    def test_invalid_metadata_and_empty_note(self):
        for changes in (
            {"metadata": "{}"},
            {"metadata": "NaN"},
            {"metadata": "invalid"},
            {"clinical_note": "   "},
            {"paired_inputs_confirmed": "false"},
        ):
            self.assertEqual(self.post(data={**self.data, **changes}).status_code, 422)

    def test_corrupt_image(self):
        self.assertEqual(self.post(image=b"not an image").status_code, 422)

    def test_unsupported_mime_and_spoofing(self):
        self.assertEqual(self.post(mime="application/pdf").status_code, 415)
        self.assertEqual(self.post(mime="image/jpeg").status_code, 415)

    def test_oversize_upload(self):
        self.assertEqual(self.post(image=b"0" * (15 * 1024 * 1024 + 1)).status_code, 413)

    def test_schema_matches_observed_fields(self):
        schema = self.client.get("/schema").json()
        self.assertEqual(set(schema["metadata_fields"]), {"age", "sex"})


if __name__ == "__main__":
    unittest.main()
