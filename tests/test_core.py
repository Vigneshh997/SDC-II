import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from backend.checkpoint import read_checkpoint
from backend.config import DISEASE_LABELS
from backend.data.dataset import DatasetReadinessError, NIHChestXrayDataModule, load_manifest
from backend.data.validation import MetadataTransform, validate_note
from backend.data.preprocessing import ImagePreprocessing
from backend.explainability import layercam_map, overlay_original, text_attribution, metadata_shap
from backend.inference import InferenceService
from backend.models.losses import AsymmetricLoss, contrastive_loss
from backend.models.multimodal import CrossAttentionFusion
from backend.pipelines.metrics import compute_metrics, select_thresholds
from backend.pipelines.train import preflight, train
from tests.helpers import inputs, tiny_model

torch.set_num_threads(2)


class MetadataTests(unittest.TestCase):
    def test_clahe_preserves_image_geometry(self):
        preprocess = ImagePreprocessing(lambda image: image, clahe=True)
        image = preprocess(Image.new("RGB", (120, 80), (100, 100, 100)))
        self.assertEqual(image.size, (120, 80))
        self.assertEqual(image.mode, "RGB")

    def test_training_only_normalization_and_categories(self):
        transform = MetadataTransform().fit([{"age": 20, "sex": "female"}, {"age": 60, "sex": "male"}])
        encoded = transform.encode({"age": 40, "sex": "unknown"})
        normalized = transform(encoded[None])[0]
        self.assertEqual(normalized[0].item(), 0)
        self.assertEqual(normalized[-1].item(), 1)

    def test_invalid_metadata_rejected(self):
        for metadata in (
            {},
            {"age": "nan", "sex": "male"},
            {"age": 121, "sex": "male"},
            {"age": "", "sex": "male"},
            {"age": True, "sex": "male"},
            {"age": 30, "sex": "invalid"},
            {"age": 30, "sex": "male", "spo2": 98},
        ):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                MetadataTransform().encode(metadata)

    def test_unfitted_normalizer_rejected(self):
        with self.assertRaises(ValueError):
            MetadataTransform()(torch.zeros(1, 5))

    def test_notes_preserve_negation(self):
        self.assertEqual(validate_note(" No  fever.\nNo cough. "), "No fever. No cough.")
        for note in (" ", None, "x" * 20001):
            with self.assertRaises(ValueError):
                validate_note(note)


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = tiny_model().eval()
        self.images, self.notes, self.metadata = inputs()

    def test_outputs_and_all_trainable_branches_receive_gradients(self):
        output = self.model(self.images, self.notes, self.metadata)
        self.assertEqual(output.logits.shape, (2, 14))
        loss = AsymmetricLoss()(output.logits, torch.zeros(2, 14)) + 0.1 * contrastive_loss(
            output.image, output.text, output.metadata
        )
        loss.backward()
        for module in (
            self.model.cnn_refinement,
            self.model.text_proj,
            self.model.meta_encoder,
            self.model.fusion,
            self.model.multi_disease_head,
        ):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.text_encoder.parameters()))

    def test_each_modality_changes_predictions(self):
        with torch.no_grad():
            base = self.model(self.images, self.notes, self.metadata).logits
            variants = [
                self.model(self.images + 2, self.notes, self.metadata).logits,
                self.model(self.images, ["different note", "another note"], self.metadata).logits,
                self.model(self.images, self.notes, self.metadata + torch.tensor([10, 0, 0, 0, 0])).logits,
            ]
        for variant in variants:
            self.assertFalse(torch.allclose(base, variant))

    def test_attention_queries_exclude_own_context(self):
        fusion = CrossAttentionFusion(16, 4, 1).eval()
        values = [torch.randn(2, n, 16) for n in (5, 3, 1)]
        observed = []
        handles = [
            attention.register_forward_pre_hook(lambda module, args: observed.append(args))
            for attention in fusion.layers[0].attention
        ]
        try:
            _, attention = fusion(*values, explain=True)
        finally:
            for handle in handles:
                handle.remove()
        for i, (query, key, value) in enumerate(observed):
            self.assertEqual(query.shape[1], values[i].shape[1])
            expected = torch.cat([values[j] + fusion.modality_ids[j] for j in range(3) if j != i], 1)
            self.assertTrue(torch.equal(key, expected))
            self.assertTrue(torch.equal(key, value))
            self.assertEqual(attention[str(i)].shape[1], 4)

    def test_padding_cannot_influence_fusion(self):
        fusion = CrossAttentionFusion(16, 4, 2).eval()
        image, text, meta = torch.randn(1, 4, 16), torch.randn(1, 3, 16), torch.randn(1, 1, 16)
        mask = torch.tensor([[1, 1, 0]])
        a, _ = fusion(image, text, meta, mask)
        text[:, 2] += 100
        b, _ = fusion(image, text, meta, mask)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_missing_modalities_rejected(self):
        for args in (
            (None, self.notes, self.metadata),
            (self.images, [], self.metadata),
            (self.images, self.notes, None),
            (self.images, ["", "note"], self.metadata),
        ):
            with self.assertRaises(ValueError):
                self.model(*args)

    def test_frozen_encoders_stay_in_eval(self):
        self.model.train()
        self.assertFalse(self.model.text_encoder.training)
        self.assertFalse(self.model.vision_model.training)
        self.assertTrue(self.model.fusion.training)

    def test_optimization_updates_parameters(self):
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-3)
        before = self.model.text_proj.weight.detach().clone()
        output = self.model(self.images, self.notes, self.metadata)
        loss = AsymmetricLoss()(output.logits, torch.ones(2, 14)) + contrastive_loss(
            output.image, output.text, output.metadata
        )
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.text_proj.weight))


class LossTests(unittest.TestCase):
    def test_extreme_asl_is_finite(self):
        logits = torch.tensor([[1000.0, -1000.0]], requires_grad=True)
        loss = AsymmetricLoss()(logits, torch.tensor([[0.0, 1.0]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_infonce_prefers_corresponding_pairs(self):
        eye = torch.eye(4)
        aligned = contrastive_loss(eye, eye, eye)
        shuffled = contrastive_loss(eye, eye.roll(1, 0), eye)
        self.assertLess(aligned.item(), shuffled.item())

    def test_invalid_contrastive_batches_rejected(self):
        with self.assertRaises(ValueError):
            contrastive_loss(torch.ones(1, 4), torch.ones(1, 4), torch.ones(1, 4))
        with self.assertRaises(ValueError):
            contrastive_loss(torch.eye(2), torch.eye(2), torch.eye(2), temperature=0)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = []
        for index, split in enumerate(["train", "train", "val", "test"]):
            image = f"{index}.png"
            Image.new("RGB", (16, 16), (index * 40, 20, 40)).save(self.root / image)
            pid, sid = str(index), f"s{index}"
            self.rows.append(
                {
                    "patient_id": pid,
                    "study_id": sid,
                    "split": split,
                    "image_path": image,
                    "clinical_note": "Artificial software fixture, never a clinical example",
                    "metadata": {"age": 20 + index * 10, "sex": "female" if index % 2 else "male"},
                    "labels": {label: index % 2 for label in DISEASE_LABELS},
                    "provenance": {
                        "observed": True,
                        "license": "TEST FIXTURE ONLY",
                        "note_available_before_prediction": True,
                        "labels_derived_from_note": False,
                        **{
                            m: {"patient_id": pid, "study_id": sid, "source": "ARTIFICIAL SOFTWARE FIXTURE"}
                            for m in ["image", "note", "metadata"]
                        },
                    },
                }
            )
        self.manifest = self.root / "manifest.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        self.manifest.write_text("\n".join(json.dumps(r) for r in self.rows))

    def test_valid_structural_manifest(self):
        self.write()
        self.assertEqual(len(load_manifest(self.manifest, MetadataTransform())), 4)

    def test_patient_leakage_rejected(self):
        self.rows[2]["patient_id"] = self.rows[0]["patient_id"]
        self.write()
        with self.assertRaisesRegex(DatasetReadinessError, "Patient"):
            load_manifest(self.manifest, MetadataTransform())

    def test_unmatched_examination_rejected(self):
        self.rows[0]["provenance"]["note"]["study_id"] = "wrong"
        self.write()
        with self.assertRaisesRegex(DatasetReadinessError, "linkage"):
            load_manifest(self.manifest, MetadataTransform())

    def test_label_derived_notes_rejected(self):
        self.rows[0]["provenance"]["labels_derived_from_note"] = True
        self.write()
        with self.assertRaises(DatasetReadinessError):
            load_manifest(self.manifest, MetadataTransform())

    def test_duplicate_content_rejected(self):
        (self.root / "1.png").write_bytes((self.root / "0.png").read_bytes())
        self.write()
        with self.assertRaisesRegex(DatasetReadinessError, "Duplicate image content"):
            load_manifest(self.manifest, MetadataTransform())

    def test_uncertain_labels_rejected(self):
        self.rows[0]["labels"]["Pneumonia"] = -1
        self.write()
        with self.assertRaises(DatasetReadinessError):
            load_manifest(self.manifest, MetadataTransform())

    def test_nih_and_missing_manifest_rejected(self):
        with self.assertRaises(DatasetReadinessError):
            NIHChestXrayDataModule()
        with self.assertRaises(DatasetReadinessError):
            preflight({"manifest": str(self.manifest)})

    def test_actual_training_loop_and_safe_checkpoint_roundtrip(self):
        self.write()
        review = self.root / "review.md"
        review.write_text("ARTIFICIAL SOFTWARE TEST ONLY. Not reviewed patient data.")
        config = {
            "manifest": str(self.manifest),
            "provenance_review": str(review),
            "batch_size": 2,
            "device": "cpu",
            "max_epochs": 2,
            "output_dir": str(self.root / "run"),
            "model": {},
        }
        with patch("backend.pipelines.train.MultimodalModel", side_effect=lambda **kw: tiny_model(**kw)):
            checkpoint = train(config)
            payload = read_checkpoint(checkpoint)
            self.assertGreaterEqual(payload["training_steps"], 1)
            self.assertEqual(payload["labels"], DISEASE_LABELS)
            self.assertTrue(all(np.isfinite(row["train_infonce"]) for row in payload["history"]))
            train(config, resume=self.root / "run" / "last.pt")


class ExplanationTests(unittest.TestCase):
    def test_layercam_uses_positive_spatial_gradients(self):
        spatial = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        gradients = torch.tensor([[[[-1.0, 1.0], [1.0, -1.0]]]])
        result = layercam_map(spatial, gradients)
        self.assertEqual(result.shape, (1, 2, 2))
        self.assertEqual(result[0, 0, 0], 0)
        self.assertEqual(result[0, 1, 0], 1)

    def test_overlay_matches_original_dimensions(self):
        result = overlay_original(Image.new("RGB", (640, 480)), torch.ones(1, 14, 14))
        image = Image.open(io.BytesIO(base64.b64decode(result.split(",")[1])))
        self.assertEqual(image.size, (640, 480))

    def test_text_and_signed_shap_attributions_execute(self):
        model = tiny_model().eval()
        image, notes, raw = inputs()
        vision, _ = model.encode_image(image[:1])
        text, mask = model.encode_text(notes[:1])
        meta = model.meta_encoder(model.metadata_transform(raw[:1])).unsqueeze(1)
        result = text_attribution(model, vision, meta, notes[0], 0)
        self.assertTrue(result["tokens"])
        self.assertTrue(all(np.isfinite(t["attribution"]) for t in result["tokens"]))
        background = [{"age": 20, "sex": "female"}, {"age": 60, "sex": "male"}]
        shap = metadata_shap(
            model, vision.detach(), text.detach(), mask, {"age": 30, "sex": "male"}, background, 0
        )
        self.assertEqual(set(shap["values"]), {"age", "sex"})
        encoded = model.metadata_transform.encode({"age": 30, "sex": "male"})[None]
        fused, _ = model.fusion(
            vision, text, model.meta_encoder(model.metadata_transform(encoded)).unsqueeze(1), mask
        )
        probability = model.multi_disease_head(fused).sigmoid()[0, 0].item()
        self.assertAlmostEqual(shap["expected_value"] + sum(shap["values"].values()), probability, places=5)

    def test_missing_shap_background_rejected(self):
        with self.assertRaises(ValueError):
            metadata_shap(tiny_model(), None, None, None, {}, [], 0)

    def test_service_report_uses_actual_output_and_releases_lock(self):
        model = tiny_model().eval()
        service = InferenceService(
            model,
            {
                "thresholds": [0.5] * 14,
                "architecture_version": "TEST ONLY",
                "manifest_sha256": "TEST ONLY",
                "training_steps": 0,
            },
        )
        result = service.predict(
            Image.new("RGB", (224, 224)), "software fixture", {"age": 30, "sex": "male"}, "TEST"
        )
        self.assertEqual(result["report"]["predictions"], result["predictions"])
        self.assertIn("Research prototype", result["report"]["disclaimer"])
        self.assertFalse(service.lock.locked())


class MetricsTests(unittest.TestCase):
    def test_perfect_multilabel_predictions(self):
        y = np.array([[0] * 14, [1] * 14])
        result = compute_metrics(y, y * 0.8 + 0.1, [0.5] * 14)
        self.assertEqual(result["micro_f1"], 1)
        self.assertEqual(result["macro_roc_auc"], 1)

    def test_undefined_auc_is_null(self):
        result = compute_metrics(np.zeros((2, 14)), np.zeros((2, 14)), [0.5] * 14)
        self.assertIsNone(result["per_class"]["Pneumonia"]["roc_auc"])
        self.assertEqual(select_thresholds(np.zeros((2, 14)), np.zeros((2, 14))), [0.5] * 14)

    def test_legacy_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"state_dict": {}}, path)
            with self.assertRaisesRegex(ValueError, "Legacy"):
                read_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
