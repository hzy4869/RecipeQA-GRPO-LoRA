"""Input-audit privacy, blank labels, storage confinement, and layout controls."""

import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("build_audit_packet", Path(__file__).resolve().parents[1] / "scripts" / "build_audit_packet.py")
packet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packet)


class AuditPacketTests(unittest.TestCase):
    def test_html_allowlist_and_literal_escaping(self):
        entry = {
            "audit_id": "R001", "qid": "SECRET_QID", "recipe_id": "SECRET_RECIPE",
            "target_labels": ["D", "A", "C", "B"], "source_filename": "SECRET_8_0.jpg",
            "steps": [{"title": '<script>alert("x")</script>', "body": "Mix & bake"}],
            "images": [{"label": label, "asset": f"assets/r001_{label.lower()}.jpg"} for label in "ABCD"],
        }
        html = packet.render_html([entry], "reviewer_1", "hash")
        for secret in ("SECRET_QID", "SECRET_RECIPE", "SECRET_8_0.jpg", "target_labels"):
            self.assertNotIn(secret, html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("Mix &amp; bake", html)
        self.assertIn('value="">尚未填写', html)
        self.assertNotIn(" selected", html)
        self.assertNotIn("https://", html)

    def test_blank_csv_has_no_fabricated_labels(self):
        text = packet.blank_csv(["R001", "R002"], "reviewer_2")
        rows = list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))))
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["reviewer_id"], "reviewer_2")
            for field, _, _ in packet.FIELDS:
                self.assertEqual(row[field], "")
            self.assertEqual(row["notes"], "")

    def test_storage_confinement_including_symlink(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as external:
            root = Path(td).resolve()
            (root / "escape").symlink_to(external)
            with self.assertRaises(ValueError):
                packet.confined(root / "escape" / "image.jpg", root)
            with self.assertRaises(ValueError):
                packet.confined(root / ".." / "outside", root)

    def test_reviewer_storage_is_independent(self):
        first = packet.render_html([], "reviewer_1", "packet")
        second = packet.render_html([], "reviewer_2", "packet")
        self.assertIn("recipeqa-input-audit:packet:reviewer_1", first)
        self.assertNotIn("recipeqa-input-audit:packet:reviewer_2", first)
        self.assertIn("recipeqa-input-audit:packet:reviewer_2", second)

    def test_real_images_anonymous_assets_and_unreadable_retained(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed locally; run this test in the GPU environment")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            images = root / "images"
            images.mkdir()
            filenames = [f"SECRET_RECIPE_{i}_0.jpg" for i in (1, 3, 5, 8)]
            for name in filenames[:3]:
                Image.new("RGB", (1200, 600), (200, 50, 40)).save(images / name)
            record = {
                "qid": "SECRET_QID", "recipe_id": "SECRET_RECIPE", "presentation_id": "SECRET_PRESENTATION",
                "image_paths_display": [filenames[i] for i in (2, 0, 3, 1)],
                "steps": [{"source_step_id": i, "title": "Step", "body": "Visible state"} for i in range(1, 10)],
                "target_labels": ["B", "D", "A", "C"],
            }
            manifest = root / "audit.jsonl"
            manifest.write_text(json.dumps(record) + "\n")
            out = root / "packet"
            report = packet.build_packet(manifest, images, out, allowed_root=root, expected_count=1)
            self.assertEqual(report["questions"], 1)
            self.assertEqual(report["thumbnails"], 3)
            self.assertEqual(report["technical_issues"][0]["label"], "C")
            page = (out / "reviewer_1.html").read_text()
            self.assertIn("图片无法读取", page)
            for secret in ("SECRET_RECIPE", "SECRET_QID", "SECRET_PRESENTATION", *filenames):
                self.assertNotIn(secret, page)
            for image_path in (out / "assets").glob("*.jpg"):
                with Image.open(image_path) as image:
                    self.assertLessEqual(max(image.size), 512)
                    self.assertFalse(image.getexif())
            self.assertTrue((out / "PRIVATE_mapping.json").exists())
            # Preserve frozen anonymous ID mapping instead of silently replacing it.
            record["qid"] = "CHANGED"
            manifest.write_text(json.dumps(record) + "\n")
            with self.assertRaises(ValueError):
                packet.build_packet(manifest, images, out, allowed_root=root, expected_count=1)


if __name__ == "__main__":
    unittest.main()
