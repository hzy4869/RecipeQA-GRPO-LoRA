"""Task-level invariants: recipe isolation, target equivariance, missing images."""

from collections import Counter
from itertools import permutations
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recipeqa.data import (
    DataValidationError, canonicalize_row, clean_text, inspect_images,
    make_presentation, make_training_visits, prepare_manifests, select_sets,
)


def source_row(recipe: str, qid: str, split: str = "train") -> dict:
    images = [f"{recipe}_{i}_0.jpg" for i in (1, 3, 5, 8)]
    return {
        "task": "visual_ordering", "recipe_id": recipe, "qid": qid, "split": split,
        "choice_list": [images, images[::-1], images[1:] + images[:1], images[2:] + images[:2]],
        "answer": 0,
        "context": [{"id": i, "title": f"Step {i}", "body": "<p>Visible <b>state</b></p>"} for i in range(1, 10)],
    }


class DataTests(unittest.TestCase):
    def sources(self) -> dict:
        return {
            split: [canonicalize_row(source_row(f"{split}-r{i}", f"{split}-q{i}-{j}", split), split)
                    for i in range(count) for j in range(2)]
            for split, count in (("train", 15), ("val", 8), ("test", 9))
        }

    def sizes(self) -> dict:
        return {"train": 8, "warmup": 3, "debug": 2, "val": 6,
                "monitor": 2, "bank": 3, "audit": 6, "orbit": 4, "process": 2}

    def test_full_context_and_choice_integrity(self):
        raw = source_row("r", "q")
        record = canonicalize_row(raw, "train")
        self.assertEqual(len(record["steps"]), 9)  # Do not select only four gold-related descriptions.
        self.assertEqual(record["source_step_ids_chronological"], [1, 3, 5, 8])
        self.assertEqual(clean_text('<p>A&nbsp; B</p><script>hidden</script><a href="https://secret">keep label</a> https://remove.me'), "A B keep label")
        raw["choice_list"][1][0] = "different_1_0.jpg"
        with self.assertRaises(DataValidationError):
            canonicalize_row(raw, "train")

    def test_recipe_isolation_and_nested_diagnostics(self):
        sources = self.sources()
        first = select_sets(sources, sizes=self.sizes())
        second = select_sets(sources, sizes=self.sizes())
        self.assertEqual(first, second)
        recipes = lambda name: {r["recipe_id"] for r in first[name]}
        for a, b in (("train", "warmup"), ("train", "debug"), ("warmup", "debug")):
            self.assertFalse(recipes(a) & recipes(b))
        for child, parent in (("monitor", "val"), ("bank", "val"), ("orbit", "audit"), ("process", "orbit")):
            self.assertLessEqual(recipes(child), recipes(parent))
        self.assertEqual(len(first["test"]), 18)  # Keep ALL test questions, not one per recipe.
        sources["val"][0]["recipe_id"] = sources["train"][0]["recipe_id"]
        with self.assertRaises(DataValidationError):
            select_sets(sources, sizes=self.sizes())

    def test_all_layouts_target_mapping_and_path_preservation(self):
        record = canonicalize_row(source_row("r", "q"), "train")
        image_info = {name: {"path": f"nested/{name}", "status": "ok"} for name in record["images_chronological"]}
        for display in permutations(range(4)):
            presented = make_presentation(record, display, manifest_set="orbit", image_info=image_info)
            labels = presented["target_labels"]
            semantic_prediction = [display["ABCD".index(label)] for label in labels]
            self.assertEqual(semantic_prediction, [0, 1, 2, 3])
            redistributed = make_presentation(presented, display[::-1], manifest_set="train_visits")
            self.assertTrue(all(path.startswith("nested/") for path in redistributed["image_paths_display"]))

    def test_visit_exposure_once_per_pass(self):
        records = self.sources()["train"][::2]
        visits = list(make_training_visits(records, passes=2))
        self.assertEqual(len(visits), len(records) * 2)
        self.assertEqual(set(Counter(row["qid"] for row in visits).values()), {2})
        self.assertEqual(len({row["visit_id"] for row in visits}), len(visits))
        for index in (0, 1):
            in_pass = [row for row in visits if row["pass_index"] == index]
            self.assertEqual(len({row["qid"] for row in in_pass}), len(records))

    def test_arbitrary_archive_layout_missing_and_corrupt(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "unexpected" / "deep").mkdir(parents=True)
            Image.new("RGB", (7, 9)).save(root / "unexpected" / "deep" / "good_1_0.jpg")
            (root / "bad_1_0.jpg").write_bytes(b"not a real image")
            info, _ = inspect_images(["good_1_0.jpg", "bad_1_0.jpg", "missing_1_0.jpg"], root)
            self.assertEqual(info["good_1_0.jpg"]["path"], "unexpected/deep/good_1_0.jpg")
            self.assertEqual(info["good_1_0.jpg"]["status"], "ok")
            self.assertEqual(info["bad_1_0.jpg"]["status"], "corrupt")
            self.assertEqual(info["missing_1_0.jpg"]["status"], "missing")
            (root / "other").mkdir()
            Image.new("RGB", (7, 9), (200, 10, 20)).save(root / "other" / "good_1_0.jpg")
            duplicates, _ = inspect_images(["good_1_0.jpg"], root)
            self.assertEqual(duplicates["good_1_0.jpg"]["status"], "ambiguous_basename")

    def test_duplicate_basename_requires_byte_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("train", "val", "test"):
                (root / folder).mkdir()
                (root / folder / "photo_1_0.jpg").write_bytes(b"byte-identical-photo")
            info, report = inspect_images(["photo_1_0.jpg"], root, verify=False)
            item = info["photo_1_0.jpg"]
            self.assertEqual(item["path"], "test/photo_1_0.jpg")
            self.assertEqual(item["status"], "not_decoded")
            self.assertEqual(item["resolution"], "identical_sha256_copies_lexicographic_path")
            self.assertEqual(len(item["candidates"]), 3)
            self.assertEqual(len(set(item["candidate_sha256"].values())), 1)
            self.assertEqual(report["duplicate_basenames_resolved_by_identical_sha256"], 1)
            # Same filename and even same byte count is insufficient evidence.
            (root / "val" / "photo_1_0.jpg").write_bytes(b"BYTE-identical-photo")
            mixed, _ = inspect_images(["photo_1_0.jpg"], root, verify=False)
            self.assertEqual(mixed["photo_1_0.jpg"]["status"], "ambiguous_basename")
            self.assertIsNone(mixed["photo_1_0.jpg"]["path"])
            self.assertEqual(mixed["photo_1_0.jpg"]["resolution"], "unresolved_different_content")

    def test_end_to_end_manifest_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "raw"
            data.mkdir()
            for split, count in (("train", 15), ("val", 8), ("test", 9)):
                (data / f"{split}.json").write_text(json.dumps({"data": [
                    source_row(f"{split}-r{i}", f"{split}-q{i}-{j}", split)
                    for i in range(count) for j in range(2)
                ]}))
            a = prepare_manifests(data, root / "a", sizes=self.sizes())
            b = prepare_manifests(data, root / "b", sizes=self.sizes())
            self.assertEqual(a["manifests"], b["manifests"])
            self.assertEqual(a["manifests"]["orbit"]["records"], 4 * 24)
            self.assertEqual(a["manifests"]["train_visits"]["records"], 16)
            test = {row["qid"]: row for row in map(json.loads, (root / "a" / "test.jsonl").read_text().splitlines())}
            for row in map(json.loads, (root / "a" / "process.jsonl").read_text().splitlines()):
                self.assertEqual(row["display_perm"], test[row["qid"]]["display_perm"])
            self.assertEqual(len(test), 18)


if __name__ == "__main__":
    unittest.main()
