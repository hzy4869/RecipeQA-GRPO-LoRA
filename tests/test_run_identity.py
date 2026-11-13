"""Content identity and canonical-history checks, without pretending to run Trainer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recipeqa.modeling import (
    assert_training_destination, begin_training_attempt, load_images, model_weight_identity,
    read_jsonl, sha256_file, write_json,
)


def checkpoint_fixture(run, step):
    """Sentinel bytes test the identity protocol, not a functional torch checkpoint."""
    checkpoint = run.output / f"checkpoint-{step}"
    checkpoint.mkdir()
    write_json(checkpoint / "trainer_state.json", {"global_step": step})
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes((name + " sentinel").encode())
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter weight sentinel")
    write_json(checkpoint / "adapter_config.json", {"r": 16})
    write_json(checkpoint / "recipeqa_checkpoint_audit.json", {
        "run_signature_sha256": run.signature_sha256,
        "state_sha256": {name: sha256_file(checkpoint / name) for name in
                         ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")},
        "adapter_identity": model_weight_identity(checkpoint, adapter=True),
    })
    return checkpoint


class ContentIdentityTests(unittest.TestCase):
    def test_actual_bytes_change_hash_even_same_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "model.safetensors"
            weight.write_bytes(b"AAAA")
            stamp = weight.stat()
            before = model_weight_identity(directory)
            weight.write_bytes(b"BBBB")
            os.utime(weight, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            after = model_weight_identity(directory)
        self.assertNotEqual(before["weights_sha256"], after["weights_sha256"])

    def test_tokenizer_changes_bundle_but_not_weight_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            write_json(root / "tokenizer_config.json", {"eos": 1})
            before = model_weight_identity(root)
            write_json(root / "tokenizer_config.json", {"eos": 2})
            after = model_weight_identity(root)
        self.assertEqual(before["weights_sha256"], after["weights_sha256"])
        self.assertNotEqual(before["bundle_sha256"], after["bundle_sha256"])

    def test_no_weight_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No base model weight"):
                model_weight_identity(directory)

    def test_missing_image_path_error_identifies_input_and_label(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "qid=q7 presentation=p9 Image B: missing/unresolved"):
                load_images({"qid": "q7", "presentation_id": "p9", "technical_status": "image_issue",
                             "image_paths_display": ["one.jpg", None, "three.jpg", "four.jpg"]}, directory)


class RunHistoryTests(unittest.TestCase):
    def test_nonempty_directory_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assert_training_destination(root, None)
            (root / "trainer_metrics.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "not empty"):
                assert_training_destination(root, None)

    def test_changed_seed_signature_is_rejected_without_modifying_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            first = begin_training_attempt(root, {"seed": 42, "budget": 20}, None)
            first.artifact("resolved_config.json", {"original": True})
            first.append("gradient_audit.jsonl", {"global_step_before_update": 0})
            checkpoint = checkpoint_fixture(first, 1)
            before = (root / "gradient_audit.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "immutable run"):
                begin_training_attempt(root, {"seed": 2026, "budget": 20}, str(checkpoint))
            self.assertEqual((root / "gradient_audit.jsonl").read_bytes(), before)

    def test_optimizer_state_byte_mutation_rejects_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            run = begin_training_attempt(root, {"budget": 20}, None)
            run.artifact("resolved_config.json", {"original": True})
            checkpoint = checkpoint_fixture(run, 1)
            (checkpoint / "optimizer.pt").write_bytes(b"changed state")
            with self.assertRaisesRegex(ValueError, "recovery state content changed"):
                begin_training_attempt(root, {"budget": 20}, str(checkpoint))

    def test_adapter_byte_mutation_rejects_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            run = begin_training_attempt(root, {"budget": 20}, None)
            run.artifact("resolved_config.json", {"original": True})
            checkpoint = checkpoint_fixture(run, 1)
            (checkpoint / "adapter_model.safetensors").write_bytes(b"changed weight")
            with self.assertRaisesRegex(ValueError, "adapter content changed"):
                begin_training_attempt(root, {"budget": 20}, str(checkpoint))

    def test_resume_18_after_20_keeps_20_canonical_steps_and_22_actual_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            payload = {"seed": 42, "budget": 20}
            first = begin_training_attempt(root, payload, None)
            first.artifact("resolved_config.json", {"original": True})
            original_config = (root / "resolved_config.json").read_bytes()
            for step in range(20):
                first.append("gradient_audit.jsonl", {"global_step_before_update": step})
                first.append("step_times.jsonl", {"global_step": step + 1, "optimizer_step_seconds": 1})
            cp18 = checkpoint_fixture(first, 18)
            checkpoint_fixture(first, 20)
            (root / "adapter_final").mkdir()
            (root / "adapter_final" / "adapter_model.safetensors").write_bytes(b"old final")
            first.finish({"status": "completed", "global_step": 20})
            second = begin_training_attempt(root, payload, str(cp18))
            second.artifact("resolved_config.json", {"this_attempt_is_resumed": True})
            self.assertEqual((root / "resolved_config.json").read_bytes(), original_config)
            self.assertEqual(len(read_jsonl(root / "gradient_audit.jsonl")), 18)
            self.assertEqual(len(read_jsonl(root / "step_times.jsonl")), 18)
            self.assertFalse((root / "checkpoint-20").exists())
            self.assertTrue((second.directory / "superseded_before_resume" / "checkpoint-20").is_dir())
            self.assertTrue((second.directory / "superseded_before_resume" / "train_outcome.json").is_file())
            self.assertTrue((second.directory / "superseded_before_resume" / "adapter_final").is_dir())
            for step in (18, 19):
                second.append("gradient_audit.jsonl", {"global_step_before_update": step})
                second.append("step_times.jsonl", {"global_step": step + 1, "optimizer_step_seconds": 1})
            second.finish({"status": "completed", "global_step": 20})
            canonical = read_jsonl(root / "gradient_audit.jsonl")
            self.assertEqual([row["global_step_before_update"] for row in canonical], list(range(20)))
            actual_rows = sum(len(read_jsonl(attempt / "gradient_audit.jsonl")) for attempt in (first.directory, second.directory))
            self.assertEqual(actual_rows, 22)
            self.assertEqual(json.loads((root / "train_outcome.json").read_text())["global_step"], 20)
            self.assertEqual(len(read_jsonl(first.directory / "gradient_audit.jsonl")), 20)


if __name__ == "__main__":
    unittest.main()
