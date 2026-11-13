"""CPU-only S2 controls: fixed data seed, new optimization seed, no GPU jobs."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / ".state/remote_evidence_full"
sys.path.insert(0, str(ROOT / "ops"))
sys.path.insert(0, str(ROOT / "src"))
import seed43_pipeline as s2


@unittest.skipUnless((EVIDENCE / "LOCAL_SNAPSHOT.json").is_file(), "Archived real seed42 CPU evidence not present")
class Seed43Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = json.loads((ROOT / "configs/seed43_execution_policy.json").read_text())
        cls.protocol = json.loads((EVIDENCE / "configs/protocol_lock.json").read_text())
        cls.reference = json.loads((EVIDENCE / "runs/exact_seed42/resolved_config.json").read_text())
        cls.records = s2.read_jsonl(EVIDENCE / "data/manifests/train_visits.jsonl")
        cls.groups = s2.read_jsonl(EVIDENCE / "runs/exact_seed42/reward_groups.jsonl")

    def test_frozen_training_CLI_accepts_model43_with_input_stream42(self):
        spec = importlib.util.spec_from_file_location("frozen_train_grpo", ROOT / "scripts/train_grpo.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        command = s2.train_command(Path("/root/autodl-tmp/rl_recipeqa"), self.reference, "exact", self.policy)
        with patch.object(sys, "argv", command):
            args = module.arguments()
        self.assertEqual(args.seed, 43)
        self.assertEqual(args.manifest, self.reference["cli"]["manifest"])
        self.assertEqual(args.max_steps, 1024)
        self.assertEqual(args.num_generations, 4)
        self.assertTrue(all(row["training_seed"] == 42 and row["visit_id"].startswith("seed42:") for row in self.records))
        self.assertNotIn("torch", sys.modules)

    def test_actual_original_order_passes_and_any_reorder_or_grouping_fails(self):
        s2.verify_stream(self.records, self.groups)
        changed = list(self.groups); changed[0], changed[1] = changed[1], changed[0]
        with self.assertRaises(RuntimeError):
            s2.verify_stream(self.records, changed)
        changed = [dict(row) for row in self.groups]; changed[0]["global_step_before_update"] = 10
        with self.assertRaises(RuntimeError):
            s2.verify_stream(self.records, changed)

    def test_original_selection_rule_and_all_four_points_required(self):
        points = json.loads((EVIDENCE / "runs/exact_seed42/best_checkpoint.json").read_text())["candidates"]
        self.assertEqual(s2.selection(points)["step"], 768)
        with self.assertRaises(RuntimeError):
            s2.selection(points[:3])
        tied = [dict(point, exact=.5, pair=.75) for point in points]
        self.assertEqual(s2.selection(tied)["step"], 256)

    def test_full_readonly_preflight_with_real_archives_and_mocked_hardware_identity(self):
        with tempfile.TemporaryDirectory(prefix="recipeqa-seed43-preflight-") as directory:
            root = Path(directory).resolve()
            for folder in ("src", "scripts", "ops"):
                (root / folder).symlink_to(ROOT / folder, target_is_directory=True)
            (root / "RESEARCH_PLAN.md").symlink_to(ROOT / "RESEARCH_PLAN.md")
            (root / "configs").mkdir(); (root / "metadata").mkdir()
            for source in (ROOT / "configs").glob("*.json"):
                shutil.copy2(source, root / "configs" / source.name)
            shutil.copy2(EVIDENCE / "configs/protocol_lock.json", root / "configs/protocol_lock.json")
            for name in ("formal_training_pipeline", "evaluation_pipeline", "runtime_threads"):
                shutil.copy2(EVIDENCE / f"metadata/{name}.json", root / f"metadata/{name}.json")
            (root / "data").mkdir(); (root / "data/manifests").symlink_to(EVIDENCE / "data/manifests", target_is_directory=True)
            (root / "data/controls").mkdir()
            monitor = s2.read_jsonl(EVIDENCE / "data/manifests/monitor.jsonl")
            controls = [dict(row, manifest_set="greedy_determinism10", control_excluded_from_main=True) for row in monitor[:10]]
            (root / "data/controls/greedy_determinism10.jsonl").write_text("".join(json.dumps(row) + "\n" for row in controls))
            for method in s2.METHODS:
                dest = root / f"runs/{method}_seed42"; dest.mkdir(parents=True)
                for name in ("resolved_config.json", "run_signature.json"):
                    shutil.copy2(EVIDENCE / f"runs/{method}_seed42/{name}", dest / name)
            identities = {self.protocol[key]["directory"]: self.protocol[key] for key in ("base", "W")}
            env = {key: "4" for key in s2.THREADS}
            real_sha = s2.sha256_file
            def fixture_sha(path):
                # Opaque macOS AppleDouble sidecars exist only on the remote
                # frozen tree, not in the source/JSON evidence checkout.
                path = Path(path)
                if not path.exists() and path.name.startswith("._"):
                    return self.protocol["file_sha256"][str(path.relative_to(root))]
                return real_sha(path)
            with patch.dict(os.environ, env), patch.object(s2, "process_info", return_value=None), patch.object(s2, "package_versions", return_value=self.protocol["versions"]), patch.object(s2, "model_weight_identity", side_effect=lambda path, **kw: identities[str(path)]), patch.object(s2, "sha256_file", side_effect=fixture_sha):
                result = s2.preflight(root, root / "configs/seed43_execution_policy.json")
                self.assertEqual(result[5]["seed43_model_seed"], 43)
                self.assertEqual(result[5]["fixed_training_data_seed"], 42)
                self.assertFalse((root / "metadata/seed43_pipeline.json").exists())
                with patch.dict(os.environ, {"OMP_NUM_THREADS": "72"}):
                    with self.assertRaises(RuntimeError):
                        s2.preflight(root, root / "configs/seed43_execution_policy.json")
                state = json.loads((root / "metadata/evaluation_pipeline.json").read_text()); state["status"] = "running"
                (root / "metadata/evaluation_pipeline.json").write_text(json.dumps(state))
                with self.assertRaises(RuntimeError):
                    s2.preflight(root, root / "configs/seed43_execution_policy.json")


if __name__ == "__main__":
    unittest.main()
