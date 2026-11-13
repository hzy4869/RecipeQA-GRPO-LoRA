"""CPU-only controls for the post-training handoff and greedy-repeat gate."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recipeqa.metrics import score_order
from recipeqa.modeling import _render_recipe, build_messages, hash_json


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "ops" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


handoff = load("wait_and_evaluate")
greedy = load("compare_greedy")
evaluation = load("evaluation_pipeline")


class HandoffControls(unittest.TestCase):
    def setUp(self):
        self.live = {"state": "S", "start_ticks": "123"}
        self.running = {"pid": 11447, "status": "running", "phase": "pair_seed42"}
        self.completed = {**self.running, "status": "completed", "phase": "three_branches_selected",
                          "selected": {name: {} for name in handoff.PRIMARY}, "commands": [{"returncode": 0}]}

    def decide(self, formal, process, observed="123"):
        return handoff.decision(formal, process, 11447, observed)[0]

    def test_wait_for_live_parent_even_after_completion_commit(self):
        self.assertEqual(self.decide(self.running, self.live), "wait")
        self.assertEqual(self.decide(self.completed, self.live), "wait")

    def test_only_committed_completion_plus_observed_exit_can_release(self):
        self.assertEqual(self.decide(self.completed, None), "ready")
        self.assertEqual(self.decide(self.completed, {**self.live, "state": "Z"}), "ready")
        self.assertEqual(self.decide(self.running, None), "stop")
        self.assertEqual(self.decide(self.completed, None, observed=None), "stop")

    def test_failure_pid_reuse_or_incomplete_receipts_stop(self):
        for formal, process in (({**self.running, "status": "failed"}, self.live),
                                ({**self.completed, "pid": 999}, None),
                                (self.completed, {**self.live, "start_ticks": "reused"}),
                                ({**self.completed, "commands": [{}]}, None),
                                ({**self.completed, "phase": "still_selecting"}, None)):
            self.assertEqual(self.decide(formal, process), "stop")

    def test_gpu_query_must_prove_idle(self):
        for rc, stdout in ((1, ""), (0, "1234\n")):
            with patch.object(handoff.subprocess, "run", return_value=type("Result", (), {"returncode": rc, "stdout": stdout, "stderr": ""})()):
                with self.assertRaises(RuntimeError):
                    handoff.require_gpu_idle()


class GreedyRepeatControls(unittest.TestCase):
    def test_independent_files_match_and_response_or_prompt_changes_fail(self):
        with tempfile.TemporaryDirectory(prefix="recipeqa-greedy-control-") as directory:
            left, right = (Path(directory) / name for name in ("a.jsonl", "b.jsonl"))
            records = []
            for index in range(2):
                row = {"run_id": "same_repeat_identity", "model_id": "fixed_model", "checkpoint": "/fixed/weights",
                       "seed": 42, "recipe_id": f"recipe{index}", "qid": f"q{index}", "presentation_id": "p",
                       "display_perm": [0, 1, 2, 3], "target_order": [0, 1, 2, 3], "target_labels": list("ABCD"),
                       "raw_response": "A B C D", "input_audit": {"prompt_text_hash": "fixed", "input_hash": "fixed"},
                       "model_weights_sha256": "fixed", "prompt_kind": "answer", "sample_index": 0,
                       "sample_seed": 42, "decoding": {"do_sample": False}, "input_tokens": 10,
                       "completion_tokens": 4, "reached_token_limit": False, "last_token_id": 1}
                row["score"] = score_order(row["raw_response"], row["display_perm"])
                records.append(row)
            def write(path, values):
                path.write_text("".join(json.dumps(row) + "\n" for row in values))
                path.with_suffix(".jsonl.config.json").write_text(json.dumps({"cli": {"output": str(path), "seed": 42}, "weight_identity": {"base": "fixed_actual_identity"}}))
            write(left, records); write(right, records)
            self.assertEqual(greedy.compare(left, right, 2)["status"], "passed")
            with self.assertRaises(ValueError):
                greedy.compare(left, left, 2)
            changed = json.loads(json.dumps(records)); changed[0]["raw_response"] = "B A C D"
            changed[0]["score"] = score_order(changed[0]["raw_response"], changed[0]["display_perm"])
            write(right, changed)
            self.assertEqual(greedy.compare(left, right, 2)["status"], "different")
            changed = json.loads(json.dumps(records)); changed[0]["input_audit"]["prompt_text_hash"] = "different"
            write(right, changed)
            self.assertEqual(greedy.compare(left, right, 2)["status"], "different")
            write(right, records[:1])
            self.assertEqual(greedy.compare(left, right, 2)["status"], "different")


class HumanVisibleTextControls(unittest.TestCase):
    def test_body_clipping_preserves_protected_target_and_layout(self):
        source = {"qid": "q", "display_perm": [1, 0, 2, 3], "target_labels": list("BACD"),
                  "steps": [{"source_step_id": 1, "title": "Title", "body": "long source body"}]}
        shown = json.loads(json.dumps(source)); shown["steps"][0]["body"] = "long"
        shown["review_text_view"] = {"uniform_body_cap": 1}
        self.assertEqual(evaluation.protected_review_record(source), evaluation.protected_review_record(shown))
        shown["target_labels"] = list("ABCD")
        self.assertNotEqual(evaluation.protected_review_record(source), evaluation.protected_review_record(shown))

    def test_review_text_must_match_actual_prompt_and_cap(self):
        shown = {"qid": "q", "display_perm": [0, 1, 2, 3],
                 "steps": [{"source_step_id": 1, "title": "Title", "body": "visible clipped body"}],
                 "review_text_view": {"uniform_body_cap": 3}}
        text = _render_recipe(shown["steps"], None, None)
        observed = {"qid": "q", "display_perm": [0, 1, 2, 3], "technical_error": None,
                    "input_audit": {"uniform_body_cap": 3, "prompt_text_hash": hash_json(build_messages(text, process=True))}}
        evaluation.check_view_against_actual_prompts([shown], [observed], process=True)
        with self.assertRaises(RuntimeError):
            evaluation.check_view_against_actual_prompts([shown], [observed], process=False)
        observed["input_audit"]["uniform_body_cap"] = 4
        with self.assertRaises(RuntimeError):
            evaluation.check_view_against_actual_prompts([shown], [observed], process=True)


if __name__ == "__main__":
    unittest.main()
