"""CPU-only handoff/selection/resource gates; no process/GPU launch."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from finite_trial_pipeline import (METHODS, POINTS, RL_ORDER, choose_checkpoint,
                                   choose_rl_family, checkpoint_interval, estimate_whole_batch,
                                   predecessor_decision, shuffle_flow, validate_greedy_row)


def predecessor():
    return {"pid": 123, "status": "completed_full_data_sft", "phase": "one_pass_and_fixed_dev_selection_complete",
            "active_child_pid": None, "commands": [{"returncode": 0}] * 11}


def candidates():
    return [{"step": step, "exact": .7, "pair": .9} for step in (0, *POINTS)]


class FiniteCoordinatorTests(unittest.TestCase):
    def test_actual_frozen_greedy_output_without_manifest_set_and_smoke_interval(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/greedy_prediction_v1.json").read_text())
        row, source = fixture["row"], fixture["manifest_fields"]
        self.assertNotIn("manifest_set", row)
        validate_greedy_row(row, source, row["model_weights_sha256"])
        bad = deepcopy(row)
        bad["decoding"]["do_sample"] = True
        with self.assertRaises(RuntimeError):
            validate_greedy_row(bad, source, row["model_weights_sha256"])
        self.assertEqual(checkpoint_interval(4), 1)
        self.assertEqual(checkpoint_interval(256), 16)

    def test_handoff_needs_committed_success_parent_exit_and_all_receipts(self):
        state = predecessor()
        self.assertEqual(predecessor_decision(state, None), "ready")
        live = {"state": "S", "start_ticks": "100"}
        self.assertEqual(predecessor_decision(state, live, 123, "100"), "wait_parent_exit")
        state["status"] = "running"
        self.assertEqual(predecessor_decision(state, live), "wait_completion")
        with self.assertRaises(RuntimeError):
            predecessor_decision(state, None)
        for changed in ("child", "receipt", "pid", "reuse"):
            state = predecessor()
            if changed == "child":
                state["active_child_pid"] = 999
            elif changed == "receipt":
                state["commands"] = [{"returncode": 0}] * 10
            with self.assertRaises(RuntimeError):
                predecessor_decision(state, live, 124 if changed == "pid" else 123,
                                     "101" if changed == "reuse" else "100")

    def test_checkpoint_selection_is_exact_then_pair_then_earlier(self):
        points = candidates()
        self.assertEqual(choose_checkpoint(points)["step"], 0)
        points[2]["pair"] = .91
        self.assertEqual(choose_checkpoint(points)["step"], 32)
        points[3]["exact"] = .71
        self.assertEqual(choose_checkpoint(points)["step"], 48)
        with self.assertRaises(RuntimeError):
            choose_checkpoint(points[1:])

    def test_family_tie_uses_declared_order_without_seed_or_decoder(self):
        arms = {method: {"selected": {"exact": .7, "pair": .9}} for method in RL_ORDER}
        self.assertEqual(choose_rl_family(arms), "exact")
        arms["rloo_kl"]["selected"]["pair"] = .91
        self.assertEqual(choose_rl_family(arms), "rloo_kl")
        arms["exact_kl"]["selected"]["exact"] = .71
        self.assertEqual(choose_rl_family(arms), "exact_kl")

    def test_flow45_keeps_every_record_and_uses_fixed_shuffle(self):
        rows = [{"qid": str(i), "metadata": [i % 4]} for i in range(256)]
        shuffled = shuffle_flow(rows, 20261116)
        self.assertEqual(shuffled, shuffle_flow(rows, 20261116))
        self.assertNotEqual(shuffled, rows)
        self.assertEqual(sorted(row["qid"] for row in shuffled), sorted(row["qid"] for row in rows))
        self.assertEqual(rows[0]["qid"], "0")
        with self.assertRaises(RuntimeError):
            shuffle_flow(rows[:255], 20261116)

    def test_whole_batch_gate_reads_timing_not_accuracy(self):
        smoke = {method: {"step_seconds": 30 + i, "command_seconds": 40 + i, "accuracy": 0}
                 for i, method in enumerate(METHODS)}
        prior = [{"stage": name + str(step), "elapsed_seconds": 30 if name == "dev" else 1}
                 for name in ("dev", "coverage") for step in (256, 512, 768, 945)]
        first = estimate_whole_batch(smoke, 400, prior)
        changed = deepcopy(smoke)
        for value in changed.values():
            value["accuracy"] = 1
        self.assertEqual(first, estimate_whole_batch(changed, 400, prior))
        self.assertFalse(first["score_based_selection"])
        self.assertGreater(first["estimated_seconds"], sum(first["components_seconds"].values()))

    def test_incomplete_or_impossible_timing_fails_closed(self):
        smoke = {method: {"step_seconds": 30, "command_seconds": 40} for method in METHODS}
        prior = [{"stage": name + str(step), "elapsed_seconds": 10}
                 for name in ("dev", "coverage") for step in (256, 512, 768, 945)]
        with self.assertRaises(RuntimeError):
            estimate_whole_batch({"raw_sft": smoke["raw_sft"]}, 400, prior)
        smoke["raw_sft"]["step_seconds"] = 41
        with self.assertRaises(RuntimeError):
            estimate_whole_batch(smoke, 400, prior)


if __name__ == "__main__":
    unittest.main()
