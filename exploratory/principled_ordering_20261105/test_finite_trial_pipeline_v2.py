"""CPU-only recovery/deadline/resource gates; no GPU/process launch."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from finite_trial_pipeline_v2 import (METHODS, OLD_STAGES, RECOVERY_DIRECTORY, RL_ORDER,
                                     estimate_recovery_batch, estimate_whole_batch,
                                     recovery_decision, remaining_seconds,
                                     required_recovery_artifacts, validate_cpu_receipt)


def stopped():
    return {"pid": 123, "status": "stopped_budget_no_retry", "active_child_pid": None, "flow_results": {},
            "commands": [{"stage": stage, "returncode": 0} for stage in OLD_STAGES],
            "whole_batch_timing_gate": {"passed": False}}


def timing():
    smoke = {method: {"step_seconds": value, "command_seconds": value + 20}
             for method, value in zip(METHODS, (12, 42, 38, 14))}
    predecessor = [{"stage": name + str(step), "elapsed_seconds": 100 if name == "dev" else 1}
                   for name in ("dev", "coverage") for step in (256, 512, 768, 945)]
    return smoke, predecessor


class FiniteRecoveryTests(unittest.TestCase):
    def test_deadline_is_minimum_of_phase_and_absolute_total(self):
        current = datetime(2026, 11, 5, 18, 0, tzinfo=timezone.utc)
        absolute = current + timedelta(seconds=20000)
        self.assertEqual(remaining_seconds(absolute, 25300, utc_now=current, monotonic_now=100), 20000)
        self.assertEqual(remaining_seconds(absolute, 10100, utc_now=current, monotonic_now=100), 10000)
        self.assertEqual(remaining_seconds(absolute, utc_now=current, monotonic_now=100), 20000)

    def test_expired_absolute_or_phase_cannot_reset_into_new_budget(self):
        current = datetime(2026, 11, 6, 2, 0, tzinfo=timezone.utc)
        self.assertEqual(remaining_seconds(current - timedelta(seconds=1), 25300,
                                          utc_now=current, monotonic_now=100), -1)
        self.assertEqual(remaining_seconds(current + timedelta(seconds=9999), 99,
                                          utc_now=current, monotonic_now=100), -1)
        with self.assertRaises(RuntimeError):
            remaining_seconds(current.replace(tzinfo=None), utc_now=current, monotonic_now=100)

    def test_only_committed_preformal_budget_stop_is_recoverable(self):
        self.assertEqual(recovery_decision(stopped(), None), "ready")
        for mutation in ("status", "child", "flow", "extra_command", "missing_command", "failed_command", "gate"):
            state = stopped()
            if mutation == "status":
                state["status"] = "failed_stopped_no_retry"
            elif mutation == "child":
                state["active_child_pid"] = 999
            elif mutation == "flow":
                state["flow_results"] = {"flow44": {}}
            elif mutation == "extra_command":
                state["commands"].append({"stage": "flow44_train_exact", "returncode": 0})
            elif mutation == "missing_command":
                state["commands"].pop()
            elif mutation == "failed_command":
                state["commands"][-1]["returncode"] = 1
            else:
                state["whole_batch_timing_gate"]["passed"] = True
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                recovery_decision(state, None)

    def test_old_parent_must_exit_before_reuse(self):
        self.assertEqual(recovery_decision(stopped(), {"state": "S"}), "wait_v1_parent_exit")
        for status in ("Z", "X"):
            self.assertEqual(recovery_decision(stopped(), {"state": status}), "ready")

    def test_required_recovery_set_has_full_caches_and_acceptance_not_weights(self):
        required = required_recovery_artifacts()
        self.assertEqual(len(required), 39)
        self.assertTrue(all(name.startswith(RECOVERY_DIRECTORY + "/") for name in required))
        self.assertIn(RECOVERY_DIRECTORY + "/reference256_flow44/reference_policy.jsonl.completion.json", required)
        self.assertIn(RECOVERY_DIRECTORY + "/reference4_smoke/run_signature.json", required)
        self.assertIn(RECOVERY_DIRECTORY + "/coordinator_tests.json", required)
        for method in METHODS:
            self.assertIn(RECOVERY_DIRECTORY + "/smoke/" + method + "/independent_verification.json", required)
        self.assertFalse(any(name.endswith((".safetensors", ".pt", ".pth")) for name in required))

    def test_recovered_cpu_counts_zero_skip_and_no_cuda_are_required(self):
        valid = {"tests_run": 11, "successful": True, "failures": 0, "errors": 0, "skipped": 0, "cuda_initialized": False}
        validate_cpu_receipt(valid, 11)
        for field, value in (("tests_run", 10), ("successful", False), ("failures", 1), ("errors", 1),
                             ("skipped", 1), ("cuda_initialized", True)):
            changed = {**valid, field: value}
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                validate_cpu_receipt(changed, 11)

    def test_rloo_bound_changes_only_training_component_not_original_allowances(self):
        smoke, predecessor = timing()
        saved = deepcopy(smoke)
        old = estimate_whole_batch(smoke, 690, predecessor)
        revised = estimate_recovery_batch(smoke, 690, predecessor)
        increment = 64 * (42 - 14)
        self.assertAlmostEqual(revised["estimated_seconds"] - old["estimated_seconds"], 1.25 * increment)
        self.assertEqual(revised["rloo_step_seconds_bound"], 42)
        for name, value in old["components_seconds"].items():
            self.assertEqual(revised["components_seconds"][name], value + (increment if name == "six_full_training_branches" else 0))
        self.assertEqual(smoke, saved)
        self.assertEqual(revised["safety_factor"], 1.25)
        self.assertEqual(revised["fixed_cpu_allowance_seconds"], 300)

    def test_timing_revision_cannot_select_by_model_score(self):
        smoke, predecessor = timing()
        first = estimate_recovery_batch(smoke, 690, predecessor)
        changed = deepcopy(smoke)
        for method in METHODS:
            changed[method]["accuracy"] = 1
            changed[method]["scientific_outcome"] = "no_effective_adapter_update"
        self.assertEqual(first, estimate_recovery_batch(changed, 690, predecessor))
        self.assertFalse(first["score_based_selection"])
        self.assertEqual(first["rloo_step_seconds_bound"], max(smoke[method]["step_seconds"] for method in RL_ORDER))


if __name__ == "__main__":
    unittest.main()
