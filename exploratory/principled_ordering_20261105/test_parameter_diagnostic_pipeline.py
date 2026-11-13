"""CPU-only handoff, lock-release and non-score budget tests."""
from __future__ import annotations

import fcntl
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import parameter_diagnostic_pipeline as pipeline


def state(status="running"):
    return {"status": status, "pid": 42, "active_child_pid": None,
            "commands": [{"returncode": 0} for _ in range(6)]}


def process(status="S", start="123"):
    return {"state": status, "start_ticks": start}


class FakeHandle:
    def __init__(self, name):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


class FakePath:
    def __init__(self, name):
        self.name = name
        self.handles = []

    def open(self, mode):
        assert mode == "a+"
        handle = FakeHandle(self.name)
        self.handles.append(handle)
        return handle

    def __str__(self):
        return self.name


class HandoffTests(unittest.TestCase):
    def test_requires_committed_completion_and_parent_exit(self):
        self.assertEqual(pipeline.parent_decision(state(), process()), "wait_completion")
        done = state("completed_development_diagnostics")
        self.assertEqual(pipeline.parent_decision(done, process()), "wait_parent_exit")
        self.assertEqual(pipeline.parent_decision(done, None), "ready")
        self.assertEqual(pipeline.parent_decision(done, process("Z")), "ready")
        self.assertEqual(pipeline.parent_decision(done, process("X")), "ready")

    def test_rejects_failure_uncommitted_exit_and_reused_pid(self):
        for value, proc in ((state(), None), (state(), process("Z")),
                            (state("failed_stopped_no_retry"), None), (state("unknown"), process())):
            with self.assertRaises(RuntimeError):
                pipeline.parent_decision(value, proc)
        with self.assertRaisesRegex(RuntimeError, "reused"):
            pipeline.parent_decision(state(), process(start="999"), expected_pid=42, observed_start="123")
        with self.assertRaisesRegex(RuntimeError, "PID changed"):
            pipeline.parent_decision(state(), process(), expected_pid=43)

    def test_incomplete_receipts_never_handoff(self):
        value = state("completed_development_diagnostics")
        value["commands"][-1]["returncode"] = 1
        with self.assertRaises(RuntimeError):
            pipeline.parent_decision(value, None)
        value = state("completed_development_diagnostics")
        value["active_child_pid"] = 44
        with self.assertRaises(RuntimeError):
            pipeline.parent_decision(value, None)

    def test_retry_releases_partial_lockset_before_trying_again(self):
        paths = [FakePath(name) for name in ("a", "b", "c")]
        held = set()
        blocked_once = False
        clock = SimpleNamespace(now=0.)

        def flock(handle, flags):
            nonlocal blocked_once
            if flags == fcntl.LOCK_UN:
                held.remove(handle.name)
            elif handle.name == "b" and not blocked_once:
                blocked_once = True
                raise BlockingIOError()
            else:
                held.add(handle.name)

        retries = []
        def retry(attempt, blocked):
            self.assertFalse(held)
            retries.append((attempt, blocked))

        def sleep(seconds):
            clock.now += seconds

        with patch.object(pipeline.fcntl, "flock", side_effect=flock), \
                patch.object(pipeline.time, "monotonic", side_effect=lambda: clock.now), \
                patch.object(pipeline.time, "sleep", side_effect=sleep):
            handles, attempts = pipeline.acquire_lockset(paths, 5, 1, retry)
            self.assertEqual(attempts, 2)
            self.assertEqual(held, {"a", "b", "c"})
            self.assertEqual(retries, [(1, "b")])
            self.assertTrue(paths[0].handles[0].closed)
            self.assertTrue(paths[1].handles[0].closed)
            pipeline.release_locks(handles)
            self.assertFalse(held)
            self.assertTrue(all(handle.closed for path in paths for handle in path.handles))

    def test_lock_retry_is_bounded_and_closes_handles(self):
        paths = [FakePath(name) for name in ("a", "b")]
        held = set()
        clock = SimpleNamespace(now=0.)

        def flock(handle, flags):
            if flags == fcntl.LOCK_UN:
                held.remove(handle.name)
            elif handle.name == "b":
                raise BlockingIOError()
            else:
                held.add(handle.name)

        def sleep(seconds):
            clock.now += seconds

        with patch.object(pipeline.fcntl, "flock", side_effect=flock), \
                patch.object(pipeline.time, "monotonic", side_effect=lambda: clock.now), \
                patch.object(pipeline.time, "sleep", side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError, "Timed out"):
                pipeline.acquire_lockset(paths, 1, .25)
        self.assertFalse(held)
        self.assertLessEqual(clock.now, 1)
        self.assertTrue(all(handle.closed for path in paths for handle in path.handles))

    def test_smoke_gate_uses_time_and_fixed_stage_count_only(self):
        result = pipeline.timing_gate(50, 3, 1000)
        self.assertTrue(result["passed"])
        self.assertEqual(result["estimated_remaining_seconds"], 900)
        self.assertFalse(result["score_based_selection"])
        self.assertFalse(pipeline.timing_gate(50, 3, 899)["passed"])
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(RuntimeError):
                pipeline.timing_gate(invalid, 3, 1000)


if __name__ == "__main__":
    unittest.main()
