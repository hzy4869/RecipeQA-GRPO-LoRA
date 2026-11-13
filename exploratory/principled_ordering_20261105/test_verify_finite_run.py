"""Independent verifier checks: no trainer import, Torch, model or GPU."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import struct
import tempfile
import unittest

from verify_finite_run import (ACTIONS, audit_effective_update, check_policy_fields,
                               independent_terms, recompute_policy, replay_draws, tensor_fingerprint)


def candidates():
    return [{"text": text, "token_ids": [100 + index, 7, 8, 9, 99], "token_count": 5, "eos_token_id": 99}
            for index, text in enumerate(ACTIONS)]


def policy(scores):
    maximum = max(scores)
    z = maximum + math.log(sum(math.exp(value - maximum) for value in scores))
    logs = [value - z for value in scores]
    return [math.exp(value) for value in logs], logs


def objective(scores, reference, gold, beta):
    q, logs = policy(scores)
    return q[gold] - beta * sum(p * (current - ref) for p, current, ref in zip(q, logs, reference))


def write_tensor_file(directory, dtype="F32", values=(1., -2.)):
    name = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight"
    inventory = [{"name": name, "shape": [1, 2], "numel": 2}]
    if dtype == "BF16":
        raw = b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16) for value in values)
    else:
        format_ = {"F32": "f", "F16": "e", "F64": "d"}[dtype]
        raw = struct.pack("<" + format_ * len(values), *values)
    header = {name.replace(".default.", "."): {"dtype": dtype, "shape": [1, 2], "data_offsets": [0, len(raw)]}}
    encoded = json.dumps(header).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    (directory / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw)
    digest = hashlib.sha256(name.encode() + struct.pack("<ff", *values)).hexdigest()
    return inventory, digest


class FiniteVerifierTests(unittest.TestCase):
    def test_exact_and_kl_match_independent_full_objective_finite_difference(self):
        scores = [-4 + i / 23 for i in range(24)]
        _, reference = policy([-5 + (23 - i) / 31 for i in range(24)])
        q, logq = policy(scores)
        for method, beta in (("exact", 0), ("exact_kl", .23)):
            terms = independent_terms(method, q, logq, 7, reference, beta)
            for i in range(24):
                high, low = scores.copy(), scores.copy()
                high[i] += 1e-5; low[i] -= 1e-5
                expected = (objective(high, reference, 7, beta) - objective(low, reference, 7, beta)) / 2e-5
                self.assertAlmostEqual(terms["coefficients"][i], expected, places=9)

    def test_raw_sft_is_unconditional_sequence_derivative(self):
        q, logq = policy([-5.] * 24)
        terms = independent_terms("raw_sft", q, logq, 3, logq, 0)
        self.assertEqual(terms["coefficients"], [float(i == 3) for i in range(24)])
        self.assertNotEqual(terms["coefficients"], [float(i == 3) - p for i, p in enumerate(q)])

    def test_repeated_rloo_samples_have_literal_leave_one_out_multiplicity(self):
        q = [.2, .3, .5] + [0.] * 21
        logq = [math.log(value) if value else -1000 for value in q]
        terms = independent_terms("rloo_kl", q, logq, 0, logq, .2, [0, 0, 0, 1, 1, 2, 2, 2])
        for actual, expected in zip(terms["coefficients"][:3], (15 / 56, -6 / 56, -9 / 56)):
            self.assertAlmostEqual(actual, expected)
        for sample in (0, 2):
            terms = independent_terms("rloo_kl", q, logq, 0, logq, .2, [sample] * 8)
            self.assertEqual(terms["advantages"], [0.] * 8)
            self.assertEqual(terms["coefficients"], [0.] * 24)

    def test_replay_preserves_every_uniform_and_strict_cdf_boundary(self):
        q = [.2, .3, .5] + [0.] * 21
        left, right = random.Random(45), random.Random(45)
        draws, uniforms = replay_draws(q, left)
        expected_uniforms = [right.random() for _ in range(8)]
        self.assertEqual(uniforms, expected_uniforms)
        self.assertEqual(draws, [0 if u < .2 else 1 if u < .5 else 2 for u in uniforms])
        self.assertEqual(left.getstate(), right.getstate())
        class Boundaries:
            values = iter([0., .199, .2, .499, .5, .999, .3, .99999])
            def random(self):
                return next(self.values)
        self.assertEqual(replay_draws(q, Boundaries())[0], [0, 0, 1, 1, 2, 2, 1, 2])

    def test_full_eos_normalization_and_tampering_fail(self):
        grammar = candidates()
        scores = [[-5 - i / 10, 0., 0., 0., 0.] for i in range(24)]
        computed = recompute_policy(scores, grammar)
        self.assertAlmostEqual(sum(computed["q"]), 1)
        self.assertEqual(computed["token_count"], 5)
        check_policy_fields(computed, computed)
        bad = {**computed, "logZ": computed["logZ"] + .01}
        with self.assertRaises(ValueError):
            check_policy_fields(bad, computed)
        grammar[3]["eos_token_id"] = 98
        with self.assertRaisesRegex(ValueError, "EOS"):
            recompute_policy(scores, grammar)
        with self.assertRaises(ValueError):
            recompute_policy(scores[:-1], grammar[:-1])

    def test_actual_safetensor_bytes_reproduce_runtime_float_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for dtype in ("F32", "F16", "BF16", "F64"):
                inventory, expected = write_tensor_file(directory, dtype)
                self.assertEqual(tensor_fingerprint(directory, inventory), expected)
            inventory, _ = write_tensor_file(directory)
            inventory[0]["shape"] = [2, 1]
            with self.assertRaisesRegex(ValueError, "shape"):
                tensor_fingerprint(directory, inventory)

    def test_nonfinite_saved_adapter_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inventory, _ = write_tensor_file(directory, values=(float("nan"), 1.))
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                tensor_fingerprint(directory, inventory)

    def test_ineffective_update_is_valid_but_false_disclosure_is_rejected(self):
        self.assertFalse(audit_effective_update("same", "same", False, "no_effective_adapter_update"))
        self.assertTrue(audit_effective_update("before", "after", True, "effective_training"))
        with self.assertRaises(ValueError):
            audit_effective_update("same", "same", True, "effective_training")


if __name__ == "__main__":
    unittest.main()
