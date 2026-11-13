"""Optional tiny Torch CPU reference, independent of the VLM and CUDA.

These tests are explicitly skipped when Torch is unavailable. They compare the
detached per-candidate shared-parameter gradient construction to differentiating
each complete objective directly, and exercise the actual geometry helper.
Passing them does not validate Qwen multimodal forwards or GPU memory.
"""
from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from gradient_alignment import coefficient_table, vector_summary
from score_permutations import CANDIDATES
from recipeqa.metrics import score_order


@unittest.skipIf(torch is None, "Optional tensor reference requires Torch; no tensor-path pass is claimed")
class TinySharedParameterGradientReference(unittest.TestCase):
    def test_candidate_accumulation_matches_direct_autograd_all_five_objectives(self):
        dtype = torch.float64
        matrix = torch.nn.Parameter(torch.tensor([[.2, -.3, .5], [-.1, .4, .7]], dtype=dtype))
        bias = torch.nn.Parameter(torch.tensor([.6, -.2, .3], dtype=dtype))
        parameters = (matrix, bias)
        features = torch.tensor([[((i * 7) % 29) / 17 - .5, i / 23 - .5] for i in range(24)], dtype=dtype)

        def all_scores():
            # A common score component changes raw sequence likelihood/logZ,
            # but cancels in closed-policy probabilities. This makes the
            # raw-SFT versus closed-CE distinction explicit in the reference.
            return -7 + (torch.tanh(features @ matrix) * bias).sum(-1) + .2 * matrix.sum()

        scores = all_scores()
        q = torch.softmax(scores, dim=0)
        gold = 8
        target = [ord(letter) - ord("A") for letter in CANDIDATES[gold].split()]
        exact = [int(i == gold) for i in range(24)]
        pair = [score_order(text, target_order=target)["pair"] for text in CANDIDATES]
        objectives = {
            "exact": q[gold],
            "pair": (q * torch.tensor(pair, dtype=dtype)).sum(),
            "closed_ce_ascent": scores[gold] - torch.logsumexp(scores, dim=0),
            "raw_sft_sequence_ascent": scores[gold],
            "log_canonical_mass": torch.logsumexp(scores, dim=0),
        }
        direct = {}
        for name, objective in objectives.items():
            gradients = torch.autograd.grad(objective, parameters, retain_graph=True)
            direct[name] = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
        coefficients = coefficient_table(q.detach().tolist(), exact, pair)
        buffers = {name: torch.zeros(sum(parameter.numel() for parameter in parameters), dtype=dtype)
                   for name in coefficients}
        for i in range(24):
            # Recompute each graph: same lifetime pattern as the VLM runner.
            candidate_score = all_scores()[i]
            gradients = torch.autograd.grad(candidate_score, parameters, retain_graph=False, create_graph=False)
            flat = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
            for name, buffer in buffers.items():
                buffer.add_(flat, alpha=coefficients[name][i])
        for name in direct:
            torch.testing.assert_close(buffers[name], direct[name], rtol=1e-10, atol=1e-12)
        result = vector_summary(buffers, float(q[gold].detach()), atol=1e-10, rtol=1e-8)
        self.assertTrue(all(check["relative_l2_gate_passed"] for check in result["identities"].values()))
        torch.testing.assert_close(matrix.detach(), torch.tensor([[.2, -.3, .5], [-.1, .4, .7]], dtype=dtype))
        torch.testing.assert_close(bias.detach(), torch.tensor([.6, -.2, .3], dtype=dtype))
        self.assertTrue(all(parameter.grad is None for parameter in parameters))
        self.assertGreater(float(torch.linalg.vector_norm(
            direct["closed_ce_ascent"] - direct["raw_sft_sequence_ascent"])), 1e-3)

    def test_relative_gate_rejects_small_absolute_but_large_relative_error(self):
        closed = torch.ones(8, dtype=torch.float64) * 1e-10
        buffers = {"closed_ce_ascent": closed,
                   "raw_sft_sequence_ascent": 2 * closed,
                   "log_canonical_mass": closed,
                   "exact": .1 * closed + 1e-12,
                   "pair": closed}
        with self.assertRaisesRegex(ValueError, "relative_l2"):
            vector_summary(buffers, .1, atol=1e-6, rtol=1e-4)

    def test_exact_zero_vectors_have_explicit_zero_identity(self):
        buffers = {name: torch.zeros(8, dtype=torch.float64) for name in
                   ("exact", "pair", "closed_ce_ascent", "raw_sft_sequence_ascent", "log_canonical_mass")}
        result = vector_summary(buffers, .1, atol=1e-6, rtol=1e-4)
        self.assertIsNone(result["comparisons"]["exact__pair"]["cosine"])
        self.assertTrue(all(check["relative_l2_gate_passed"] for check in result["identities"].values()))


if __name__ == "__main__":
    unittest.main()
