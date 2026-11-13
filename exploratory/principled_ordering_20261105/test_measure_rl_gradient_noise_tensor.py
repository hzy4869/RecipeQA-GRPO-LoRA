"""Optional installed-Torch CPU path tests; no CUDA/model/checkpoint needed.

Local skips do not establish tensor-path correctness. Before any GPU diagnosis,
run this suite in the existing installed remote environment with zero skips.
"""
import math
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from measure_rl_gradient_noise import (batch_summary, centered_gram_torch,
                                      coefficient_covariance, geometry_from_gram)


@unittest.skipIf(torch is None, "Torch is absent; no tensor-path pass is claimed")
class RLOONoiseTensorTests(unittest.TestCase):
    def test_chunk_centering_uses_actual_gradients_and_preserves_common_shift(self):
        q, u = [.17, .28, .55], [.9, -.2, .4]
        raw = torch.tensor([[.25, -.75, .125, .375], [1.125, .5, -.25, -.5], [-.5, .75, .875, .125]], dtype=torch.float64)
        coefficients = coefficient_covariance(q, u, 8)["mean_coefficients"]
        first, mean1, report1 = centered_gram_torch(raw, q, coefficients, chunk_elements=2)
        shifted, mean2, report2 = centered_gram_torch(raw + 2 ** 30, q, coefficients, chunk_elements=3)
        weights = torch.tensor(q, dtype=torch.float64)
        center = raw - (weights @ raw).unsqueeze(0)
        expected = center @ center.T
        torch.testing.assert_close(torch.tensor(first, dtype=torch.float64), expected, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(torch.tensor(shifted, dtype=torch.float64), expected, rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(mean1, mean2, rtol=1e-12, atol=1e-14)
        self.assertLess(report1["q_weighted_center_residual_max"], 1e-15)
        self.assertEqual(report2["actual_chunk_elements"], 3)
        _, _, raw_mean, report3 = centered_gram_torch(raw + 2 ** 30, q, coefficients, chunk_elements=2, include_raw_mean=True)
        direct_raw = torch.tensor(coefficients, dtype=torch.float64) @ (raw + 2 ** 30)
        # Raw common-mode cancellation depends on reduction order; keep that
        # discrepancy visible rather than demanding bitwise BLAS agreement.
        torch.testing.assert_close(raw_mean, direct_raw, rtol=1e-6, atol=1e-7)
        self.assertTrue(report3["raw_mean_vector_returned_for_batch"])

    def test_tiny_shared_autograd_exact_kl_mean_equals_actual_parameter_vector(self):
        theta = torch.nn.Parameter(torch.tensor([[.2, -.3], [.4, .5]], dtype=torch.float64))
        features = torch.tensor([[.2, -.6], [.4, .9], [-.7, .3]], dtype=torch.float64)
        qref = torch.tensor([.2, .5, .3], dtype=torch.float64)
        reward = torch.tensor([0., 1., 0.], dtype=torch.float64)
        beta = .7
        def scores():
            return torch.tanh(features @ theta).sum(-1) + theta.sum() * .3
        values = scores()
        logq = values.log_softmax(0)
        q = logq.exp()
        objective = (q * reward).sum() - beta * (q * (logq - qref.log())).sum()
        actual = torch.autograd.grad(objective, theta)[0].reshape(-1)
        utility = (reward - beta * (logq.detach() - qref.log())).tolist()
        coefficients = coefficient_covariance(q.detach().tolist(), utility, 8)["mean_coefficients"]
        rows = []
        for index in range(3):
            rows.append(torch.autograd.grad(scores()[index], theta)[0].detach().reshape(-1))
        bank = torch.stack(rows)
        gram, measured, report = centered_gram_torch(bank, q.detach().tolist(), coefficients, chunk_elements=2)
        torch.testing.assert_close(measured, actual, rtol=1e-11, atol=1e-13)
        result = geometry_from_gram(gram, q.detach().tolist(), utility, 8, 5)
        self.assertAlmostEqual(result["loss_signal_norm2"], float(torch.dot(-actual / 5, -actual / 5)), places=13)
        self.assertLess(report["raw_centered_mean_error_norm2"], 1e-28)
        self.assertIsNone(theta.grad)

    def test_actual_four_prompt_mean_vector_and_loss_scaling(self):
        q, utility = [.17, .28, .55], [.9, -.2, .4]
        base = torch.tensor([[.3, -.9, .2], [1.1, .4, -.3], [-.5, .7, .8]], dtype=torch.float64)
        coefficients = coefficient_covariance(q, utility, 8)["mean_coefficients"]
        cases, means = [], []
        for multiplier in (1., -.5, .25, 1.2):
            gram, mean, _ = centered_gram_torch(base * multiplier, q, coefficients, chunk_elements=2)
            cases.append({str(k): geometry_from_gram(gram, q, utility, k, 5) for k in (2, 4, 8)})
            means.append(mean)
        summed = torch.stack(means).sum(0)
        summary = batch_summary(float(torch.dot(summed, summed)), cases)
        direct_loss = -summed / 20
        self.assertAlmostEqual(summary["loss_mean_gradient_norm2"], float(torch.dot(direct_loss, direct_loss)), places=13)
        expected_variance = sum(case["8"]["noise_trace_raw"] for case in cases) / 400
        self.assertAlmostEqual(summary["k"]["8"]["loss_variance_trace"], expected_variance, places=13)
        self.assertTrue(summary["signal_resolved"])
        self.assertTrue(math.isfinite(summary["k"]["8"]["noise_to_signal"]))


if __name__ == "__main__":
    unittest.main()
