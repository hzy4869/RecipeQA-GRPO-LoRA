"""Optional realCPUautograd checks; never load a VLM or use CUDA."""
from __future__ import annotations

import importlib.util
import unittest

from finite_policy_train import log_policy, policy_coefficients

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "Local environment has no torch; run explicitly in installed remoteCPUenvironment")
class FinitePolicyTensorTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.features = torch.tensor([[((i + 1) * (j + 2) % 13 - 6) / 7 for j in range(3)] for i in range(24)], dtype=torch.float64)
        self.initial = torch.tensor([.1, -.2, .3], dtype=torch.float64)
        self.rewards = torch.tensor([int(i == 6) for i in range(24)], dtype=torch.float64)
        self.reference = torch.log_softmax(self.features @ (self.initial + .1), dim=0).detach()

    def test_per_candidate_backward_matches_direct_regularized_objective(self):
        torch = self.torch
        for method, beta in (("raw_sft", 0), ("exact", 0), ("exact_kl", .37)):
            direct_theta = torch.nn.Parameter(self.initial.clone())
            scores = self.features @ direct_theta
            logq = torch.log_softmax(scores, dim=0)
            if method == "raw_sft":
                objective = scores[6]
            else:
                q = logq.exp()
                objective = (q * self.rewards).sum() - beta * (q * (logq - self.reference)).sum()
            direct_gradient = torch.autograd.grad(-objective / (5 * 4), direct_theta)[0]
            theta = torch.nn.Parameter(self.initial.clone())
            policy = log_policy((self.features @ theta).detach().tolist())
            coefficients = policy_coefficients(method, policy, self.rewards.int().tolist(), self.reference.tolist(), beta)["coefficients"]
            for index, coefficient in enumerate(coefficients):
                if coefficient:
                    (-coefficient * (self.features[index] @ theta) / (5 * 4)).backward()
            torch.testing.assert_close(theta.grad, direct_gradient, rtol=1e-12, atol=1e-13)

    def test_rloo_logq_and_aggregated_raw_score_backwards_match(self):
        torch = self.torch
        samples = [0, 6, 6, 2, 3, 6, 5, 0]
        theta = torch.nn.Parameter(self.initial.clone())
        scores = self.features @ theta
        logq = torch.log_softmax(scores, dim=0)
        policy = log_policy(scores.detach().tolist())
        terms = policy_coefficients("rloo_kl", policy, self.rewards.int().tolist(), self.reference.tolist(), .17, samples)
        advantages = torch.tensor(terms["advantages"], dtype=torch.float64)
        literal = -(advantages * logq[samples]).mean() / (5 * 4)
        direct_gradient = torch.autograd.grad(literal, theta)[0]
        theta2 = torch.nn.Parameter(self.initial.clone())
        for index, coefficient in enumerate(terms["coefficients"]):
            if coefficient:
                (-coefficient * (self.features[index] @ theta2) / (5 * 4)).backward()
        torch.testing.assert_close(theta2.grad, direct_gradient, rtol=1e-12, atol=1e-13)

    def test_explicit_zero_gradient_materialization_preserves_adam_steps(self):
        torch = self.torch
        parameter = torch.nn.Parameter(torch.tensor([.1], dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=.001, weight_decay=0)
        optimizer.zero_grad(set_to_none=False)
        self.assertIsNone(parameter.grad)  # reason explicit initialization is necessary
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        self.assertEqual(float(optimizer.state[parameter]["step"]), 1)
        torch.testing.assert_close(parameter.detach(), torch.tensor([.1], dtype=torch.float64))
        parameter.grad.fill_(1)
        optimizer.step()
        previous = parameter.detach().clone()
        optimizer.zero_grad(set_to_none=False)
        optimizer.step()  # zero objective gradient still permits existing momentum
        self.assertEqual(float(optimizer.state[parameter]["step"]), 3)
        self.assertNotEqual(float(parameter.detach()), float(previous))


if __name__ == "__main__":
    unittest.main()
