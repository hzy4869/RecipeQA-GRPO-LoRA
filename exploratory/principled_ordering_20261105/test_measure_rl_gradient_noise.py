"""Independent pure-CPU finite-action checks; no Torch/model/GPU needed."""
import itertools
import math
import unittest

from measure_rl_gradient_noise import (NumericalFailure, batch_summary, coefficient_covariance,
    dot, geometry_from_gram, matrix_vector, multiply, psd_diagnostics, symmetric_eigenvalues, transpose)

Q = [.17, .28, .55]
U = [.9, -.2, .4]
G = [[.3, -.9, .2], [1.1, .4, -.3], [-.5, .7, .8]]


def centered_vectors(vectors, q):
    mean = [math.fsum(p * row[d] for p, row in zip(q, vectors)) for d in range(len(vectors[0]))]
    return [[value - mean[d] for d, value in enumerate(row)] for row in vectors]


def gram(vectors):
    return [[dot(left, right) for right in vectors] for left in vectors]


def enumerated(q, u, vectors, k):
    dimension, n = len(vectors[0]), len(q)
    mean = [0.] * dimension
    second = [[0.] * dimension for _ in range(dimension)]
    for ids in itertools.product(range(n), repeat=k):
        probability = math.prod(q[i] for i in ids)
        values = [u[i] for i in ids]
        total = math.fsum(values)
        coefficients = [(value - (total - value) / (k - 1)) / k for value in values]
        estimate = [math.fsum(c * vectors[i][d] for c, i in zip(coefficients, ids)) for d in range(dimension)]
        for x in range(dimension):
            mean[x] += probability * estimate[x]
            for y in range(dimension):
                second[x][y] += probability * estimate[x] * estimate[y]
    covariance = [[second[x][y] - mean[x] * mean[y] for y in range(dimension)] for x in range(dimension)]
    return mean, covariance


class RLOONoiseMathTests(unittest.TestCase):
    def test_enumerated_k2_k3_k4_full_parameter_covariance(self):
        for k in (2, 3, 4):
            terms = coefficient_covariance(Q, U, k)
            mean, covariance = enumerated(Q, U, G, k)
            actual_mean = matrix_vector(transpose(G), terms["mean_coefficients"])
            reconstructed = multiply(transpose(G), multiply(terms["covariance"], G))
            for x in range(3):
                self.assertAlmostEqual(mean[x], actual_mean[x], places=13)
                for y in range(3):
                    self.assertAlmostEqual(covariance[x][y], reconstructed[x][y], places=13)
            result = geometry_from_gram(gram(centered_vectors(G, Q)), Q, U, k, 5)
            self.assertAlmostEqual(result["noise_trace_raw"], sum(covariance[i][i] for i in range(3)), places=13)
            self.assertAlmostEqual(result["noise_trace_raw"], result["noise_trace_independent_pairwise_raw"], places=13)

    def test_spectrum_gram_equals_actual_parameter_covariance(self):
        terms = coefficient_covariance(Q, U, 8)
        actual = multiply(transpose(G), multiply(terms["covariance"], G))
        expected, _ = symmetric_eigenvalues(actual)
        result = geometry_from_gram(gram(centered_vectors(G, Q)), Q, U, 8, 5)
        measured = result["parameter_covariance_spectrum"]["positive_eigenvalues_above_declared_tolerance"]
        expected = [value for value in expected if value > 1e-10 * max(expected)]
        self.assertEqual(len(expected), len(measured))
        for left, right in zip(expected, measured):
            self.assertAlmostEqual(left, right, places=13)
        self.assertFalse(result["parameter_covariance_spectrum"]["eigenvalues_clipped"])

    def test_raw_centered_pairwise_pathwise_and_repeats(self):
        centered = centered_vectors(G, Q)
        for ids in itertools.product(range(3), repeat=3):
            values = [U[i] for i in ids]
            total = math.fsum(values)
            coeff = [(value - (total - value) / 2) / 3 for value in values]
            raw = [math.fsum(c * G[i][d] for c, i in zip(coeff, ids)) for d in range(3)]
            center = [math.fsum(c * centered[i][d] for c, i in zip(coeff, ids)) for d in range(3)]
            pairwise = [math.fsum(.5 * (U[ids[a]] - U[ids[b]]) * (G[ids[a]][d] - G[ids[b]][d]) / 3
                                 for a in range(3) for b in range(a + 1, 3)) for d in range(3)]
            for x, y, z in zip(raw, center, pairwise):
                self.assertAlmostEqual(x, y, places=13)
                self.assertAlmostEqual(x, z, places=13)

    def test_kl_parameter_finite_difference_same_beta_and_reference(self):
        offsets, reference, beta, reward = [.3, -.9, .7], [.2, .5, .3], .7, [0, 1, 0]
        theta = [.17, -.13, .21]
        def objective(t):
            scores = [offset + dot(row, t) for offset, row in zip(offsets, G)]
            raw = [math.exp(s - max(scores)) for s in scores]
            q = [p / math.fsum(raw) for p in raw]
            u = [r - beta * math.log(p / ref) for r, p, ref in zip(reward, q, reference)]
            return dot(q, u), q, u
        _, q, u = objective(theta)
        coefficients = coefficient_covariance(q, u, 8)["mean_coefficients"]
        expected = matrix_vector(transpose(G), coefficients)
        for d in range(3):
            plus, minus = theta[:], theta[:]
            plus[d] += 1e-6
            minus[d] -= 1e-6
            measured = (objective(plus)[0] - objective(minus)[0]) / 2e-6
            self.assertAlmostEqual(expected[d], measured, places=8)

    def test_zero_utilities_common_directions_and_constant_shift(self):
        h = gram(centered_vectors(G, Q))
        constant = geometry_from_gram(h, Q, [1e20] * 3, 8, 5)
        common_gradient = geometry_from_gram([[0.] * 3 for _ in range(3)], Q, U, 8, 5)
        for result in (constant, common_gradient):
            self.assertEqual(result["noise_trace_raw"], 0)
            self.assertFalse(result["signal_resolved"])
            self.assertIsNone(result["noise_to_signal"])
        original = geometry_from_gram(h, Q, U, 8, 5)
        shifted = geometry_from_gram(h, Q, [u + 17 for u in U], 8, 5)
        self.assertAlmostEqual(original["noise_trace_raw"], shifted["noise_trace_raw"], places=13)

    def test_tiny_scale_is_preserved_and_negative_psd_is_not_clipped(self):
        h = gram(centered_vectors(G, Q))
        original = geometry_from_gram(h, Q, U, 8, 5)
        tiny = geometry_from_gram([[value * 1e-40 for value in row] for row in h], Q, U, 8, 5)
        self.assertAlmostEqual(tiny["noise_trace_raw"] / original["noise_trace_raw"], 1e-40, delta=1e-52)
        self.assertAlmostEqual(tiny["noise_to_signal"], original["noise_to_signal"], places=11)
        with self.assertRaises(NumericalFailure) as caught:
            psd_diagnostics([[-1e-50, 0], [0, 1e-50]])
        self.assertEqual(caught.exception.diagnostics["minimum_eigenvalue"], -1e-50)
        self.assertFalse(caught.exception.diagnostics["eigenvalues_clipped"])

    def test_k_scaling_and_normalization(self):
        h = gram(centered_vectors(G, Q))
        four = geometry_from_gram(h, Q, U, 4, 5)
        eight = geometry_from_gram(h, Q, U, 8, 5)
        ratio = four["noise_trace_raw"] / eight["noise_trace_raw"]
        self.assertGreaterEqual(ratio, 2)
        self.assertLessEqual(ratio, 14 / 3)
        self.assertAlmostEqual(eight["loss_noise_trace"], eight["noise_trace_raw"] / 25)
        self.assertAlmostEqual(eight["loss_signal_norm2"], eight["ascent_signal_norm2_raw"] / 25)

    def test_accumulation_uses_actual_batch_mean_not_sum_individual_norms(self):
        h = gram(centered_vectors(G, Q))
        case = {str(k): geometry_from_gram(h, Q, U, k, 5) for k in (2, 4, 8)}
        signal2 = case["2"]["ascent_signal_norm2_raw"]
        same = batch_summary(16 * signal2, [case] * 4)
        self.assertAlmostEqual(same["loss_mean_gradient_norm2"], signal2 / 25)
        self.assertAlmostEqual(same["k"]["8"]["loss_variance_trace"], case["8"]["noise_trace_raw"] / 100)
        canceled = batch_summary(0, [case] * 4)
        self.assertFalse(canceled["signal_resolved"])
        self.assertIsNone(canceled["k"]["8"]["noise_to_signal"])
        self.assertGreater(canceled["k"]["8"]["loss_variance_trace"], 0)

    def test_invalid_distributions_and_shapes_are_rejected(self):
        for q, u, k in (([.2, .3], [1, 2], 8), ([1, -1, 1], U, 8), (Q, [1, math.nan, 2], 8), (Q, U, 1)):
            with self.assertRaises(ValueError):
                coefficient_covariance(q, u, k)
        with self.assertRaises(ValueError):
            geometry_from_gram([[1]], Q, U, 8, 5)

    def test_rare_gold_action_zero_does_not_disappear_in_utility_centering(self):
        q, reward = [1e-20, .4, .6], [1., 0., 0.]
        terms = coefficient_covariance(q, reward, 8)
        self.assertAlmostEqual(terms["mean_coefficients"][0] / 1e-20, 1)
        self.assertAlmostEqual(terms["mean_coefficients"][1] / 1e-20, -.4)
        self.assertAlmostEqual(terms["mean_coefficients"][2] / 1e-20, -.6)
        result = geometry_from_gram(gram(centered_vectors(G, q)), q, reward, 8, 5)
        self.assertGreater(result["noise_trace_raw"], 0)
        self.assertTrue(result["signal_resolved"])


if __name__ == "__main__":
    unittest.main()
