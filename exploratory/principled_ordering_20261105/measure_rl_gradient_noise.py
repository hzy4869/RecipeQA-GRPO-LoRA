#!/usr/bin/env python3
"""Read-only, fixed-training-prefix LoRA RLOO noise diagnosis.

NEW analysis only. No optimizer, updates, generation, held-out data, model
selection, or persistence of parameter-sized gradient vectors. A future root-
locked execution policy is required; this file does not launch itself remotely.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ops"))
from recipeqa.metrics import score_order
from recipeqa.modeling import (InputConfig, adapter_fingerprint, encode_messages, ensure_data_disk,
    frozen_vision_fingerprint, hash_json, load_images, load_model, load_processor,
    model_weight_identity, package_versions, prepare_record, read_jsonl,
    sha256_file, trainable_inventory, write_json)
from score_permutations import CANDIDATES, candidate_suffixes, score_candidates
from gradient_alignment import differentiable_logp, parameter_fingerprint
from finite_policy_train import log_policy, pixel_identities, policy_coefficients, sources as trainer_sources
from calibrate_beta import validate_reference
from parameter_diagnostic_pipeline import acquire_lockset, release_locks
from wait_and_evaluate import process_info, require_gpu_idle

MANIFEST = "exploratory/principled_ordering_20261105/full_train_data/train_all7558.jsonl"
MANIFEST_SHA = "b9979ee0612679d2e9460dd3bb8a6f68e7dea6794bc6cdcc3e1defdc87d8d949"
REFERENCE = "exploratory/principled_ordering_20261105/finite_trials/reference256_flow44/reference_policy.jsonl"
BETA_REPORT = "exploratory/principled_ordering_20261105/finite_trials/beta_calibration.json"
FINITE_STATE = "exploratory/principled_ordering_20261105/finite_trials_v2/state.json"
DEADLINE = "2026-11-06T01:31:05Z"
LABELS = ("common_initial", "exact64", "exact_kl64", "rloo_kl64")
LOCKS = ("formal_training_pipeline", "evaluation_pipeline", "evaluation_handoff", "seed43_pipeline", "reward_search_20261105")
FROZEN = {
    "exploratory/principled_ordering_20261105/finite_policy_train.py": "34e29cc48744626dec27f99afdd5bdf34660130cf45f17e6d3ff205786a446a5",
    "exploratory/principled_ordering_20261105/score_permutations.py": "affeda80416145d2b6a134172e812d15d48a86d77c67128bd2e5f84e0c341819",
    "exploratory/principled_ordering_20261105/gradient_alignment.py": "1adff082850e0b83a9a70c83565638c02b8efeebdf0fb0ade49af937df9fe41b",
}


class NumericalFailure(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class BudgetStop(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def dot(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def matrix_vector(matrix, vector):
    return [dot(row, vector) for row in matrix]


def quadratic(matrix, vector):
    return dot(vector, matrix_vector(matrix, vector))


def transpose(matrix):
    return list(map(list, zip(*matrix)))


def multiply(left, right):
    columns = transpose(right)
    return [[dot(row, column) for column in columns] for row in left]


def symmetric_eigenvalues(matrix, rtol=2e-14):
    """Small symmetric Jacobi eigensolver; retains signed raw eigenvalues.

    Pure Python keeps mathematical CPU tests independent of CUDA/Torch. No
    eigenvalue is clipped or floored; the caller reports tolerated roundoff.
    """
    n = len(matrix)
    require(n and all(len(row) == n for row in matrix), "Need a square matrix")
    a = [list(map(float, row)) for row in matrix]
    require(all(math.isfinite(v) for row in a for v in row), "Nonfinite matrix")
    scale = max(abs(v) for row in a for v in row)
    if scale == 0:
        return [0.0] * n, {"iterations": 0, "max_offdiagonal": 0.0, "matrix_scale": 0.0}
    residual = None
    for iteration in range(max(1, 100 * n * n)):
        p, q, residual = 0, 0, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > residual:
                    p, q, residual = i, j, abs(a[i][j])
        if residual <= rtol * scale:
            return sorted(a[i][i] for i in range(n)), {"iterations": iteration, "max_offdiagonal": residual,
                                                       "matrix_scale": scale, "relative_offdiagonal": residual / scale}
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        tau = (aqq - app) / (2 * apq)
        t = math.copysign(1.0, tau) / (abs(tau) + math.hypot(1.0, tau))
        c = 1 / math.sqrt(1 + t * t)
        s = t * c
        for k in range(n):
            if k not in (p, q):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = a[p][k] = c * akp - s * akq
                a[k][q] = a[q][k] = s * akp + c * akq
        a[p][p], a[q][q] = app - t * apq, aqq + t * apq
        a[p][q] = a[q][p] = 0.0
    raise NumericalFailure("Jacobi spectrum did not converge", {"residual": residual, "scale": scale})


def psd_diagnostics(matrix, rtol=1e-9):
    n = len(matrix)
    require(n and all(len(row) == n for row in matrix), "PSD input is not square")
    scale = max(abs(v) for row in matrix for v in row)
    asymmetry = max(abs(matrix[i][j] - matrix[j][i]) for i in range(n) for j in range(n))
    eigenvalues, convergence = symmetric_eigenvalues(matrix)
    tolerance = rtol * scale  # No fixed absolute floor that masks tiny geometry.
    report = {"eigenvalues_raw_signed": eigenvalues, "minimum_eigenvalue": min(eigenvalues),
              "matrix_scale": scale, "absolute_psd_tolerance": tolerance, "relative_psd_tolerance": rtol,
              "max_symmetry_residual": asymmetry, "raw_negative_eigenvalues": [v for v in eigenvalues if v < 0],
              "effective_positive_rank": sum(v > tolerance for v in eigenvalues),
              "positive_eigenvalues_above_declared_tolerance": [v for v in eigenvalues if v > tolerance],
              "eigenvalues_clipped": False, "convergence": convergence}
    if asymmetry > tolerance or min(eigenvalues) < -tolerance:
        raise NumericalFailure("Matrix is not PSD/symmetric within its reported scale", report)
    report["status"] = "exact_zero_matrix" if scale == 0 else "psd_with_explicit_roundoff_tolerance"
    return report


def coefficient_covariance(q, utility, k):
    """Exact conditional RLOO coefficient covariance; unrestricted support size.

    Returns a constructive covariance factor F (n by 2n+1) and direct B. This
    avoids taking square roots of numerically negative eigenvalues of B.
    """
    n = len(q)
    require(n >= 2 and len(utility) == n and type(k) is int and k >= 2, "Invalid RLOO dimensions/K")
    require(all(math.isfinite(v) and v >= 0 for v in q) and abs(math.fsum(q) - 1) <= 1e-12
            and all(math.isfinite(v) for v in utility), "Invalid probabilities/utilities")
    # Anchor at the most probable utility, not a possibly vanishingly rare
    # gold outlier. Subtracting a rare reward1 can round an expected1e-20
    # reward to zero when its other probabilities already sum to float1.
    anchor = utility[max(range(n), key=lambda index: q[index])]
    utility_differences = [u - anchor for u in utility]
    difference_mean = dot(q, utility_differences)
    mean = anchor + difference_mean
    v = [value - difference_mean for value in utility_differences]
    c = [p * value for p, value in zip(q, v)]
    sigma2 = dot(q, [value * value for value in v])
    z = [[float(i == j) - q[j] for j in range(n)] for i in range(n)]
    s0 = [[float(i == j) * q[i] - q[i] * q[j] for j in range(n)] for i in range(n)]
    t0 = [[math.fsum(q[h] * v[h] ** 2 * z[h][i] * z[h][j] for h in range(n))
           for j in range(n)] for i in range(n)]
    b = [[(t0[i][j] - c[i] * c[j]) / k
          + (c[i] * c[j] + sigma2 * s0[i][j]) / (k * (k - 1)) for j in range(n)] for i in range(n)]
    columns = [[math.sqrt(q[i] / k) * (v[i] * z[i][j] - c[j]) for j in range(n)] for i in range(n)]
    columns.append([value / math.sqrt(k * (k - 1)) for value in c])
    columns += [[math.sqrt(sigma2 * q[i] / (k * (k - 1))) * z[i][j] for j in range(n)] for i in range(n)]
    factor = transpose(columns)
    reconstructed = multiply(factor, transpose(factor))
    residual = max(abs(b[i][j] - reconstructed[i][j]) for i in range(n) for j in range(n))
    scale = max(abs(value) for row in b for value in row)
    require(residual <= 1e-11 * scale if scale else residual == 0, "Constructive coefficient covariance differs")
    return {"mean_utility": mean, "centered_utility": v, "utility_variance": sigma2,
            "mean_coefficients": c, "coefficient_sum_raw": math.fsum(c), "covariance": b, "factor": factor,
            "factor_reconstruction_absolute_residual": residual,
            "factor_reconstruction_relative_residual": residual / scale if scale else None}


def geometry_from_gram(gram, q, utility, k, token_count, psd_rtol=1e-9):
    n = len(q)
    require(len(gram) == n and all(len(row) == n for row in gram) and type(token_count) is int and token_count > 0,
            "Gram/token-count shape mismatch")
    gram_psd = psd_diagnostics(gram, psd_rtol)
    terms = coefficient_covariance(q, utility, k)
    c, b, f = terms["mean_coefficients"], terms["covariance"], terms["factor"]
    noise_gram = multiply(transpose(f), multiply(gram, f))
    # Symmetry is checked before any calculation that assumes it; do not silently symmetrize.
    noise_psd = psd_diagnostics(noise_gram, psd_rtol)
    noise_trace = math.fsum(noise_gram[i][i] for i in range(len(noise_gram)))
    direct_trace = math.fsum(gram[i][j] * b[j][i] for i in range(n) for j in range(n))
    signal2 = quadratic(gram, c)
    # Independent pairwise U-statistic route on the actual centered Gram.
    centered_u = terms["centered_utility"]
    m_trace = math.fsum(q[i] * centered_u[i] ** 2 * gram[i][i] for i in range(n))
    first_projection_variance = m_trace - signal2
    kernel_variance = .25 * math.fsum(q[i] * q[j] * (utility[i] - utility[j]) ** 2
        * (gram[i][i] + gram[j][j] - 2 * gram[i][j]) for i in range(n) for j in range(n)) - signal2
    pairwise_trace = ((k - 2) * first_projection_variance + 2 * kernel_variance) / (k * (k - 1))
    scale = max(abs(noise_trace), abs(direct_trace))
    require(abs(noise_trace - direct_trace) <= 1e-9 * scale if scale else noise_trace == direct_trace,
            "Gram trace and constructive covariance trace differ")
    pairwise_scale = max(abs(noise_trace), abs(pairwise_trace))
    if not (abs(noise_trace - pairwise_trace) <= 1e-8 * pairwise_scale if pairwise_scale else noise_trace == pairwise_trace):
        raise NumericalFailure("Independent pairwise U-statistic trace differs", {"constructive_trace": noise_trace,
            "pairwise_trace": pairwise_trace, "scale": pairwise_scale, "raw_M_trace": m_trace,
            "raw_first_projection_variance": first_projection_variance, "raw_kernel_variance": kernel_variance})
    require(noise_trace >= 0 and signal2 >= 0, "Negative signal/noise scalar; preserve failure, no clamping")
    # A scale-relative threshold makes an unresolved near-zero signal explicit.
    signal_scale = math.fsum(abs(c[i] * gram[i][j] * c[j]) for i in range(n) for j in range(n))
    signal_resolved = signal2 > psd_rtol * signal_scale if signal_scale else False
    hc = matrix_vector(gram, c)
    along = quadratic(b, hc) / signal2 if signal_resolved else None
    perpendicular = noise_trace - along if along is not None else None
    scalar_tolerance = psd_rtol * noise_trace
    if along is not None and (along < -scalar_tolerance or perpendicular < -scalar_tolerance):
        raise NumericalFailure("Projected noise scalar is outside its covariance scale",
            {"along_raw": along, "perpendicular_raw": perpendicular, "trace_raw": noise_trace,
             "absolute_tolerance": scalar_tolerance, "clipped": False})
    return {"k": k, "objective": "same-theta same-beta same-qref analytic Exact-KL target; beta0 only explicitly labelled",
            "ascent_signal_norm2_raw": signal2, "noise_trace_raw": noise_trace,
            "noise_trace_direct_formula_raw": direct_trace, "trace_formula_residual_raw": noise_trace - direct_trace,
            "noise_trace_independent_pairwise_raw": pairwise_trace,
            "pairwise_trace_residual_raw": pairwise_trace - noise_trace,
            "pairwise_trace_relative_residual": abs(pairwise_trace - noise_trace) / pairwise_scale if pairwise_scale else None,
            "pairwise_first_projection_variance_raw": first_projection_variance, "pairwise_kernel_variance_raw": kernel_variance,
            "noise_along_unit_mean_raw": along, "noise_perpendicular_raw": perpendicular,
            "projection_absolute_roundoff_tolerance": scalar_tolerance, "projection_scalars_clipped": False,
            "signal_scale_absolute_terms": signal_scale, "signal_resolved": signal_resolved,
            "signal_status": "resolved" if signal_resolved else "zero_or_numerically_unresolved",
            "noise_to_signal": noise_trace / signal2 if signal_resolved else None,
            "loss_gradient_rule": "-G/C; four independent prompts accumulate with another/4; no clipping/Adam here",
            "token_count_C": token_count, "loss_signal_norm2": signal2 / token_count ** 2,
            "loss_noise_trace": noise_trace / token_count ** 2,
            "loss_noise_along_unit_mean": along / token_count ** 2 if along is not None else None,
            "loss_noise_perpendicular": perpendicular / token_count ** 2 if perpendicular is not None else None,
            "loss_covariance_eigenvalues_raw_signed": [value / token_count ** 2 for value in noise_psd["eigenvalues_raw_signed"]],
            "loss_noise_nonzero_spectrum_above_tolerance": [value / token_count ** 2 for value in
                 noise_psd["positive_eigenvalues_above_declared_tolerance"]],
            "centered_gram_psd": gram_psd, "parameter_covariance_spectrum": noise_psd,
            "coefficient_covariance": b, "mean_coefficients": c,
            "coefficient_sum_raw": terms["coefficient_sum_raw"],
            "factor_reconstruction_absolute_residual": terms["factor_reconstruction_absolute_residual"],
            "utility_variance": terms["utility_variance"]}


def centered_gram_torch(raw, q, coefficients, chunk_elements=65536, include_raw_mean=False):
    """FP64 chunk centering on actual measured CPU gradients; RAM only."""
    import torch

    require(raw.device.type == "cpu" and raw.ndim == 2 and raw.shape[0] == len(q), "Expected candidate-by-parameter CPU matrix")
    require(chunk_elements > 0 and len(coefficients) == len(q), "Invalid chunk/coefficient shape")
    n, d = raw.shape
    gram = torch.zeros((n, n), dtype=torch.float64)
    mean = torch.empty(d, dtype=torch.float64)
    actual_raw_mean = torch.empty(d, dtype=torch.float64) if include_raw_mean else None
    weights = torch.tensor(q, dtype=torch.float64)
    coeff = torch.tensor(coefficients, dtype=torch.float64)
    raw_mean_error2 = raw_mean_norm2 = centered_mean_norm2 = 0.0
    center_residual_max = 0.0
    for begin in range(0, d, chunk_elements):
        block = raw[:, begin:begin + chunk_elements].double()
        require(bool(torch.isfinite(block).all()), "Nonfinite measured candidate gradient")
        # Subtract a fixed actual candidate first, then the q-mean of differences.
        differences = block - block[0:1]
        centered = differences - (weights @ differences).unsqueeze(0)
        gram += centered @ centered.T
        local_mean = coeff @ centered
        raw_mean = coeff @ block
        mean[begin:begin + block.shape[1]] = local_mean
        if actual_raw_mean is not None:
            actual_raw_mean[begin:begin + block.shape[1]] = raw_mean
        raw_mean_error2 += float(torch.sum((raw_mean - local_mean) ** 2))
        raw_mean_norm2 += float(torch.sum(raw_mean ** 2))
        centered_mean_norm2 += float(torch.sum(local_mean ** 2))
        center_residual_max = max(center_residual_max, float((weights @ centered).abs().max()))
    report = {"raw_centered_mean_error_norm2": raw_mean_error2,
            "raw_mean_norm2": raw_mean_norm2, "centered_mean_norm2": centered_mean_norm2,
            "raw_centered_mean_relative_l2": math.sqrt(raw_mean_error2 / centered_mean_norm2) if centered_mean_norm2 else None,
            "q_weighted_center_residual_max": center_residual_max,
            "coefficient_sum_raw": math.fsum(coefficients), "actual_chunk_elements": chunk_elements,
            "raw_mean_vector_returned_for_batch": include_raw_mean,
            "aggregation": "FP64 fixed-candidate subtraction then actual q-centering before Gram; raw gradients held CPU in original dtype"}
    return (gram.tolist(), mean, actual_raw_mean, report) if include_raw_mean else (gram.tolist(), mean, report)


def batch_summary(mean_sum_norm2, cases, psd_rtol=1e-9):
    """Four actual mean vectors have already been summed in parameter space."""
    require(len(cases) == 4 and mean_sum_norm2 >= 0 and math.isfinite(mean_sum_norm2), "Need one complete accumulation batch")
    counts = {case["2"]["token_count_C"] for case in cases}
    require(len(counts) == 1, "Batch canonical completion normalization C differs")
    count = counts.pop()
    scale = math.fsum(math.sqrt(case["2"]["ascent_signal_norm2_raw"]) for case in cases) ** 2
    resolved = mean_sum_norm2 > psd_rtol * scale if scale else False
    result = {"prompts": 4, "C": count, "actual_raw_mean_sum_norm2": mean_sum_norm2,
              "signal_scale_sum_norm_upper_bound": scale, "signal_resolved": resolved,
              "loss_mean_gradient_norm2": mean_sum_norm2 / (16 * count ** 2), "k": {},
              "rule": "loss mean = -sum(G_t)/(4C); independent sampling covariance=sum(Cov_t)/(16C^2)",
              "cross_prompt_signal": "norm of actual FP64 accumulated mean vector; not inferred from separate per-prompt Grams"}
    for k in (2, 4, 8):
        variance_sum = math.fsum(case[str(k)]["noise_trace_raw"] for case in cases)
        result["k"][str(k)] = {"raw_variance_trace_sum": variance_sum,
                "loss_variance_trace": variance_sum / (16 * count ** 2),
                "noise_to_signal": variance_sum / mean_sum_norm2 if resolved else None}
    return result


def load_json(path):
    return json.loads(Path(path).read_text())


def project_path(value):
    result = (ROOT / value).resolve()
    require(result.is_relative_to(ROOT), "Input/output escaped mounted project data disk")
    return result


def file_receipt(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def required_artifacts(cpu_preflight):
    result = {MANIFEST, REFERENCE, REFERENCE + ".config.json", REFERENCE + ".completion.json", BETA_REPORT, FINITE_STATE, cpu_preflight}
    for method in ("exact", "exact_kl", "rloo_kl"):
        run = "exploratory/principled_ordering_20261105/finite_trials_v2/flow44/" + method + "/"
        result.update(run + name for name in ("run_signature.json", "train_outcome.json", "independent_verification.json",
                                              "checkpoint-64/finite_policy_checkpoint_audit.json"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="Future root-reviewed locked execution policy; no defaults or model picking")
    args = parser.parse_args()
    require(ROOT.is_relative_to("/root/autodl-tmp") and ROOT.stat().st_dev != Path("/").stat().st_dev,
            "Execution requires the separately mounted project data disk")
    ensure_data_disk(args.policy)
    require(Path(sys.prefix).resolve().is_relative_to(ROOT) and Path(sys.executable).absolute().is_relative_to(ROOT),
            "Python environment escaped data disk")
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TMPDIR", "TORCH_HOME"):
        require(os.environ.get(name) and Path(os.environ[name]).resolve().is_relative_to(ROOT), "Cache/tmp escaped data disk: " + name)
    for name in ("XDG_CACHE_HOME", "PIP_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        require(not os.environ.get(name) or Path(os.environ[name]).resolve().is_relative_to(ROOT), "Active cache escaped data disk: " + name)
    require(all(os.environ.get(name) == "4" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")),
            "Use the reviewed four-thread CPU runtime")
    policy_path = args.policy.resolve()
    policy = load_json(policy_path)
    policy_sha = sha256_file(policy_path)
    require(policy["status"] == "locked_before_rl_gradient_noise" and policy["wall_seconds"] == 1800
            and policy["absolute_deadline_utc"] == DEADLINE and policy["limit"] == 8 and policy["pilot_records"] == 2
            and policy["chunk_elements"] == 65536 and policy["ks"] == [2, 4, 8], "Unreviewed diagnostic dimensions/deadlines")
    require(policy["manifest"] == MANIFEST and policy["manifest_sha256"] == MANIFEST_SHA
            and policy["reference_cache"] == REFERENCE and policy["beta_report"] == BETA_REPORT
            and policy["finite_state"] == FINITE_STATE, "Scientific inputs must be fixed training prefix/reference and completed v2 state")
    require([entry["label"] for entry in policy["models"]] == list(LABELS)
            and type(policy["include_beta0_counterfactual"]) is bool, "Four fixed checkpoint sources required")
    require(policy["model_path"] == "models/Qwen3-VL-2B-Instruct"
            and policy["revision"] == "89644892e4d85e24eaac8bacfd4f463576704203"
            and policy["logp_atol"] == .01 and policy["psd_rtol"] == 1e-9, "Frozen input/numerical settings differ")
    require(all(policy["sources_sha256"].get(name) == digest for name, digest in FROZEN.items()), "Frozen numerical sources were replaced")
    own_sources = {"exploratory/principled_ordering_20261105/" + name for name in
        ("measure_rl_gradient_noise.py", "test_measure_rl_gradient_noise.py", "test_measure_rl_gradient_noise_tensor.py")}
    required_sources = own_sources | {"exploratory/principled_ordering_20261105/calibrate_beta.py",
        "exploratory/principled_ordering_20261105/verify_finite_run.py",
        "src/recipeqa/modeling.py", "src/recipeqa/metrics.py", "ops/wait_and_evaluate.py",
        "exploratory/principled_ordering_20261105/parameter_diagnostic_pipeline.py"} | set(FROZEN)
    require(required_sources <= set(policy["sources_sha256"]), "Execution policy omits required analysis/runtime sources")
    require(project_path(policy["cpu_preflight"]).is_relative_to(HERE), "CPU preflight receipt escaped analysis namespace")
    require(required_artifacts(policy["cpu_preflight"]) <= set(policy["artifacts_sha256"]), "Execution policy omits required input receipts")
    output = project_path(policy["output_directory"])
    require(output.is_relative_to(HERE) and not output.exists(), "Fresh exploratory output namespace required; no resume/overwrite")
    output.mkdir()
    state_path = output / "state.json"
    state = {"status": "cpu_preflight", "started_utc": now(), "policy_sha256": policy_sha,
             "optimizer_constructed": False, "updates": 0, "test_access": False, "confirmation_access": False,
             "rows": 0, "models_completed": [], "post_hoc_exploration": True}
    deadline = None
    absolute = datetime.fromisoformat(DEADLINE.replace("Z", "+00:00"))
    locks = []
    old_signals = {}
    all_results = {}

    def update(**values):
        state.update(values, updated_utc=now())
        write_json(state_path, state)

    def remaining():
        value = (absolute - datetime.now(timezone.utc)).total_seconds()
        if deadline is not None:
            value = min(value, deadline - time.monotonic())
        if value <= 0:
            raise BudgetStop("Gradient diagnosis phase/absolute deadline exhausted; preserve partial evidence, no retry")
        return value

    def immutable():
        remaining()
        require(sha256_file(policy_path) == policy_sha, "Diagnostic policy changed")
        for mapping in (policy["sources_sha256"], policy["artifacts_sha256"]):
            for name, digest in mapping.items():
                require(sha256_file(project_path(name)) == digest, "Bound source/artifact changed: " + name)

    def alarm_stop(number, frame):
        raise BudgetStop("Owned read-only diagnosis reached its phase/absolute alarm deadline")

    try:
        update()
        immutable()
        cpu = load_json(project_path(policy["cpu_preflight"]))
        require(cpu["tests_run"] == 13 and cpu["successful"] is True and cpu["failures"] == cpu["errors"] == cpu["skipped"] == 0
                and cpu["cuda_initialized"] is False and cpu["versions"] == package_versions()
                and all(cpu["source_sha256"].get(name) == policy["sources_sha256"][name] == sha256_file(project_path(name))
                        for name in own_sources), "Installed CPU13 preflight did not fully pass with matching source/runtime and no CUDA")
        prior = load_json(project_path(FINITE_STATE))
        require(prior["status"] == "completed_finite_trials_development_only" and prior["active_child_pid"] is None
                and all(command.get("returncode") == 0 for command in prior["commands"]), "Six-arm v2 must finish successfully before diagnosis")
        process = process_info(prior["pid"])
        require(process is None or process["state"] in ("Z", "X"), "Completed v2 parent must exit first; do not compete for GPU")
        require(sha256_file(project_path(MANIFEST)) == MANIFEST_SHA, "Fixed train manifest changed")
        records256 = read_jsonl(project_path(MANIFEST))[:256]
        records = records256[:8]
        ref_path = project_path(REFERENCE)
        ref_config = load_json(str(ref_path) + ".config.json")
        ref_completion = load_json(str(ref_path) + ".completion.json")
        references = read_jsonl(ref_path)
        validation = validate_reference(references, ref_config, records256)
        require(ref_completion["status"] == "complete_fixed_initial_reference"
                and ref_completion["rows"] == 256 and ref_completion["cache_sha256"] == sha256_file(ref_path)
                and ref_completion["reference_spec_sha256"] == hash_json(ref_config), "Fixed reference cache completion differs")
        require(ref_config["manifest_sha256"] == MANIFEST_SHA and ref_config["versions"] == package_versions()
                and ref_config["pixel_sha256"] == pixel_identities(records256, project_path(policy["image_root"])),
                "Reference manifest/runtime/current image bytes differ")
        for name, digest in ref_config["scoring_source_sha256"].items():
            require(policy["sources_sha256"].get(name) == digest == sha256_file(project_path(name)), "Reference scoring source differs")
        beta_report = load_json(project_path(BETA_REPORT))
        require(beta_report["status"] == "calibrated_positive_beta" and beta_report["exit_code"] == 0
                and beta_report["prefix"] == 64 and beta_report["delta"] == .02
                and beta_report["beta"] == 1.6363096678310884
                and beta_report["input_receipts"]["reference_cache"]["sha256"] == sha256_file(ref_path)
                and beta_report["input_receipts"]["training_manifest"]["sha256"] == MANIFEST_SHA
                and beta_report["reference_weight_identity"] == ref_config["initial_identity"]
                and beta_report["analyzer_source_sha256"] == sha256_file(HERE / "calibrate_beta.py"), "Positive locked beta/reference provenance differs")
        beta = beta_report["beta"]
        require(policy["input"] == ref_config["input"], "Diagnostic preprocessing changed")
        config = InputConfig(**{key: policy["input"][key] for key in
                  ("min_pixels", "max_pixels", "recipe_token_budget", "max_input_tokens", "max_completion_tokens")})
        require(asdict(config) == ref_config["input"], "InputConfig defaults differ")
        model_path = project_path(policy["model_path"])
        image_root = project_path(policy["image_root"])
        require(model_weight_identity(model_path) == ref_config["initial_identity"]["base"], "Current base/processor bytes differ")
        for entry in policy["models"]:
            label, adapter = entry["label"], project_path(entry["adapter"])
            identity = model_weight_identity(adapter, adapter=True)
            require(identity == entry["identity"], "Locked checkpoint actual identity differs: " + label)
            if label == "common_initial":
                require(identity == ref_config["initial_identity"]["adapter"], "Common initial is not the reference model")
            else:
                method = label[:-2]
                expected = HERE / "finite_trials_v2/flow44" / method / "checkpoint-64"
                require(adapter == expected, "Use fixed flow44 step64, never best-dev selection")
                audit = load_json(adapter / "finite_policy_checkpoint_audit.json")
                run_signature = load_json(adapter.parent / "run_signature.json")
                outcome = load_json(adapter.parent / "train_outcome.json")
                verification = load_json(adapter.parent / "independent_verification.json")
                require(audit["global_step"] == 64 and audit["adapter_identity"] == identity
                        and audit["run_signature_sha256"] == run_signature["sha256"] == hash_json(run_signature["payload"])
                        and run_signature["payload"]["initial_identity"] == ref_config["initial_identity"]
                        and run_signature["payload"]["cli"]["method"] == method
                        and run_signature["payload"]["resolved_beta"] == (0 if method == "exact" else beta)
                        and run_signature["payload"]["selected_records_sha256"] == hash_json(records256)
                        and outcome["status"] == "completed" and outcome["global_step"] == 64
                        and outcome["run_signature_sha256"] == run_signature["sha256"]
                        and outcome["final_adapter_identity"]["weights_sha256"] == identity["weights_sha256"]
                        and verification["status"] == "complete_verified_finite_policy_run" and verification["passed"] is True
                        and verification["method"] == method and verification["seed"] == 44
                        and verification["limit"] == 256 and verification["optimizer_steps"] == 64
                        and verification["beta"] == (0 if method == "exact" else beta)
                        and verification["reference"]["sha256"] == sha256_file(ref_path)
                        and verification["run_signature_sha256"] == run_signature["sha256"]
                        and verification["provenance"]["verifier_sha256"] == sha256_file(HERE / "verify_finite_run.py"),
                        "Fixed model is not a verified step64 checkpoint")
        def lock_wait(attempt, blocked):
            remaining()
            update(status="waiting_five_locks", lock_attempts=attempt, blocked_lock=blocked)
        locks, attempts = acquire_lockset([ROOT / "metadata" / (name + ".lock") for name in LOCKS], 120, 2, lock_wait)
        require_gpu_idle()
        phase_started = time.monotonic()
        deadline = phase_started + min(1800, remaining())
        old_signals = {number: signal.signal(number, alarm_stop) for number in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
        signal.setitimer(signal.ITIMER_REAL, remaining())
        import torch
        require(torch.is_grad_enabled(), "Do not run diagnosis inside no_grad/inference mode")
        provenance = {"policy": policy, "versions": package_versions(), "selected_records_sha256": hash_json(records),
                      "selection": "original full-train prefix8; no correctness/confidence filtering; fixed step64 models",
                      "beta": beta, "reference_validation": validation, "frozen_sources": trainer_sources(),
                      "parameter_coordinate_system": "all392 shared trainable LoRA tensors; raw Euclidean preclip/preAdam gradients",
                      "no_parameter_vectors_persisted": True, "phase_wall_seconds_cap": 1800, "absolute_deadline_utc": DEADLINE}
        write_json(output / "configuration.json", provenance)
        processor = load_processor(model_path, policy["revision"], config)
        score_args = SimpleNamespace(microbatch=1, batch_atol=.01, tie_atol=1e-8)
        for model_index, entry in enumerate(policy["models"]):
            immutable()
            setup_start = time.monotonic()
            model = load_model(model_path, policy["revision"], adapter_path=project_path(entry["adapter"]), trainable=True, seed=44)
            model.to("cuda")
            model.eval()
            model.config.use_cache = False
            inventory = trainable_inventory(model)
            require(inventory["trainable_parameters"] == 17432576 and len(inventory["parameters"]) == 392,
                    "Actual trainable LoRA dimensions differ")
            named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
            parameters = tuple(parameter for _, parameter in named)
            require(len(parameters) == 392 and all(parameter.grad is None for parameter in parameters), "Unexpected .grad accumulation/inventory")
            fingerprint_before = parameter_fingerprint(model)
            adapter_before = adapter_fingerprint(model)
            vision_before = frozen_vision_fingerprint(model)
            setup_seconds = time.monotonic() - setup_start
            model_dir = output / entry["label"]
            model_dir.mkdir()
            write_json(model_dir / "model_provenance.json", {"entry": entry, "inventory": inventory,
                "all_parameters_before": fingerprint_before, "adapter_before": adapter_before, "vision_before": vision_before,
                "setup_seconds": setup_seconds, "gradient_mode": "eval/autograd.grad, no optimizer, no .backward accumulation"})
            model_rows, batches, cases, pilot_seconds = [], [], [], []
            mean_sum = torch.zeros(inventory["trainable_parameters"], dtype=torch.float64)
            for index, (record, reference) in enumerate(zip(records, references[:8])):
                immutable()
                torch.cuda.synchronize()
                case_start = time.monotonic()
                update(status="measuring_read_only", model_label=entry["label"], record_index=index)
                prepared, audit = prepare_record(record, processor, image_root, config)
                candidates = candidate_suffixes(processor, prepared["prompt"])
                require(candidates == reference["candidates"] and audit["input_hash"] == reference["input_audit"]["input_hash"],
                        "Current prompt/visual input/canonical EOS differs from reference")
                encoded = encode_messages(processor, prepared["prompt"], load_images(record, image_root)).to(model.device)
                torch.cuda.reset_peak_memory_stats()
                token_logps, checks = score_candidates(model, encoded, candidates, score_args, literal_check=index == 0)
                current = log_policy([sum(values) for values in token_logps])
                require(current["logZ"] <= 1e-4, "Canonical mass exceeds1")
                reward = [score_order(candidate["text"], record["display_perm"], record["target_order"])["exact"] for candidate in candidates]
                require(reward == reference["exact_rewards"] and sum(reward) == 1, "Verifier source target differs")
                utility = [r - beta * (lq - lr) for r, lq, lr in zip(reward, current["logq"], reference["logq"])]
                stable_coefficients = coefficient_covariance(current["q"], utility, 8)["mean_coefficients"]
                frozen_terms = policy_coefficients("exact_kl", current, reward, reference["logq"], beta)
                coefficients = frozen_terms["coefficients"]
                coefficient_error = max(abs(left - right) for left, right in zip(coefficients, stable_coefficients))
                coefficient_scale = max(max(abs(value) for value in coefficients), max(abs(value) for value in stable_coefficients))
                require(coefficient_error <= 1e-9 * coefficient_scale if coefficient_scale else coefficient_error == 0,
                        "Stable utility centering and actual frozen Exact-KL mean coefficients disagree")
                # CPU-only scratch, destroyed after Gram/mean calculation; never torch.save.
                raw = torch.empty((24, inventory["trainable_parameters"]), dtype=parameters[0].dtype, device="cpu")
                require(all(parameter.dtype == parameters[0].dtype for parameter in parameters), "Mixed LoRA dtypes need an explicit new storage protocol")
                candidate_checks = []
                for candidate_index, candidate in enumerate(candidates):
                    remaining()
                    scalar, grad_token_logps = differentiable_logp(model, encoded, candidate["token_ids"])
                    require(len(grad_token_logps) == len(token_logps[candidate_index]) == candidate["token_count"], "Differentiable EOS score length differs")
                    discrepancies = [a - b for a, b in zip(grad_token_logps, token_logps[candidate_index])]
                    maximum = max(abs(value) for value in discrepancies)
                    require(maximum <= .01, "Differentiable/no-grad logp mismatch beyond frozen tolerance")
                    gradients = torch.autograd.grad(scalar, parameters, retain_graph=False, create_graph=False, allow_unused=False)
                    offset = 0
                    for parameter, gradient in zip(parameters, gradients):
                        size = parameter.numel()
                        require(bool(torch.isfinite(gradient).all()), "Nonfinite actual LoRA score gradient")
                        raw[candidate_index, offset:offset + size].copy_(gradient.detach().reshape(-1).cpu())
                        offset += size
                    require(offset == raw.shape[1], "Actual392-tensor flattening differs")
                    candidate_checks.append({"candidate_index": candidate_index, "token_discrepancies_grad_minus_nograd": discrepancies,
                         "maximum_token_discrepancy": maximum, "raw_score_discrepancy": math.fsum(discrepancies),
                         "differentiable_raw_logp": float(scalar.detach()), "gradient_dtype": str(raw.dtype)})
                    del gradients, scalar
                gram, centered_mean, mean, centering = centered_gram_torch(
                    raw, current["q"], coefficients, policy["chunk_elements"], include_raw_mean=True)
                scratch_bytes = raw.numel() * raw.element_size()
                del raw, encoded
                grad_forward_policy = log_policy([check["differentiable_raw_logp"] for check in candidate_checks])
                row = {"model_label": entry["label"], "record_index": index, "qid": record["qid"], "recipe_id": record["recipe_id"],
                       "presentation_id": record["presentation_id"], "source_split": "train", "input_audit": audit,
                       "q": current["q"], "logq": current["logq"], "logZ_canonical": current["logZ"], "qref": reference["q"],
                       "logqref": reference["logq"], "exact_rewards": reward, "beta_locked": beta, "utility": utility,
                       "canonical_candidates": candidates, "token_logps_including_eos": token_logps,
                       "candidate_gradient_forward_checks": candidate_checks, "numerical_reference_checks": checks,
                       "differentiable_forward_policy": grad_forward_policy,
                       "max_logq_discrepancy_grad_vs_nograd": max(abs(a - b) for a, b in zip(grad_forward_policy["logq"], current["logq"])),
                       "centered_parameter_gradient_gram_fp64": gram, "centering": centering,
                       "frozen_analytic_exact_kl_coefficients": coefficients,
                       "stable_vs_frozen_coefficient_max_absolute_error": coefficient_error,
                       "stable_vs_frozen_coefficient_scale": coefficient_scale,
                       "scratch_gradient_matrix_bytes_in_ram_not_persisted": scratch_bytes,
                       "same_theta_beta_qref": True, "objective": "analytic Exact-KL target at this checkpoint, independent of checkpoint training-arm name"}
                write_json(output / "current_record_geometry.json", row)
                relative_mean_error = centering["raw_centered_mean_relative_l2"]
                require(relative_mean_error <= 1e-8 if relative_mean_error is not None else
                        centering["raw_centered_mean_error_norm2"] == 0,
                        "Raw and centered actual mean vectors do not agree in relative L2; preserve residual, no silent mean correction")
                count = validation["canonical_token_count"]
                geometry = {str(k): geometry_from_gram(gram, current["q"], utility, k, count, policy["psd_rtol"]) for k in (2, 4, 8)}
                actual_mean2 = float(torch.dot(mean, mean))
                gram_mean2 = geometry["8"]["ascent_signal_norm2_raw"]
                comparison_scale = max(actual_mean2, gram_mean2)
                require(abs(actual_mean2 - gram_mean2) <= 1e-8 * comparison_scale if comparison_scale else actual_mean2 == gram_mean2,
                        "Actual FP64 mean vector and centered Gram signal norm differ")
                row.update(geometry=geometry, actual_mean_gradient_norm2_raw=actual_mean2,
                           actual_loss_mean_gradient_norm2=actual_mean2 / count ** 2)
                if policy["include_beta0_counterfactual"]:
                    row["beta0_counterfactual_explicitly_not_primary"] = {str(k): geometry_from_gram(
                        gram, current["q"], reward, k, count, policy["psd_rtol"]) for k in (2, 4, 8)}
                mean_sum += mean
                del mean, centered_mean
                cases.append(geometry)
                if (index + 1) % 4 == 0:
                    batch = batch_summary(float(torch.dot(mean_sum, mean_sum)), cases, policy["psd_rtol"])
                    batch.update(record_indices=list(range(index - 3, index + 1)), beta=beta,
                                 reference_cache_sha256=sha256_file(ref_path), model_label=entry["label"])
                    batches.append(batch)
                    mean_sum.zero_()
                    cases = []
                require(all(parameter.grad is None for parameter in parameters), "autograd.grad unexpectedly wrote .grad")
                require(adapter_fingerprint(model) == adapter_before, "Read-only diagnostic changed actual adapter")
                torch.cuda.synchronize()
                row.update(wall_seconds=time.monotonic() - case_start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(), parameter_vectors_persisted=False, completed_utc=now())
                with (model_dir / "measurements.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                model_rows.append({"qid": row["qid"], "record_index": index, "wall_seconds": row["wall_seconds"]})
                update(rows=state["rows"] + 1)
                print(json.dumps({"model": entry["label"], "record": index + 1, "seconds": row["wall_seconds"]}), flush=True)
                if model_index == 0 and index < 2:
                    pilot_seconds.append(row["wall_seconds"])
                if model_index == 0 and index == 1:
                    estimate = 1.25 * (30 * max(pilot_seconds) + 7 * setup_seconds)
                    gate = {"status": "resource_only_pilot_gate", "records": [0, 1], "pilot_seconds": pilot_seconds,
                            "setup_seconds": setup_seconds, "estimated_remaining_seconds": estimate,
                            "remaining_seconds": remaining(), "passed": estimate <= remaining(),
                            "rule": "1.25*(30*max first2 case wall +7*initial model setup); includes later3 setups and4 fingerprint/cleanup allowances",
                            "score_based_selection": False, "is_runtime_guarantee": False}
                    write_json(output / "pilot_timing_gate.json", gate)
                    if not gate["passed"]:
                        raise BudgetStop("All32 fixed diagnostic cases do not fit; no model/question subset selection")
                del row, gram, geometry
            remaining()
            fingerprint_after = parameter_fingerprint(model)
            vision_after = frozen_vision_fingerprint(model)
            require(fingerprint_after == fingerprint_before and vision_after == vision_before
                    and adapter_fingerprint(model) == adapter_before, "Model or frozen visual parameters changed")
            require(len(batches) == 2 and not cases, "Fixed8 prompts did not make two full accumulation batches")
            completion = {"status": "complete_read_only_model_gradient_noise", "label": entry["label"], "rows": 8,
                "batches": batches, "measurements": file_receipt(model_dir / "measurements.jsonl"),
                "all_parameters_before": fingerprint_before, "all_parameters_after": fingerprint_after,
                "vision_before": vision_before, "vision_after": vision_after, "weights_unchanged": True,
                "optimizer_constructed": False, "updates": 0, "parameter_vectors_persisted": False, "case_times": model_rows}
            write_json(model_dir / "completion.json", completion)
            all_results[entry["label"]] = completion
            update(models_completed=list(all_results))
            del model, parameters, named, mean_sum
            torch.cuda.empty_cache()
        immutable()
        require(model_weight_identity(model_path) == ref_config["initial_identity"]["base"], "Base/processor files changed during diagnosis")
        for entry in policy["models"]:
            require(model_weight_identity(project_path(entry["adapter"]), adapter=True) == entry["identity"], "Checkpoint bytes changed during diagnosis")
        summary = {"status": "complete_read_only_rl_gradient_noise", "models": all_results, "rows": 32,
                   "beta": beta, "policy_sha256": policy_sha, "wall_seconds": time.monotonic() - phase_started,
                   "optimizer_constructed": False, "updates": 0, "test_access": False, "confirmation_access": False,
                   "parameter_vectors_persisted": False, "limits": ["Fixed8 training questions, not a prevalence estimate",
                     "Same theta/beta/qref conditional analysis; training-arm names do not set diagnostic beta",
                     "Raw Euclidean preclip/preAdam noise, not update noise or evidence of performance/SOTA",
                     "No-grad versus differentiable score consistency has the explicitly recorded frozen numerical tolerance",
                     "No model-level Monte Carlo sampler check was executed; exact covariance and independent pairwise trace are checked",
                     "Scratch matrix bytes are recorded; process host-RAM peak is not measured"]}
        remaining()
        write_json(output / "summary.json", summary)
        update(status="complete_read_only_rl_gradient_noise", finished_utc=now(), summary=file_receipt(output / "summary.json"))
    except BaseException as error:
        failure = {"status": "stopped_budget_no_retry" if isinstance(error, BudgetStop) else "failed_stopped_no_retry",
                   "error": str(error), "error_type": type(error).__name__, "finished_utc": now(),
                   "numerical_diagnostics": getattr(error, "diagnostics", None)}
        update(**failure)
        raise
    finally:
        if old_signals:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for number, handler in old_signals.items():
                signal.signal(number, handler)
        release_locks(locks)


if __name__ == "__main__":
    main()
