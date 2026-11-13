"""Independent stdlib mathematics for fixed endpoint likelihood diagnostics.

No producer scorer, optimizer, model, GPU, network or filesystem import.  The
gold-dependent analytic policy is a diagnostic reference, never a predictor or
an upper bound on held-out accuracy.
"""
from itertools import permutations
import math

BETA = 1.6363096678310884
ACTIONS = tuple(" ".join(x) for x in permutations("ABCD"))
TIE_POLICY = "exact_maximum_then_lowest_canonical_index"
NEAR_TIE_NATS = 1e-9


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(actual, expected, label, atol=1e-9):
    require(finite_number(actual) and finite_number(expected)
            and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=atol),
            "Numeric mismatch: " + label)
    return abs(actual - expected)


def four_permutation(values):
    require(isinstance(values, (list, tuple)) and len(values) == 4
            and all(type(x) is int for x in values)
            and set(values) == set(range(4)), "Malformed four-permutation")
    return tuple(values)


def target_index(record):
    display = four_permutation(record["display_perm"])
    target = four_permutation(record["target_order"])
    labels = [chr(65 + display.index(item)) for item in target]
    require(record["target_labels"] == labels, "Gold labels disagree with photo coordinates")
    return ACTIONS.index(" ".join(labels))


def reconstruct(token_logps, candidates):
    """Sum complete EOS-inclusive token logps, then normalize 24 strings."""
    require(len(token_logps) == len(candidates) == 24, "Complete 24 candidates required")
    require(tuple(x["text"] for x in candidates) == ACTIONS,
            "Canonical support/order changed")
    require(len({x["eos_token_id"] for x in candidates}) == 1,
            "Candidate EOS identities differ")
    scores = []
    for values, candidate in zip(token_logps, candidates):
        ids = candidate["token_ids"]
        require(len(values) == len(ids) == candidate["token_count"] == 5
                and all(type(x) is int and x >= 0 for x in ids)
                and ids[-1] == candidate["eos_token_id"], "Full C5 including EOS required")
        require(all(finite_number(x) and x <= 1e-6 for x in values),
                "Invalid token log probability")
        scores.append(math.fsum(values))
    high = max(scores)
    logz = high + math.log(math.fsum(math.exp(x - high) for x in scores))
    require(math.isfinite(logz) and logz <= 1e-4,
            "Complete-string canonical mass exceeds one")
    logq = [x - logz for x in scores]
    q = [math.exp(x) for x in logq]
    close(math.fsum(q), 1., "q24 normalization")
    return {"scores": scores, "logq": logq, "q": q, "logZ": logz,
            "canonical_mass": math.exp(logz),
            "entropy_nats": -math.fsum(p * l for p, l in zip(q, logq)), "C": 5}


def map_details(log_scores, gold):
    require(len(log_scores) == 24 and all(finite_number(x) for x in log_scores)
            and type(gold) is int and 0 <= gold < 24, "Invalid MAP support/gold")
    maximum = max(log_scores)
    tied = [i for i, x in enumerate(log_scores) if x == maximum]
    chosen = tied[0]  # Exact equality only: near-tie reporting does not select.
    higher = sum(x > log_scores[gold] for x in log_scores)
    equal = sum(x == log_scores[gold] for x in log_scores)
    preceding_ties = sum(x == log_scores[gold] for x in log_scores[:gold])
    competitor = max(x for i, x in enumerate(log_scores) if i != gold)
    return {"chosen_index": chosen, "map_exact": int(chosen == gold),
            "tied_max_indices": tied, "tied_max_count": len(tied),
            "gold_in_max_set": gold in tied,
            "near_max_count_at_1e_minus9_nats": sum(maximum - x <= NEAR_TIE_NATS for x in log_scores),
            "gold_rank_tie_broken": higher + preceding_ties + 1,
            "gold_best_rank": higher + 1, "gold_worst_rank": higher + equal,
            "gold_log_margin_nats": log_scores[gold] - competitor}


def analytic_policy(reference, gold):
    """Unrestricted per-question maximizer of q_gold-beta KL(q||qref)."""
    logz = math.log1p(reference["q"][gold] * math.expm1(1 / BETA))
    logq = [l + (int(i == gold) / BETA) - logz
            for i, l in enumerate(reference["logq"])]
    q = [math.exp(l) for l in logq]
    close(math.fsum(q), 1., "analytic qstar normalization")
    return {"q": q, "logq": logq, "log_reward_normalizer": logz,
            "Jstar": BETA * logz, "odds_multiplier": math.exp(1 / BETA),
            "map": map_details(logq, gold),
            "gold_dependent_not_deployable": True,
            "not_heldout_accuracy_upper_bound": True}


def metrics(current, reference, gold):
    kl = math.fsum(p * (l - r) for p, l, r in
                   zip(current["q"], current["logq"], reference["logq"]))
    require(math.isfinite(kl) and kl >= -1e-9, "Invalid current-to-reference KL")
    qstar = analytic_policy(reference, gold)
    kl_star = math.fsum(p * (l - s) for p, l, s in
                        zip(current["q"], current["logq"], qstar["logq"]))
    j = current["q"][gold] - BETA * kl
    gap = qstar["Jstar"] - j
    residual = close(gap, BETA * kl_star, "Jstar-J=beta KL(q||qstar)")
    require(kl_star >= -1e-9 and gap >= -1e-9, "Negative analytic objective gap")
    initial_map = map_details(reference["logq"], gold)
    group = ("initial_MAP_correct" if initial_map["map_exact"] else
             "initial_wrong_qstar_MAP_correct" if qstar["map"]["map_exact"] else
             "initial_wrong_qstar_MAP_wrong")
    result = {"gold_index": gold, "qgold": current["q"][gold],
              "qref_gold": reference["q"][gold], "KL_current_reference": kl,
              "common_beta_J": j, "Jstar": qstar["Jstar"],
              "Jref": reference["q"][gold],
              "available_objective_gain": qstar["Jstar"] - reference["q"][gold],
              "achieved_objective_gain": j - reference["q"][gold],
              "Jstar_minus_J": gap, "KL_current_qstar": kl_star,
              "gap_identity_residual": residual,
              "raw_sequence_NLL": -current["scores"][gold],
              "raw_sequence_CE_div_C": -current["scores"][gold] / 5,
              "conditional_NLL": -current["logq"][gold],
              "log_canonical_mass": current["logZ"],
              "canonical_mass": current["canonical_mass"],
              "entropy_nats": current["entropy_nats"],
              "map": map_details(current["logq"], gold),
              "initial_map": initial_map, "qstar_map": qstar["map"],
              "qstar_gold_probability": qstar["q"][gold],
              "stratum": group, "tie_policy": TIE_POLICY,
              "qstar_gold_dependent_not_deployable": True,
              "qstar_MAP_not_accuracy_upper_bound": True}
    return result


def score_greedy(text, record):
    """Existing open-decoder answer only, independent photo-coordinate score."""
    if not isinstance(text, str):
        return {"valid": 0, "exact": 0, "semantic_order": None}
    candidate = text.strip()
    if candidate[:6].casefold() == "answer":
        candidate = candidate[6:].lstrip()
        if candidate.startswith(":"):
            candidate = candidate[1:].lstrip()
    legal = (bool(candidate) and candidate[0].upper() in "ABCD"
             and candidate[-1].upper() in "ABCD"
             and all(c.upper() in "ABCD" or c == "," or c.isspace() for c in candidate))
    letters = [c.upper() for c in candidate if c.upper() in "ABCD"] if legal else []
    if len(letters) != 4 or set(letters) != set("ABCD"):
        return {"valid": 0, "exact": 0, "semantic_order": None}
    semantic = [record["display_perm"][ord(c) - 65] for c in letters]
    return {"valid": 1, "exact": int(semantic == record["target_order"]),
            "semantic_order": semantic}
