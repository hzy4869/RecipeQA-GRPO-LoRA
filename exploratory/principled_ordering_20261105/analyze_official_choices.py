#!/usr/bin/env python3
"""CPU-only official-four-choice bridge from complete fixed-val200 likelihoods.

No new inference, training, method/checkpoint selection or test/confirmation
access. This restricts existing candidate likelihoods; the model was not newly
prompted with the four options. It is not an official test or SOTA result.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VAL_SOURCE_SHA = "cca6c2c7a3f5516251a8eb86e6c26548a56cebff871a8fba330b51bc788c4e04"
DEV_SHA = "c998796fbc07a24c7a24062d6d0aba07500c75b34568fd7ba67f8c38daae03c9"
SCORER_SHA = "affeda80416145d2b6a134172e812d15d48a86d77c67128bd2e5f84e0c341819"
METRIC_SHA = "5287d44a3bb4ede3a44ce0a6625aee399260afecd42c984441a8e07492a2423a"
CANONICALS = tuple(" ".join(order) for order in permutations("ABCD"))
TIE_ATOL = 1e-8
sys.path.insert(0, str(ROOT / "src"))
from recipeqa.metrics import score_order


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def receipt(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def no_test_or_confirmation(path):
    for component in Path(path).parts:
        lowered = component.lower()
        require(not (lowered == "test" or lowered.startswith("test.") or lowered.startswith("test_")
                    or "confirmation" in lowered or lowered.startswith("confirm")),
                "Test/confirmation paths are forbidden: " + str(path))


def audit_source(rows):
    """Reconstruct source chronology independently, including duplicate options."""
    result = {}
    for source_index, row in enumerate(rows):
        if row.get("task") != "visual_ordering":
            continue
        qid, recipe = str(row["qid"]), str(row["recipe_id"])
        require(qid not in result, "Duplicate source question")
        choices, answer = row["choice_list"], row["answer"]
        require(isinstance(choices, list) and len(choices) == 4 and type(answer) is int and 0 <= answer < 4,
                "Expected four official options and integer gold index")
        require(all(isinstance(option, list) and len(option) == 4 and len(set(option)) == 4
                    and all(isinstance(name, str) and Path(name).name == name and "\\" not in name for name in option)
                    for option in choices), "Source option has malformed/repeated photos")
        names = set(choices[0])
        require(all(set(option) == names for option in choices), "Options use different photo sets")
        steps = {name: int(Path(name).stem.rsplit("_", 2)[1]) for name in names}
        require(len(set(steps.values())) == 4, "Expected photos from four distinct source steps")
        chronological = sorted(names, key=steps.get)
        semantic = [[chronological.index(name) for name in option] for option in choices]
        require(all(sorted(option) == [0, 1, 2, 3] for option in semantic), "Source option is not a permutation")
        correct_indices = [index for index, option in enumerate(semantic) if option == [0, 1, 2, 3]]
        require(correct_indices == [answer], "Gold is not the unique chronological source option")
        context = [int(step["id"]) for step in row["context"]]
        require(context == sorted(set(context)) and set(steps.values()) <= set(context),
                "Recipe context order or source-step coverage failed")
        incorrect = [tuple(option) for index, option in enumerate(semantic) if index != answer]
        result[qid] = {"qid": qid, "recipe_id": recipe, "source_row_index": source_index,
                       "source_ordering_index": len(result), "answer_index": answer,
                       "images_chronological": chronological, "source_step_ids_chronological": [steps[name] for name in chronological],
                       "choice_images": choices, "choice_semantic_permutations": semantic,
                       "duplicate_incorrect_options": len(incorrect) - len(set(incorrect)),
                       "backend_only_not_model_visible": True}
    return result


def validate_manifest(rows, source):
    require(len(rows) == len({str(row["qid"]) for row in rows}) ==
            len({str(row["recipe_id"]) for row in rows}) == 200, "Require complete200 distinct questions/recipes")
    for row in rows:
        require(row.get("source_split") == row.get("manifest_set") == "val", "Only original val rows allowed")
        original = source[str(row["qid"])]
        display = row["display_perm"]
        require(sorted(display) == [0, 1, 2, 3] and all(type(value) is int for value in display), "Bad display permutation")
        require(str(row["recipe_id"]) == original["recipe_id"]
                and row["images_chronological"] == original["images_chronological"]
                and row["source_step_ids_chronological"] == original["source_step_ids_chronological"],
                "Source question/photos differ from fixed development manifest")
        require(row.get("target_order") == [0, 1, 2, 3]
                and row["target_labels"] == [chr(65 + display.index(i)) for i in range(4)], "Manifest gold/display mismatch")


def choose(scores, texts, namespace, tag):
    """Tie decision accepts no gold argument; original wrong duplicates retained."""
    require(len(scores) == len(texts) and bool(scores), "Empty/misaligned decision set")
    require(all(type(value) in (float, int) and math.isfinite(value) for value in scores), "Nonfinite score")
    maximum = max(scores)
    tied = [index for index, value in enumerate(scores) if maximum - value <= TIE_ATOL]
    if tag == "canonical-map-tie-v1":
        winner = min(tied, key=lambda index: digest([tag, namespace, texts[index]]))
    else:
        winner = min(tied, key=lambda index: digest([tag, namespace, texts[index], index]))
    return winner, tied


def validate_score_row(row, manifest, config, config_sha, identity_sha):
    require(row.get("source_split") == row.get("manifest_set") == "val" and not row.get("technical_error"),
            "Only successfully scored val rows allowed")
    require(row["configuration_sha256"] == config_sha and row["model_weights_sha256"] == identity_sha,
            "Score row configuration or model identity differs")
    for field in ("qid", "recipe_id", "presentation_id", "display_perm", "target_order", "target_labels"):
        require(row[field] == manifest[field], "Score row differs from fixed manifest: " + field)
    require(row["run_id"] == config["cli"]["run_id"], "Score row run ID differs")
    candidates = row["candidates"]
    require(len(candidates) == 24 and tuple(item["text"] for item in candidates) == CANONICALS,
            "Require all24 canonical candidate answers exactly once, in declared order")
    values = []
    for item in candidates:
        tokens, token_logps = item["token_ids"], item["token_logps_including_eos"]
        require(len(tokens) == len(token_logps) == item["token_count"] and len(tokens) >= 2
                and tokens[-1] == item["eos_token_id"], "Candidate lacks complete EOS token likelihood")
        require(all(type(token) is int for token in tokens)
                and all(type(value) in (float, int) and math.isfinite(value) and value <= 1e-6 for value in token_logps),
                "Malformed token ID/log probability")
        total = sum(token_logps)
        require(math.isclose(total, item["logp_including_eos"], abs_tol=1e-9, rel_tol=1e-10)
                and math.isclose(sum(token_logps[:-1]), item["logp_without_eos_diagnostic"], abs_tol=1e-9, rel_tol=1e-10),
                "Stored sequence probability differs from complete token sum")
        values.append(total)
    maximum = max(values)
    require(maximum + math.log(sum(math.exp(value - maximum) for value in values)) <= 1e-4,
            "Disjoint canonical complete-string mass exceeds one")
    require(isinstance(row["input_audit"]["prompt_text_hash"], str) and len(row["input_audit"]["prompt_text_hash"]) == 64,
            "Missing target-independent prompt tie namespace")
    namespace = row["input_audit"]["prompt_text_hash"]
    map_index, _ = choose(values, CANONICALS, namespace, "canonical-map-tie-v1")
    require(row["canonical_map"]["map_text"] == CANONICALS[map_index]
            and row["canonical_map"]["map_candidate_index"] == map_index, "Independent24MAP disagrees")
    map_score = score_order(CANONICALS[map_index], row["display_perm"], row["target_order"])
    require(map_score == row["canonical_map_score"], "Stored24MAP score disagrees with raw rescore")
    greedy = row["greedy"]
    require(isinstance(greedy["raw_response"], str), "Missing raw greedy answer")
    greedy_score = score_order(greedy["raw_response"], row["display_perm"], row["target_order"])
    require(greedy_score == greedy["score"], "Stored greedy score disagrees with raw rescore")
    return values, map_index, map_score, greedy_score


def analyze_row(row, source_item, values, map_index, map_score, greedy_score):
    display = row["display_perm"]
    option_texts = [" ".join(chr(65 + display.index(semantic)) for semantic in option)
                    for option in source_item["choice_semantic_permutations"]]
    lookup = dict(zip(CANONICALS, values))
    choice_logps = [lookup[text] for text in option_texts]
    choice, tied = choose(choice_logps, option_texts, row["input_audit"]["prompt_text_hash"], "official-choice-map-tie-v1")
    # Gold is used only after the label-independent decision above.
    gold = source_item["answer_index"]
    require(option_texts[gold] == " ".join(row["target_labels"]), "Official gold label mapping disagrees")
    return {"run_id": row["run_id"], "qid": row["qid"], "recipe_id": row["recipe_id"],
            "presentation_id": row["presentation_id"], "source_split": "val", "manifest_set": "val",
            "display_perm": display, "official_option_texts_current_display": option_texts,
            "official_option_complete_logps": choice_logps, "predicted_official_choice_index": choice,
            "official_tied_choice_indices": tied, "official_gold_choice_index": gold,
            "four_choice_exact": int(choice == gold), "four_choice_prediction": option_texts[choice],
            "canonical24_map_prediction": CANONICALS[map_index], "canonical24_map_score": map_score,
            "greedy_raw_response": row["greedy"]["raw_response"], "greedy_score": greedy_score,
            "duplicate_incorrect_options": source_item["duplicate_incorrect_options"],
            "model_weights_sha256": row["model_weights_sha256"], "configuration_sha256": row["configuration_sha256"],
            "selection_or_training": False}


def aggregate(rows):
    if not rows:
        return {"questions": 0, "recipes": 0, "four_choice_question_micro": None, "four_choice_recipe_macro": None,
                "canonical24_map_question_micro": None, "canonical24_map_recipe_macro": None,
                "greedy_question_micro": None, "greedy_recipe_macro": None}
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["recipe_id"])].append(row)
    getters = {"four_choice": lambda row: row["four_choice_exact"],
               "canonical24_map": lambda row: row["canonical24_map_score"]["exact"],
               "greedy": lambda row: row["greedy_score"]["exact"]}
    result = {"questions": len(rows), "recipes": len(grouped)}
    for name, getter in getters.items():
        result[name + "_question_micro"] = sum(map(getter, rows)) / len(rows)
        result[name + "_recipe_macro"] = sum(sum(map(getter, group)) / len(group) for group in grouped.values()) / len(grouped)
    result["greedy_valid_question_micro"] = sum(row["greedy_score"]["valid"] for row in rows) / len(rows)
    return result


def analyze_file(path, manifest, source):
    config_path = path.with_suffix(path.suffix + ".config.json")
    completion_path = path.with_suffix(path.suffix + ".completion.json")
    config, completion, rows = load(config_path), load(completion_path), read_jsonl(path)
    require(config["manifest_sha256"] == DEV_SHA and config["cli"]["limit"] == 200,
            "Only the full fixed val200 diagnostic can be analyzed")
    require(tuple(config["candidate_texts"]) == CANONICALS and config.get("post_hoc_exploration") is True,
            "Configuration does not declare all24 canonical actions")
    require(config["cli"]["tie_atol"] == TIE_ATOL and config["cli"]["microbatch"] == 1,
            "Analysis accepts only locked microbatch1/tie rule")
    require(config["source_sha256"]["exploratory/principled_ordering_20261105/score_permutations.py"] == SCORER_SHA
            and config["source_sha256"]["src/recipeqa/metrics.py"] == METRIC_SHA,
            "Scorer/metric provenance is not the reviewed diagnostic")
    require(config["input"] == {"min_pixels": 65536, "max_pixels": 262144, "recipe_token_budget": 1536,
                                "max_input_tokens": 4096, "max_completion_tokens": 32, "prompt_version": "recipeqa-order-v1"},
            "Diagnostic input budget differs")
    config_sha, identity_sha = digest(config), digest(config["weight_identity"])
    require(config["model_weights_sha256"] == identity_sha, "Configuration weight identity hash disagrees")
    require(config["selected_rows_sha256"] == digest(manifest), "Configuration uses another200-row manifest content")
    require(completion["status"] == "complete" and completion["rows"] == completion["expected_rows"] == 200
            and completion["configuration_sha256"] == config_sha and completion["output_sha256"] == sha(path),
            "Completion receipt does not bind this complete raw output")
    require(len(rows) == 200 and len({str(row["qid"]) for row in rows}) == 200,
            "Incomplete or duplicate likelihood rows")
    expected = {str(row["qid"]): row for row in manifest}
    require({str(row["qid"]) for row in rows} == set(expected), "Likelihood question set differs from fixed val200")
    predictions = []
    for row in rows:
        values, map_index, map_score, greedy_score = validate_score_row(
            row, expected[str(row["qid"])], config, config_sha, identity_sha)
        predictions.append(analyze_row(row, source[str(row["qid"])], values, map_index, map_score, greedy_score))
    metrics = aggregate(predictions)
    strata = {name: aggregate([row for row in predictions if bool(row["duplicate_incorrect_options"]) == flag])
              for name, flag in (("no_duplicate_incorrect_options", False), ("duplicate_incorrect_options", True))}
    summary = {"run_id": config["cli"]["run_id"], "metrics": metrics, "duplicate_incorrect_options_strata": strata,
               "source_receipts": {"scores": receipt(path), "configuration": receipt(config_path), "completion": receipt(completion_path)},
               "weight_identity": config["weight_identity"], "model_weights_sha256": identity_sha,
               "configuration_sha256": config_sha, "scorer_source_sha256": config["source_sha256"],
               "all_rows_retained": True, "method_selection": False}
    return predictions, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/val.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/val.jsonl")
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "official_choice_analysis")
    parser.add_argument("--check-only", action="store_true", help="Perform full CPU validation and print summaries without writing")
    args = parser.parse_args()
    source_path, manifest_path = args.source.resolve(), args.manifest.resolve()
    for path in [source_path, manifest_path, *args.scores]:
        no_test_or_confirmation(path)
    require(source_path.name == "val.json" and sha(source_path) == VAL_SOURCE_SHA, "Only exact original val.json is accepted")
    require(sha(manifest_path) == DEV_SHA, "Only the fixed original val200 manifest is accepted")
    require(sha(ROOT / "src/recipeqa/metrics.py") == METRIC_SHA, "Frozen metric source changed")
    raw = load(source_path)
    source = audit_source(raw["data"])
    require(len(source) == 929 and len({row["recipe_id"] for row in source.values()}) == 630,
            "Official validation ordering source denominator changed")
    manifest = read_jsonl(manifest_path)
    validate_manifest(manifest, source)
    output = args.output_dir.resolve()
    all_predictions, summaries, run_ids = [], [], set()
    for path in args.scores:
        predictions, summary = analyze_file(path.resolve(), manifest, source)
        require(summary["run_id"] not in run_ids, "Duplicate diagnostic model/run input")
        run_ids.add(summary["run_id"])
        all_predictions.extend(predictions)
        summaries.append(summary)
    report = {"status": "verified_development_four_choice_bridge", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_receipt": receipt(source_path), "fixed_manifest_receipt": receipt(manifest_path),
              "source_questions_audited": len(source), "evaluated_questions_per_model": 200,
              "source_duplicate_incorrect_questions": sum(bool(row["duplicate_incorrect_options"]) for row in source.values()),
              "models": summaries, "test_access": False, "confirmation_access": False, "new_gpu_inference": False,
              "training_or_method_selection": False,
              "tie_rule": "Maximum complete EOS logp within1e-8; target-independent SHA256(prompt namespace,text,option index)",
              "interpretation": "Candidate-set restriction of existing val likelihoods; original four options were NOT newly shown in model prompt",
              "comparison_boundary": "Development-only supplementary bridge; not original905question test or an official SOTA claim",
              "macro_note": "This val200 has one question per recipe, so question micro and recipe macro coincide",
              "analyzer_source_sha256": sha(__file__), "metric_source_sha256": METRIC_SHA}
    if args.check_only:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return
    require(not output.exists(), "Output exists; never overwrite or automatically resume")
    require(ROOT.is_relative_to("/root/autodl-tmp") and output.is_relative_to(ROOT)
            and ROOT.stat().st_dev != Path("/").stat().st_dev, "All remote artifacts must stay on the mounted project data disk")
    output.mkdir()
    write_jsonl(output / "predictions.jsonl", all_predictions)
    write_jsonl(output / "official_validation_choice_audit.jsonl", source.values())
    report["output_receipts"] = {"predictions": receipt(output / "predictions.jsonl"),
                                 "source_choice_audit": receipt(output / "official_validation_choice_audit.jsonl")}
    write_json(output / "summary.json", report)
    print(json.dumps({"status": report["status"], "output_dir": str(output),
                      "models": [{"run_id": item["run_id"], **item["metrics"]} for item in summaries]}, indent=2))


if __name__ == "__main__":
    main()
