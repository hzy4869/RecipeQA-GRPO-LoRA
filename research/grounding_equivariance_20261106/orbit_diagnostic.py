#!/usr/bin/env python3
"""New bounded, read-only four-endpoint display-orbit coordinator.

No SSH, optimizer, updates, model implementation, checkpoint selection, or held-out
access. This file calls the frozen predictor and coverage gate as subprocesses.
DRAFT policies cannot execute. An existing output directory cannot be resumed.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from itertools import permutations, combinations
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import threading

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recipeqa.modeling import hash_json, model_weight_identity, read_jsonl, sha256_file, write_json

PERMS = tuple(permutations(range(4)))
ARM_IDS = ("flow44_raw_sft", "flow44_exact_kl", "flow45_raw_sft", "flow45_exact_kl")
BASE = "/root/autodl-tmp/rl_recipeqa"
DEV_SHA = "c998796fbc07a24c7a24062d6d0aba07500c75b34568fd7ba67f8c38daae03c9"
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
LOCK_NAMES = ("formal_training_pipeline", "evaluation_pipeline", "evaluation_handoff", "seed43_pipeline", "reward_search_20261105", "finite_trials_20261105", "orbit_diagnostic_20261106")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def remaining_budget(phase_deadline, absolute_deadline, *, monotonic_now=None, utc_now=None):
    monotonic_now = time.monotonic() if monotonic_now is None else monotonic_now
    utc_now = datetime.now(timezone.utc) if utc_now is None else utc_now
    require(absolute_deadline.tzinfo is not None and utc_now.tzinfo is not None, 'UTC-aware deadline required')
    seconds = min(phase_deadline - monotonic_now, (absolute_deadline - utc_now).total_seconds())
    require(seconds > 0, 'New two-hour phase or absolute deadline reached; preserve partial evidence, no retry')
    return seconds


def install_termination_handlers():
    """Route normal coordinator termination through owned-child cleanup.

    SIGKILL cannot be handled. A second normal termination is ignored while the
    first exception performs bounded cleanup; prior handlers are restored last.
    """
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGHUP)}
    def terminate(number, frame):
        for selected in previous:
            signal.signal(selected, signal.SIG_IGN)
        raise InterruptedError('Coordinator received termination signal ' + str(number))
    for number in previous:
        signal.signal(number, terminate)
    return previous


def restore_termination_handlers(previous):
    for number, handler in previous.items():
        signal.signal(number, handler)


def install_budget_alarm(seconds):
    """A real wall alarm also interrupts CPU hashing/preflight/summary work."""
    require(seconds > 0 and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'Require positive budget and no pre-existing real timer')
    previous = signal.getsignal(signal.SIGALRM)
    def expired(number, frame):
        raise TimeoutError('Frozen research-work deadline expired; only bounded owned-child teardown may follow')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    return previous


def cancel_budget_alarm(previous):
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous)


def utc():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def perm(value):
    require(isinstance(value, (list, tuple)) and len(value) == 4 and all(type(v) is int for v in value)
            and set(value) == set(range(4)), "Malformed four-permutation")
    return tuple(value)


def make_orbit(source_rows):
    """First16 fixed rows, all24 layouts: display position -> semantic image ID.

    Derive semantic image identities from the original actual display mapping;
    independently cross-check chronological paths/statuses when provided.
    target_order is preserved, so this supports nonidentity gold as well.
    """
    require(len(source_rows) == 200, "Expected fixed development200, no alternate cohort")
    chosen = source_rows[:16]
    require(len({str(r["qid"]) for r in chosen}) == len({str(r["recipe_id"]) for r in chosen}) == 16,
            "First16 must contain16 distinct questions and recipes")
    output = []
    for original in chosen:
        require(original.get("source_split") == "val" and original.get("manifest_set") == "val", "Only fixed dev input allowed")
        display, target = perm(original["display_perm"]), perm(original.get("target_order", [0, 1, 2, 3]))
        require(original["target_labels"] == [chr(65 + display.index(s)) for s in target], "Original gold inverse differs")
        paths = original["image_paths_display"]
        require(len(paths) == 4 and len(set(paths)) == 4, "Four distinct source photographs required")
        by_semantic = {s: path for s, path in zip(display, paths)}
        chronological = [by_semantic[s] for s in range(4)]
        if "image_paths_chronological" in original:
            require(original["image_paths_chronological"] == chronological, "Original display/chronological image mapping differs")
        statuses = original.get("image_status_display")
        status_by_semantic = {s: status for s, status in zip(display, statuses)} if statuses else None
        if statuses:
            require(len(statuses) == 4, "Four image statuses required")
            if "image_status_chronological" in original:
                require(original["image_status_chronological"] == [status_by_semantic[s] for s in range(4)], "Status identity mapping differs")
        for layout in PERMS:
            row = deepcopy(original)
            row.update(presentation_id=f"{original['qid']}::p{''.join(map(str, layout))}",
                       display_perm=list(layout), target_order=list(target),
                       target_labels=[chr(65 + layout.index(s)) for s in target],
                       image_paths_display=[by_semantic[s] for s in layout],
                       image_paths_chronological=chronological,
                       orbit_origin_presentation_id=original["presentation_id"],
                       orbit_origin_display_perm=list(display))
            if statuses:
                row["image_status_display"] = [status_by_semantic[s] for s in layout]
            # Gold display answer must map back to the unchanged semantic target.
            require(tuple(layout[ord(c)-65] for c in row["target_labels"]) == target, "Generated inverse gold does not round-trip")
            require(all(row["image_paths_display"][p] == chronological[s] for p, s in enumerate(layout)), "Generated image identity changed")
            output.append(row)
    require(len(output) == 384 and len({(r['qid'], r['presentation_id']) for r in output}) == 384, "Orbit IDs are not one-to-one")
    return output


def independent_score(text, display, target):
    """Separate whole-response parser and integer six-relation scorer.

    Match the frozen accepted grammar; never trust predictor's saved scores.
    """
    display, target = perm(display), perm(target)
    candidate = re.sub(r"^answer\s*:?\s*", "", text.strip(), count=1, flags=re.I) if isinstance(text, str) else ""
    valid_syntax = re.fullmatch(r"[A-D](?:[\s,]*[A-D]){3}", candidate, flags=re.I)
    letters = re.sub(r"[\s,]", "", candidate).upper() if valid_syntax else ""
    semantic = tuple(display[ord(c)-65] for c in letters) if len(set(letters)) == 4 else None
    correct = sum(semantic.index(target[a]) < semantic.index(target[b]) for a, b in combinations(range(4), 2)) if semantic is not None else 0
    return {"semantic_order": list(semantic) if semantic is not None else None,
            "valid": int(semantic is not None), "exact": int(semantic == target), "pair_count": correct, "pair": correct / 6}


def rescore(manifest, raw, expected_weights_sha):
    expected = {(str(r["qid"]), str(r["presentation_id"])): r for r in manifest}
    seen, scored = set(), []
    for prediction in raw:
        key = str(prediction["qid"]), str(prediction["presentation_id"])
        require(key in expected and key not in seen, "Unknown or duplicate raw prediction")
        seen.add(key)
        gold = expected[key]
        require(not prediction.get("technical_error"), "Technical failure means incomplete run, never model-invalid zero")
        require(prediction.get("sample_index") == 0 and prediction.get("prompt_kind") == "answer", "Unexpected sample or prompt")
        require(prediction.get("model_weights_sha256") == expected_weights_sha, "Raw weights identity differs")
        require(prediction.get("decoding", {}).get("do_sample") is False and prediction["decoding"]["max_new_tokens"] == 32,
                "Raw decoder differs from fixed greedy32")
        for field in ("recipe_id", "display_perm", "target_order", "target_labels"):
            require(prediction[field] == gold[field], "Raw/gold identity mismatch: " + field)
        metrics = independent_score(prediction["raw_response"], gold["display_perm"], gold["target_order"])
        for field in ("semantic_order", "valid", "exact", "pair"):
            require(prediction.get("score", {}).get(field) == metrics[field], "Independent score differs: " + field)
        scored.append({"qid": gold["qid"], "recipe_id": gold["recipe_id"], "presentation_id": gold["presentation_id"],
                       "display_perm": gold["display_perm"], "target_order": gold["target_order"], **metrics})
    require(seen == set(expected), "Incomplete raw denominator")
    return scored


def summarize(scored):
    by_recipe = defaultdict(list)
    for row in scored:
        by_recipe[str(row["recipe_id"])].append(row)
    require(len(by_recipe) == 16, "Expected16 recipes")
    recipes = []
    for recipe, rows in sorted(by_recipe.items()):
        require(len(rows) == 24 and {tuple(r['display_perm']) for r in rows} == set(PERMS), "Need full24 display orbit")
        successes = sum(r["exact"] for r in rows)
        counts = Counter(tuple(r["semantic_order"]) for r in rows if r["valid"])
        recipes.append({"recipe_id": recipe, "qid": rows[0]["qid"], "K": successes, "K_over24": successes / 24,
                        "all24_exact": int(successes == 24), "semantic_modal_consistency": max(counts.values(), default=0) / 24,
                        "pair": sum(r["pair_count"] for r in rows) / (24 * 6), "valid": sum(r["valid"] for r in rows) / 24})
    names = ("K_over24", "all24_exact", "semantic_modal_consistency", "pair", "valid")
    return {"n_recipes": 16, "n_predictions": 384, "recipe_macro": {k: sum(r[k] for r in recipes) / 16 for k in names},
            "per_recipe": recipes, "K_histogram": {str(k): sum(r["K"] == k for r in recipes) for k in range(25)},
            "consistency_note": "Modal consistency may be1 for the same consistently wrong semantic order; invalids remain denominator24 but are not a semantic mode."}


def estimate_remaining(pilot_entries, raw_sets):
    require(len(pilot_entries) == len(raw_sets) == 4 and all(len(rows) == 4 for rows in raw_sets), "Need all four arm pilots first")
    emissions = [entry["prediction_emissions"] for entry in pilot_entries]
    require(all(len(events) == 4 for events in emissions), "Need timestamped output for all four pilot records per model")
    # Arrival-to-arrival time includes the next record's preprocessing, encoding,
    # generation and JSON write; generation_seconds alone omits CPU input cost.
    intervals = [events[i]["seconds_since_start"] - events[i-1]["seconds_since_start"]
                 for events in emissions for i in range(1, 4)]
    require(all(value > 0 for value in intervals), "Pilot emission timestamps are not monotone")
    maximum_seconds = max(*intervals, *(row["generation_seconds"] for rows in raw_sets for row in rows))
    startup_sum = sum(events[0]["seconds_since_start"] for events in emissions)
    return {"estimated_seconds": 1.5 * (startup_sum + 1536 * maximum_seconds) + 300,
            "safety_factor": 1.5, "cpu_io_allowance_seconds": 300, "max_pilot_per_record_seconds_including_preprocessing": maximum_seconds,
            "four_load_overheads_seconds": startup_sum,
            "warning": "First4 fixed presentations per arm are a runtime pilot, not representative-tail proof. Hard deadline independently enforced."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=HERE / "ORBIT_PROTOCOL.json")
    args = parser.parse_args()
    policy_path = args.policy.resolve()
    policy = load(policy_path)
    require(policy["status"] == "locked_before_first_orbit_prediction", "Draft or unreviewed policy cannot execute")
    require(ROOT == Path(BASE) and ROOT.stat().st_dev != Path('/').stat().st_dev, "Remote separate data disk required")
    require(Path(sys.prefix).resolve().is_relative_to(ROOT) and Path(sys.executable).absolute().is_relative_to(ROOT), "Python environment escaped data disk")
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TMPDIR", "TORCH_HOME"):
        require(os.environ.get(name) and Path(os.environ[name]).resolve().is_relative_to(ROOT), "Cache/temp escaped data disk: " + name)
    for name in ("XDG_CACHE_HOME", "PIP_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE"):
        require(not os.environ.get(name) or Path(os.environ[name]).resolve().is_relative_to(ROOT), "Active cache escaped data disk: " + name)
    require(all(os.environ.get(name) == '4' for name in THREADS), "Require reviewed four-thread environment")
    required_sources = {str(path.relative_to(ROOT)) for path in (Path(__file__).resolve(), HERE / 'test_orbit_diagnostic.py', ROOT / 'scripts/predict.py', ROOT / 'scripts/validate_run.py', ROOT / 'src/recipeqa/modeling.py', ROOT / 'src/recipeqa/metrics.py')}
    require(required_sources.issubset(policy['sources_sha256']), 'Policy does not bind every coordinator/predictor/scorer source')
    require(policy["wall_limit_seconds"] == 7200 and policy["source_manifest"] == "data/manifests/val.jsonl"
            and policy["source_manifest_sha256"] == DEV_SHA and policy["n_recipes"] == 16 and policy["layouts"] == 24
            and policy["updates"] == 0 and policy["model_order"] == list(ARM_IDS), "Fixed scientific design changed")
    require(policy.get("absolute_deadline_utc"), "Freeze a new absolute deadline before launch")
    absolute_deadline = datetime.fromisoformat(policy["absolute_deadline_utc"].replace('Z', '+00:00'))
    require(absolute_deadline.tzinfo is not None, "UTC-aware deadline required")
    destination = (ROOT / policy["output_directory"]).resolve()
    require(destination.is_relative_to(ROOT / "exploratory") and not destination.exists(), "Fresh output required; no resume or restart")
    destination.mkdir(parents=True)
    (destination / "logs").mkdir()
    started = time.monotonic()
    deadline = started + 7200
    policy_sha = sha256_file(policy_path)
    state = {"status": "preflight", "pid": os.getpid(), "started_utc": utc(), "policy_sha256": policy_sha,
             "updates": 0, "optimizer_constructed": False, "test_access": False, "confirmation_access": False,
             "automatic_retry": False, "resume": False, "commands": [], "active_child_pid": None,
             "wall_limit_seconds": 7200, "absolute_deadline_utc": policy["absolute_deadline_utc"]}
    locks, active = [], None

    def update(**values):
        state.update(values, updated_utc=utc())
        write_json(destination / 'state.json', state)

    def remaining():
        return remaining_budget(deadline, absolute_deadline)

    def confined(value):
        path = (ROOT / value).resolve()
        require(path.is_relative_to(ROOT), "Path escaped data disk: " + str(path))
        return path

    def immutable():
        require(sha256_file(policy_path) == policy_sha, "Policy changed after launch")
        for path, digest in policy["sources_sha256"].items():
            require(sha256_file(confined(path)) == digest, "Frozen source/input changed: " + path)

    def stop_child():
        nonlocal active
        if active and active.poll() is None:
            try:
                os.killpg(active.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                active.wait(timeout=4)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(active.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                active.wait(timeout=4)

    def run(tag, arguments):
        nonlocal active
        immutable()
        entry = {"stage": tag, "argv": [str(v) for v in arguments], "started_utc": utc(), "returncode": None}
        state["commands"].append(entry)
        child_started = time.monotonic()
        with (destination / 'logs' / (tag + '.log')).open('x') as log:
            active = subprocess.Popen([sys.executable, '-u', *map(str, arguments)], cwd=ROOT,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            entry['prediction_emissions'] = []
            stream = active.stdout
            def consume_output():
                for line in stream:
                    log.write(line)
                    log.flush()
                    try:
                        payload = json.loads(line)
                        if all(key in payload for key in ('qid', 'presentation_id', 'raw_response', 'seconds')):
                            entry['prediction_emissions'].append({'qid': payload['qid'], 'presentation_id': payload['presentation_id'],
                                                                 'seconds_since_start': time.monotonic()-child_started})
                    except (ValueError, TypeError):
                        pass
            reader = threading.Thread(target=consume_output, daemon=True)
            reader.start()
            entry['pid'] = active.pid
            update(active_child_pid=active.pid)
            try:
                entry['returncode'] = active.wait(timeout=remaining())
            except BaseException:
                clear_budget_alarm()
                stop_child()
                entry['returncode'] = active.returncode
                raise
            finally:
                reader.join(timeout=2)
                require(not reader.is_alive(), 'Child output reader did not finish')
                stream.close()
                entry['wall_seconds'] = time.monotonic() - child_started
                entry['finished_utc'] = utc()
                active = None
                update(active_child_pid=None)
        require(entry['returncode'] == 0, 'Command failed; no automatic retry: ' + tag)
        return entry

    def fingerprint():
        return {"base": model_weight_identity(confined(policy['model_path'])),
                "adapters": {arm: model_weight_identity(confined(policy['models'][arm]['checkpoint']), adapter=True) for arm in ARM_IDS}}

    def predict(arm, manifest, tag):
        spec = policy['models'][arm]
        output = destination / tag / 'predictions.jsonl'
        arguments = [confined('scripts/predict.py'), '--model-path', confined(policy['model_path']), '--revision', policy['revision'],
                     '--adapter-path', confined(spec['checkpoint']), '--manifest', manifest, '--image-root', confined(policy['image_root']),
                     '--output', output, '--run-id', 'orbit_' + arm + '_' + tag, '--seed', str(spec['seed']),
                     '--temperature', '0', '--max-new-tokens', '32', '--min-pixels', '65536', '--max-pixels', '262144',
                     '--recipe-token-budget', '1536', '--max-input-tokens', '4096']
        entry = run(tag, arguments)
        config = load(str(output) + '.config.json')
        require(config['weight_identity']['base'] == before['base']
                and config['weight_identity']['adapter'] == before['adapters'][arm], 'Predictor weight fingerprint differs')
        require(config['input'] == policy['input'] and config['manifest_sha256'] == sha256_file(manifest), 'Predictor input/config differs')
        return output, entry, config

    previous_signal_handlers = install_termination_handlers()
    budget_alarm_previous = None
    def clear_budget_alarm():
        nonlocal budget_alarm_previous
        if budget_alarm_previous is not None:
            cancel_budget_alarm(budget_alarm_previous)
            budget_alarm_previous = None
    try:
        budget_alarm_previous = install_budget_alarm(remaining())
        update()
        immutable()
        remaining()
        for name in LOCK_NAMES:
            handle = (ROOT / 'metadata' / (name + '.lock')).open('a+')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise
            locks.append(handle)
        process_query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=min(15, remaining()))
        health_query = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=min(15, remaining()))
        health = {'compute_rc': process_query.returncode, 'compute_stdout': process_query.stdout, 'compute_stderr': process_query.stderr,
                  'health_rc': health_query.returncode, 'health_stdout': health_query.stdout, 'health_stderr': health_query.stderr}
        write_json(destination / 'gpu_health.json', health)
        require(process_query.returncode == 0 and not process_query.stdout.strip() and health_query.returncode == 0, 'GPU idle/health query failed')
        lines = health_query.stdout.strip().splitlines()
        require(len(lines) == 1, 'Require the existing single GPU')
        temperature, utilization, used, total = [float(v.strip()) for v in lines[0].split(',')]
        require(temperature < 85 and utilization < 10 and used < 1024 and total > 20000, 'GPU idle temperature/utilization/memory health gate failed')
        before = fingerprint()
        require(before['base'] == policy['base_identity'], 'Base weight fingerprint changed')
        for arm in ARM_IDS:
            spec = policy['models'][arm]
            require(spec['step'] == 64 and spec['checkpoint'].endswith('/' + arm.split('_', 1)[0] + '/' + arm.split('_', 1)[1] + '/checkpoint-64'), 'Only fixed budget endpoint64 allowed')
            require(before['adapters'][arm] == spec['identity'], 'Frozen endpoint weight fingerprint changed: ' + arm)
        write_json(destination / 'weights_before.json', before)
        source = confined(policy['source_manifest'])
        require(sha256_file(source) == DEV_SHA, 'Development manifest content changed')
        orbit = make_orbit(read_jsonl(source))
        manifest = destination / 'orbit16x24.jsonl'
        jsonl(manifest, orbit)
        pilot_manifest = destination / 'pilot4.jsonl'
        jsonl(pilot_manifest, orbit[:4])
        write_json(destination / 'cohort_receipt.json', {'source_sha256': DEV_SHA, 'manifest_sha256': sha256_file(manifest),
                   'selection': 'Existing fixed dev200 first16 file rows; not selected by any score',
                   'question_recipe_order': [[r['qid'], r['recipe_id']] for r in read_jsonl(source)[:16]],
                   'rows': 384, 'n_recipes': 16, 'layouts_per_question': 24, 'pilot_manifest_sha256': sha256_file(pilot_manifest)})
        pilots, raw_sets = [], []
        update(status='four_fixed_timing_pilots')
        for arm in ARM_IDS:
            path, entry, config = predict(arm, pilot_manifest, 'pilot_' + arm)
            raw = read_jsonl(path)
            rescore(orbit[:4], raw, config['model_weights_sha256'])
            pilots.append(entry)
            raw_sets.append(raw)
        gate = estimate_remaining(pilots, raw_sets)
        gate.update(remaining_seconds=remaining())
        gate['passed'] = gate['estimated_seconds'] < gate['remaining_seconds']
        write_json(destination / 'whole_four_arm_timing_gate.json', gate)
        require(gate['passed'], 'Whole-four-arm timing gate failed before any formal arm; no retries or truncated replacement design')
        results = {}
        update(status='running_all_four_endpoints')
        for arm in ARM_IDS:
            path, entry, config = predict(arm, manifest, 'formal_' + arm)
            coverage = destination / ('coverage_' + arm + '.json')
            run('coverage_' + arm, [confined('scripts/validate_run.py'), 'predictions', '--manifest', manifest,
                '--predictions', path, '--run-id', 'orbit_' + arm + '_formal_' + arm,
                '--checkpoint', confined(policy['models'][arm]['checkpoint']), '--seed', str(policy['models'][arm]['seed']),
                '--expected-questions', '16', '--expected-recipes', '16', '--require-orbit', '--output', coverage])
            require(load(coverage)['passed'] is True, 'Frozen coverage gate failed')
            scored = rescore(orbit, read_jsonl(path), config['model_weights_sha256'])
            jsonl(destination / ('independent_scored_' + arm + '.jsonl'), scored)
            results[arm] = summarize(scored)
            write_json(destination / ('summary_' + arm + '.json'), results[arm])
        remaining()
        immutable()
        remaining()
        after = fingerprint()
        remaining()
        write_json(destination / 'weights_after.json', after)
        require(before == after, 'On-disk weights or auxiliary model files changed')
        paired = {}
        for flow in ('flow44', 'flow45'):
            sft = {r['recipe_id']: r for r in results[flow + '_raw_sft']['per_recipe']}
            rl = {r['recipe_id']: r for r in results[flow + '_exact_kl']['per_recipe']}
            require(sft.keys() == rl.keys(), 'Paired recipe coverage differs')
            paired[flow] = {'direction': 'exact_kl_minus_raw_sft', 'per_recipe': [
                {'recipe_id': key, **{metric: rl[key][metric] - sft[key][metric] for metric in ('K_over24', 'all24_exact', 'semantic_modal_consistency', 'pair', 'valid')}}
                for key in sorted(sft)]}
        summary = {'status': 'complete_four_endpoint_orbits_development_only', 'results': results, 'paired': paired,
                   'updates': 0, 'optimizer_constructed': False, 'weights_before_after_equal': True,
                   'fingerprint_scope': 'On-disk base/adapter and auxiliary bytes; frozen predictor uses inference_mode and trainable=False; no claim of an additional in-memory parameter checksum.',
                   'limits': ['Post-hoc existing development first16 only; no population or causal claim.',
                              'Same fullSFT initialization, only two optimization flows; not independent full training seeds.',
                              'All models fixedstep64; do not mix with earlier dev-selected flow45 SFTstep48.',
                              'Semantic consistency can mean consistently wrong. Full24 layouts are nested within16 recipes, not384 independent samples.',
                              'No reward change, test/confirmation inference, checkpoint selection, or SOTA claim.']}
        remaining()
        write_json(destination / 'summary.json', summary)
        receipts = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                    for p in sorted(destination.rglob('*')) if p.is_file() and p.name not in ('state.json', 'artifacts_receipt.json')]
        write_json(destination / 'artifacts_receipt.json', {'files': receipts, 'policy_sha256': policy_sha})
        remaining()
        update(status=summary['status'], summary_sha256=sha256_file(destination / 'summary.json'), wall_seconds=time.monotonic()-started)
    except BaseException as error:
        clear_budget_alarm()
        stop_child()
        update(status='failed_or_budget_stopped_no_retry', error=type(error).__name__ + ': ' + str(error), wall_seconds=time.monotonic()-started)
        raise
    finally:
        clear_budget_alarm()
        for handle in reversed(locks):
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        restore_termination_handlers(previous_signal_handlers)


if __name__ == '__main__':
    main()
