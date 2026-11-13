#!/usr/bin/env python3
"""Independent CPU/file/mathematical verifier; never imports model libraries."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import bank_math as m

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REMOTE = Path('/root/autodl-tmp/rl_recipeqa')
ENDPOINTS = ('initial_sft768', 'flow44_raw_sft_256', 'flow44_exact_kl_256',
             'flow45_raw_sft_256', 'flow45_exact_kl_256')
BANK_COUNTS = {'train256': 256, 'dev200': 200}
INPUT = {'min_pixels': 65536, 'max_pixels': 262144, 'recipe_token_budget': 1536,
         'max_input_tokens': 4096, 'max_completion_tokens': 32,
         'prompt_version': 'recipeqa-order-v1'}
ORIGINAL_BANKS = {
    'train256': ('exploratory/principled_ordering_20261105/full_train_data/train_all7558.jsonl',
                 'b9979ee0612679d2e9460dd3bb8a6f68e7dea6794bc6cdcc3e1defdc87d8d949'),
    'dev200': ('data/manifests/val.jsonl',
               'c998796fbc07a24c7a24062d6d0aba07500c75b34568fd7ba67f8c38daae03c9')}
NEW_SOURCES = ('endpoint_bank_contract.py', 'score_endpoint_bank.py', 'run_endpoint_bank.py',
               'launch_endpoint_bank.py', 'run_cpu_preflight.py', 'build_draft_policy.py',
               'test_endpoint_bank_contract.py', 'bank_math.py', 'verify_endpoint_scores.py',
               'analyze_endpoint_bank.py', 'test_bank_math.py', 'test_verify_endpoint_scores.py',
               'test_analyze_endpoint_bank.py')
OLD_SOURCES = {
    'exploratory/principled_ordering_20261105/score_permutations.py': 'affeda80416145d2b6a134172e812d15d48a86d77c67128bd2e5f84e0c341819',
    'src/recipeqa/modeling.py': '5fedb7a60877e965861aa9174179880754cb0555c4167265fdd8c3781af39771',
    'src/recipeqa/metrics.py': '5287d44a3bb4ede3a44ce0a6625aee399260afecd42c984441a8e07492a2423a'}
SOURCE_PATHS = {('research/endpoint_bank_20261106/' + n) for n in NEW_SOURCES} | set(OLD_SOURCES)


def obj_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def utc(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    m.require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0,
              'Explicit UTC timestamp required')
    return result


class Files:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.inputs = {}; self.contents = {}

    def physical(self, name):
        """Hash large physical files without retaining their bytes in RAM."""
        path = self.path(name); digest = hashlib.sha256(); size = 0
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block); size += len(block)
        result = {'sha256': digest.hexdigest(), 'bytes': size}
        key = str(self.relative(name))
        m.require(key not in self.inputs or self.inputs[key] == result, 'Physical file changed during verification')
        self.inputs[key] = result
        return result

    def relative(self, name):
        path = Path(name)
        if path.is_absolute():
            path = path.relative_to(REMOTE) if path.is_relative_to(REMOTE) else path.relative_to(self.root)
        m.require('..' not in path.parts and str(path) not in ('', '.')
                  and not any(x.lower().startswith('confirmation') for x in path.parts),
                  'Unsafe/confirmation evidence path')
        if path.parts[:2] == ('data', 'manifests'):
            m.require(path.name == 'val.jsonl', 'Only known development manifest permitted')
        return path

    def path(self, name):
        result = (self.root / self.relative(name)).resolve()
        m.require(result.is_relative_to(self.root), 'Evidence escaped root')
        return result

    def read(self, name):
        key = str(self.relative(name))
        if key not in self.contents:
            data = self.path(name).read_bytes()
            self.contents[key] = data
            self.inputs[key] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        return self.contents[key]

    def sha(self, name):
        return hashlib.sha256(self.read(name)).hexdigest()

    def json(self, name): return json.loads(self.read(name))
    def rows(self, name): return [json.loads(x) for x in self.read(name).decode().splitlines() if x.strip()]

    def stable(self):
        for name, receipt in self.inputs.items():
            if name not in self.contents:
                m.require(self.physical(name) == receipt, 'Physical evidence changed')
            else:
                data = self.path(name).read_bytes()
                m.require(hashlib.sha256(data).hexdigest() == receipt['sha256']
                          and len(data) == receipt['bytes'], 'Evidence changed during independent verification')


def fixed_tasks():
    return [{'id': e + '__' + b, 'endpoint': e, 'bank': b, 'rows': n}
            for e in ENDPOINTS for b, n in BANK_COUNTS.items()
            if (e, b) != ('initial_sft768', 'train256')]


def check_policy(policy):
    m.require(policy['status'] == 'locked_before_endpoint_bank', 'Draft policy cannot execute')
    m.require(set(policy['endpoints']) == set(ENDPOINTS) and set(policy['banks']) == set(BANK_COUNTS)
              and policy['tasks'] == fixed_tasks(), 'Five endpoints/two banks/nine fixed tasks required')
    m.require(policy['input'] == INPUT and policy['beta'] == m.BETA
              and policy['candidate_policy'] == 'canonical24_full_sequence_plusEOS_temperature1'
              and policy['tie_policy'] == m.TIE_POLICY, 'Input/objective/candidate/tie policy changed')
    m.require(policy['formal_rows'] == 2024 and policy['pilot_rows'] == 18
              and policy['pilot_rows_per_task'] == 2, 'Formal and pilot denominators changed')
    m.require(policy['wall_limit_seconds'] == 10800 and policy['budget_cny_cap'] == 4.8
              and (utc(policy['absolute_deadline_utc']) - utc(policy['first_preflight_started_utc'])).total_seconds() == 10800,
              'Original independent3h budget/clock changed')
    m.require(all(policy[k] is False for k in ('test_access', 'confirmation_access', 'resume', 'automatic_retry'))
              and policy['optimizer_steps'] == policy['backward_calls'] == policy['generation_calls'] == 0,
              'Only zero-update/no-generation permitted')
    m.require(set(policy['sources_sha256']) == SOURCE_PATHS
              and all(policy['sources_sha256'][p] == h for p, h in OLD_SOURCES.items()),
              'Independent source closure/historical source pin differs')
    for bank, (name, digest) in ORIGINAL_BANKS.items():
        spec = policy['banks'][bank]
        m.require((spec['source_manifest'], spec['source_manifest_sha256']) == (name, digest),
                  'Bank changed from fixed original training/development source')
    timing = policy['historical_timing']
    m.require(timing['reference_outcome'] == 'exploratory/coverage_20261106/stage_v1/reference/reference_only_outcome.json'
              and timing['rows'] == 1024 and m.finite_number(timing['elapsed_seconds']) and timing['elapsed_seconds'] > 0
              and policy['artifacts_sha256'][timing['reference_outcome']] == timing['sha256']
              and policy['timing_gate_policy'] == 'first_row_once_remaining_second_row_with_historical_floor_1.5_plus600',
              'Historical resource floor/gate policy changed')
    m.require(policy['revision'] == '89644892e4d85e24eaac8bacfd4f463576704203', 'Model revision changed')
    m.require(isinstance(policy['required_versions'], dict) and policy['required_versions'],
              'Sealed source package versions required')
    initial = policy['endpoints']['initial_sft768']['weight_identity']
    for endpoint in ENDPOINTS:
        spec = policy['endpoints'][endpoint]
        expected = ('exploratory/principled_ordering_20261105/full_sft/train/checkpoint-768'
                    if endpoint == 'initial_sft768' else
                    'exploratory/coverage_20261106/stage_v1/' + endpoint[:-4] + '/train/checkpoint-256')
        m.require(spec['adapter_path'] == expected and spec['weight_identity']['base'] == initial['base']
                  and spec['weight_identity']['adapter']['directory'] == str(REMOTE / expected),
                  'Fixed checkpoint/common base identity changed')


def bank_records(files, policy, bank):
    spec = policy['banks'][bank]; records = files.rows(spec['manifest'])
    m.require(files.sha(spec['manifest']) == spec['manifest_sha256']
              and obj_sha(records) == spec['selected_records_sha256']
              and len(records) == spec['rows'] == BANK_COUNTS[bank], 'Fixed bank bytes/count changed')
    source = files.rows(spec['source_manifest'])
    m.require(files.sha(spec['source_manifest']) == spec['source_manifest_sha256']
              and records == (source[:256] if bank == 'train256' else source), 'Not exact original source/order')
    m.require(len({(r['qid'], r['presentation_id']) for r in records}) == len(records)
              and len({r['recipe_id'] for r in records}) == (254 if bank == 'train256' else 200),
              'Question or recipe-cluster denominator changed')
    for row in records:
        m.require(row['source_split'] == ('train' if bank == 'train256' else 'val')
                  and row['manifest_set'] == ('train_all7558' if bank == 'train256' else 'val')
                  and row['technical_status'] == 'ok' and len(row['image_paths_display']) == 4,
                  'Heldout/damaged/missing image record')
        m.target_index(row)
    return records


def check_rows(records, rows, signature, phase):
    payload = signature['payload']; out = []; residual = 0.
    m.require(len(rows) == len(records), 'Missing scoring denominator')
    for index, (record, row) in enumerate(zip(records, rows)):
        for key in ('qid', 'recipe_id', 'presentation_id', 'source_split', 'manifest_set',
                    'display_perm', 'target_order', 'target_labels'):
            m.require(row[key] == record[key], 'Record/display coordinate mismatch: ' + key)
        m.require(row['index'] == index and row['endpoint'] == payload['endpoint']
                  and row['bank'] == payload['bank'] and row['configuration_sha256'] == signature['sha256']
                  and row['model_weights_sha256'] == payload['model_weights_sha256'], 'Score signature/identity/order mismatch')
        m.require(row['optimizer_steps'] == row['backward_calls'] == row['generation_calls'] == 0
                  and row['candidate_forwards'] == (0 if phase == 'cached' else 25), 'Unexpected update/generation/compute')
        audit = row['input_audit']
        m.require(all(audit[k] == record[k] for k in ('qid', 'recipe_id', 'presentation_id'))
                  and audit['steps'] == audit['titles_retained'] == len(record['steps'])
                  and audit['input_tokens'] <= 4096 and audit['recipe_tokens_used'] <= 1536
                  and len(audit['prompt_text_hash']) == len(audit['input_hash']) == 64
                  and len(audit['image_grid_thw']) == 4, 'Actual input clipping/identity/grid differs')
        for grid in audit['image_grid_thw']:
            m.require(len(grid) == 3 and all(type(x) is int and x > 0 for x in grid), 'Bad photograph grid')
        m.require(row['pixel_sha256'] == {x: payload['pixel_sha256'][x] for x in record['image_paths_display']},
                  'Row photograph bytes identity differs')
        p = m.reconstruct(row['token_logps'], row['candidates'])
        checks = row['numerical_checks']
        m.require(len(checks) == 1 and checks[0]['candidate_index'] == 0
                  and checks[0]['literal_all_position_logits'] is (index == 0), 'Original literal numerical-check policy changed')
        literal = checks[0]['reference_token_logps']
        m.require(len(literal) == 5 and all(m.finite_number(x) and x <= 1e-6 for x in literal), 'Missing numerical reference tokens')
        difference = max(abs(a-b) for a, b in zip(literal, row['token_logps'][0]))
        residual = max(residual, m.close(checks[0]['max_token_logp_abs_difference'], difference,
                                         'actual numerical token difference'))
        m.require(difference <= .01 and m.finite_number(row['elapsed_seconds'])
                  and (row['elapsed_seconds'] == 0 if phase == 'cached' else row['elapsed_seconds'] > 0),
                  'Numerical/runtime observation failed')
        out.append({'record': record, 'row': row, 'policy': p, 'gold_index': m.target_index(record)})
    return out, residual


def check_completion(done, signature, count, phase):
    identity = signature['payload']['weight_identity']
    m.require(done['status'] == ('complete_reused_initial_train_scores' if phase == 'cached' else 'complete_zero_update_endpoint_scores')
              and done['signature_sha256'] == signature['sha256'] and done['rows'] == done['expected_rows'] == count
              and done['optimizer_steps'] == done['backward_calls'] == done['generation_calls'] == 0
              and done['candidate_forwards'] == (0 if phase == 'cached' else 25 * count), 'Incomplete/count/update outcome')
    m.require(done['weight_identity_before'] == done['weight_identity_after'] == identity,
              'Endpoint physical identity changed')
    if phase == 'cached':
        m.require(done['model_was_loaded'] is False and done['runtime_model_reloaded'] is False
                  and done['new_model_calls'] == 0
                  and all(done[k] is None for k in ('runtime_lora_before', 'runtime_lora_after', 'runtime_visual_before', 'runtime_visual_after')),
                  'Cache reuse pretends new model/fingerprint observations')
    else:
        before = done['runtime_lora_before']
        m.require(before == done['runtime_lora_after'] and before['tensors'] == 392
                  and before['numel'] == 17432576 and len(before['sha256']) == 64
                  and before['selection'] == 'all language-model lora_ parameters regardless of requires_grad',
                  'Nonempty inference LoRA fingerprint changed/missing')
        m.require(done['runtime_visual_before'] == done['runtime_visual_after']
                  and isinstance(done['runtime_visual_before'], str) and len(done['runtime_visual_before']) == 64
                  and done['all_modules_eval'] is True
                  and done['all_parameters_frozen'] is True and done['trainable_parameters_numel'] == 0
                  and done['model_was_loaded'] is True, 'Frozen eval visual/model state changed')
    m.require(m.finite_number(done['elapsed_seconds']) and done['elapsed_seconds'] > 0,
              'Missing actual process wall time')


def check_authorization(files, policy, policy_sha, payload, phase, task_id, count, begin):
    root = Path(policy['output_directory'])
    path = root / 'authorizations' / (phase + '_' + task_id + '.json')
    auth = files.json(path)
    m.require(payload['authorization_sha256'] == files.sha(path)
              and auth['policy_sha256'] == policy_sha and auth['phase'] == phase
              and auth['task_id'] == task_id and auth['rows'] == count
              and auth['pgid'] == payload['pgid'], 'GPU authorization scope/signature/group changed')
    cpu_path = root / 'cpu_preflight.json'; cpu = files.json(cpu_path)
    cached_path = root / 'cached_initial_train/independent_verification.json'; cached = files.json(cached_path)
    m.require(auth['cpu_preflight_sha256'] == files.sha(cpu_path) and cpu['passed'] is True
              and cpu['policy_sha256'] == policy_sha and cpu['sources_sha256'] == policy['sources_sha256']
              and cpu['first_preflight_started_utc'] == policy['first_preflight_started_utc']
              and cpu['absolute_deadline_utc'] == policy['absolute_deadline_utc']
              and cpu['tests'] == cpu['expected_tests'] == policy['expected_cpu_tests']
              and all(cpu[k] == 0 for k in ('failures', 'errors', 'skipped', 'expected_failures', 'unexpected_successes'))
              and cpu['torch_imported'] is False and cpu['model_gpu_calls'] == 0
              and cpu['native_watchdog_probe']['passed'] is True, 'Actual CPU/native prerequisites incomplete')
    m.require(auth['cached_initial_train_audit_sha256'] == files.sha(cached_path)
              and cached['passed'] is True and cached['phase'] == 'cached'
              and cached['task_id'] == 'initial_sft768__train256' and cached['rows'] == 256
              and cached['policy_sha256'] == policy_sha
              and cached['verifier_sha256'] == policy['sources_sha256']['research/endpoint_bank_20261106/verify_endpoint_scores.py'],
              'Cached independent audit not policy/source bound')
    issued = utc(auth['issued_utc'])
    m.require(max(utc(cpu['completed_utc']), utc(cached['completed_utc'])) <= issued <= begin,
              'GPU authorization/scoring preceded actual prerequisite completion')
    if phase == 'formal':
        gate_path = root / 'whole_nine_task_gate.json'; gate = files.json(gate_path)
        m.require(auth['whole_gate_sha256'] == files.sha(gate_path) and gate['passed'] is True
                  and gate['formal_rows'] == 2024 and gate['pilot_rows_separate'] == 18
                  and gate['all9_formal_tasks_required'] is True
                  and gate['estimated_seconds'] < gate['remaining_seconds']
                  and gate['required_free_bytes'] <= gate['actual_free_bytes']
                  and set(gate['pilot_receipts_sha256']) == {t['id'] for t in fixed_tasks()},
                  'Whole-nine formal gate incomplete')
        for task in fixed_tasks():
            pilot_path = root / 'pilot' / task['id'] / 'independent_verification.json'
            pilot = files.json(pilot_path)
            m.require(files.sha(pilot_path) == gate['pilot_receipts_sha256'][task['id']]
                      and pilot['passed'] is True and pilot['phase'] == 'pilot'
                      and pilot['task_id'] == task['id'] and pilot['rows'] == 2
                      and pilot['policy_sha256'] == policy_sha
                      and pilot['verifier_sha256'] == policy['sources_sha256']['research/endpoint_bank_20261106/verify_endpoint_scores.py']
                      and utc(pilot['completed_utc']) <= utc(gate['completed_utc']),
                      'Missing/mis-scoped pilot independent audit')
        m.require(utc(gate['completed_utc']) <= issued, 'Formal authorization preceded all-nine gate')
        verify_resource_gate(files, policy, gate)
    else:
        m.require(auth['whole_gate_sha256'] is None, 'Pilot improperly claims formal gate')


def verify_resource_gate(files, policy, gate):
    """Independently recompute declared all-nine resource estimate, not scores."""
    spec = policy['historical_timing']; historical = files.json(spec['reference_outcome'])
    m.require(files.sha(spec['reference_outcome']) == spec['sha256']
              and historical['rows'] == spec['rows'] == 1024
              and historical['elapsed_seconds'] == spec['elapsed_seconds']
              and historical['updates'] == 0 and historical['optimizer_constructed'] is False,
              'Historical floor not actual complete zero-update reference timing')
    m.require(gate['historical_reference_timing'] == spec, 'Gate historical timing provenance differs')
    floor = historical['elapsed_seconds'] / 1024
    m.close(gate['historical_per_row_floor_seconds'], floor, 'historical per-row floor')
    inputs = gate['pilot_timing_inputs']; tasks = fixed_tasks()
    m.require(set(inputs) == {t['id'] for t in tasks}, 'Resource gate omits a pilot')
    scoring = loading = payload_bytes = 0.
    for task in tasks:
        row = inputs[task['id']]
        proof = files.json(Path(policy['output_directory']) / 'pilot' / task['id'] / 'independent_verification.json')
        m.require(row['passed'] is True and row['rows'] == proof['rows'] == 2
                  and row['row_seconds'] == proof['row_seconds'] and row['raw_bytes'] == proof['raw_bytes']
                  and len(row['row_seconds']) == 2 and all(m.finite_number(x) and x > 0 for x in row['row_seconds'])
                  and row['command_wall_seconds'] >= proof['command_wall_seconds'] - 1e-6,
                  'Actual pilot resource inputs differ from independent receipt')
        first, second = row['row_seconds']; rest = max(second, floor)
        score = first + (task['rows'] - 1) * rest
        load = max(0., row['command_wall_seconds'] - first - second)
        scoring += score; loading += load; payload_bytes += task['rows'] * row['raw_bytes'] / 2
        parts = gate['task_components'][task['id']]
        for key, value in (('first_row_once_seconds', first), ('second_row_observed_seconds', second),
                           ('later_row_forecast_seconds', rest), ('formal_score_forecast_seconds', score),
                           ('model_load_hash_fingerprint_io_allowance_seconds', load)):
            m.close(parts[key], value, 'task resource component ' + key)
    m.close(gate['formal_scoring_seconds'], scoring, 'whole scoring forecast')
    m.close(gate['nine_model_load_allowance_seconds'], loading, 'whole load allowance')
    m.close(gate['estimated_seconds'], 1.5 * (scoring + loading) + 600, 'whole conservative seconds')
    required = math.ceil(2 * payload_bytes + 256 * 1024 ** 2 + 1024 ** 3)
    m.require(gate['required_free_bytes'] == required and gate['actual_free_bytes'] >= required
              and gate['safety_factor'] == 1.5 and gate['validation_allowance_seconds'] == 600
              and gate['passed'] is True and gate['estimated_seconds'] < gate['remaining_seconds'],
              'Whole resource capacity/time gate failed or formula changed')
    return {'scoring_seconds': scoring, 'load_allowance_seconds': loading,
            'estimated_seconds': 1.5 * (scoring + loading) + 600, 'required_free_bytes': required}


def verify_task(files, policy_path, phase, task_id, run_dir, physical=True):
    policy = files.json(policy_path); check_policy(policy); policy_sha = files.sha(policy_path)
    if phase == 'cached':
        m.require(task_id == 'initial_sft768__train256', 'Unknown cached role')
        endpoint, bank, count = 'initial_sft768', 'train256', 256
    else:
        m.require(phase in ('pilot', 'formal'), 'Unknown score phase')
        task = next((x for x in fixed_tasks() if x['id'] == task_id), None)
        m.require(task is not None, 'Unknown/unplanned task')
        endpoint, bank = task['endpoint'], task['bank']; count = 2 if phase == 'pilot' else task['rows']
    expected_dir = Path(policy['output_directory']) / ('cached_initial_train' if phase == 'cached' else phase + '/' + task_id)
    m.require(files.relative(run_dir) == expected_dir, 'Score output escaped fixed task')
    for group in ('sources_sha256', 'artifacts_sha256'):
        for name, digest in policy[group].items():
            m.require(files.sha(name) == digest, 'Frozen source/artifact changed: ' + name)
    for module in ('bank_math.py', 'verify_endpoint_scores.py'):
        name = 'research/endpoint_bank_20261106/' + module
        m.require(hashlib.sha256((HERE / module).read_bytes()).hexdigest() == policy['sources_sha256'][name],
                  'Actually executing independent mathematical/verifier source differs from frozen source')
    records = bank_records(files, policy, bank)[:count]
    signature = files.json(expected_dir / 'signature.json'); payload = signature['payload']
    m.require(signature['sha256'] == obj_sha(payload) and payload['policy_sha256'] == policy_sha
              and payload['phase'] == phase and payload['task_id'] == task_id
              and payload['endpoint'] == endpoint and payload['bank'] == bank
              and payload['zero_updates'] is True, 'Run signature/locked policy mismatch')
    identity = policy['endpoints'][endpoint]['weight_identity']
    source_config_path = policy['cached_initial_train']['source_config']
    source_config = files.json(source_config_path)
    m.require(files.sha(source_config_path) == policy['cached_initial_train']['source_receipt_sha256'][source_config_path]
              and source_config['versions'] == policy['required_versions'] == payload['versions']
              and payload['versions_source'] == ('sealed_source_cache' if phase == 'cached' else 'actual_runtime_package_versions'),
              'Actual/source-cache package versions differ or provenance missing')
    m.require(payload['weight_identity'] == identity and payload['model_weights_sha256'] == obj_sha(identity)
              and payload['selected_records_sha256'] == obj_sha(records)
              and payload['manifest_sha256'] == policy['banks'][bank]['manifest_sha256']
              and payload['input'] == policy['input'] and payload['sources_sha256'] == policy['sources_sha256']
              and payload['first_preflight_started_utc'] == policy['first_preflight_started_utc']
              and payload['absolute_deadline_utc'] == policy['absolute_deadline_utc'], 'Fixed model/input/selection/clock mismatch')
    all_photos = {x for record in records for x in record['image_paths_display']}
    m.require(set(payload['pixel_sha256']) == all_photos, 'Signature image denominator differs')
    if physical:
        for name, digest in payload['pixel_sha256'].items():
            m.require(files.sha(Path(policy['image_root']) / name) == digest, 'Actual photograph bytes changed')
        for part in ('base', 'adapter'):
            item = identity[part]
            for spec in item['weight_files'] + item.get('auxiliary_files', []):
                name = Path(item['directory']) / spec['name']
                actual = files.physical(name)
                m.require(actual['sha256'] == spec['sha256']
                          and ('bytes' not in spec or actual['bytes'] == spec['bytes']), 'Physical endpoint file changed')
    raw_path = expected_dir / 'scores.jsonl'; done = files.json(expected_dir / 'completion.json')
    check_completion(done, signature, count, phase)
    m.require(done['output_sha256'] == files.sha(raw_path), 'Scores bytes not committed complete')
    begin, finish = utc(done['started_utc']), utc(done['completed_utc'])
    m.require(utc(policy['first_preflight_started_utc']) <= begin <= finish < utc(policy['absolute_deadline_utc']),
              'Score completion outside original budget clock')
    if phase != 'cached':
        check_authorization(files, policy, policy_sha, payload, phase, task_id, count, begin)
    raw = files.rows(raw_path); parsed, residual = check_rows(records, raw, signature, phase)
    for row in raw:
        m.require(begin <= utc(row['completed_utc']) <= finish, 'Row emission outside committed score run')
    if phase == 'cached':
        spec = policy['cached_initial_train']
        m.require(payload['source_receipt_sha256'] == spec['source_receipt_sha256'], 'Cache source lineage differs')
        for name, digest in spec['source_receipt_sha256'].items():
            m.require(files.sha(name) == digest, 'Cached initial source changed')
        source = files.rows(spec['source_cache']); config = files.json(spec['source_config'])
        source_done = files.json(spec['source_completion']); proof = files.json(spec['source_audit'])
        signed = files.json(spec['source_signature'])
        source_payload = signed['payload']
        m.require(len(source) == 1024 and signed['sha256'] == obj_sha(source_payload)
                  and config['initial_identity'] == identity and config['input'] == policy['input']
                  and config['revision'] == policy['revision'], 'Cached source model/config incomplete')
        source_records = files.rows(spec['source_manifest'])
        m.require(records == source_records[:256]
                  and source_payload['initial_identity'] == identity and source_payload['input'] == config['input']
                  and source_payload['reference_policy_sha256'] == files.sha(spec['source_cache'])
                  and source_payload['reference_spec_sha256'] == obj_sha(config)
                  and source_payload['selected_records_sha256'] == obj_sha(source_records)
                  and proof['run_signature_sha256'] == signed['sha256'], 'Cache proof/signature/config not same actual source run')
        m.require(source_done['status'] == 'complete_fixed_initial_reference'
                  and source_done['rows'] == 1024 and source_done['cache_sha256'] == files.sha(spec['source_cache'])
                  and source_done['reference_spec_sha256'] == obj_sha(config)
                  and proof['passed'] is True and proof['updates'] == 0
                  and proof['status'] == 'complete_verified_coverage_reference_only', 'Cached source lacks actual complete independent proof')
        lineage = files.json(expected_dir / 'lineage.json')
        m.require(done['lineage_sha256'] == files.sha(expected_dir / 'lineage.json')
                  and lineage['rows'] == len(lineage['rows_lineage']) == 256
                  and lineage['new_candidate_forwards'] == 0, 'Cache copy lineage incomplete')
        for i, row in enumerate(raw):
            source_row = source[i]
            m.require(row['source_index'] == source_row['index'] == i
                      and row['source_payload'] == source_row and row['source_row_sha256'] == obj_sha(source_row),
                      'Cache source row/payload modified')
            m.require(source_row['reference_spec_sha256'] == obj_sha(config), 'Cache row source spec differs')
            for key in ('qid', 'recipe_id', 'presentation_id', 'source_split', 'display_perm', 'target_order', 'target_labels'):
                m.require(source_row[key] == records[i][key], 'Cache source gold/identity differs')
            for name, digest in row['pixel_sha256'].items():
                m.require(config['pixel_sha256'][name] == digest, 'Cache source pixel identity differs')
            for key in ('input_audit', 'candidates', 'token_logps', 'numerical_checks'):
                m.require(row[key] == source_row[key], 'Cache numeric/input value changed')
            link = lineage['rows_lineage'][i]
            m.require(link['index'] == i and link['source_row_sha256'] == obj_sha(source_row)
                      and link['record_sha256'] == obj_sha(records[i]) and link['output_row_sha256'] == obj_sha(row),
                      'Cache identity-only copy receipt differs')
    files.stable()
    return {'passed': True, 'status': 'independently_verified_zero_update_endpoint_scores',
            'phase': phase, 'task_id': task_id, 'endpoint': endpoint, 'bank': bank, 'rows': count,
            'policy_sha256': policy_sha, 'signature_sha256': signature['sha256'],
            'output_sha256': files.sha(raw_path), 'raw_bytes': len(files.read(raw_path)),
            'row_seconds': [x['elapsed_seconds'] for x in raw], 'command_wall_seconds': done['elapsed_seconds'],
            'candidate_forwards': 0 if phase == 'cached' else 25 * count,
            'optimizer_steps': 0, 'backward_calls': 0, 'generation_calls': 0,
            'maximum_numeric_check_residual': residual, 'physical_pixels_and_model_files_verified': physical,
            'completed_utc': datetime.now(timezone.utc).isoformat(),
            'runtime_fingerprints_bound_not_locally_replayed': True,
            'input_receipts': files.inputs,
            'input_files_verified': len(files.inputs),
            'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            '_parsed': parsed}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--phase', choices=('pilot', 'formal', 'cached'), required=True)
    parser.add_argument('--task-id', required=True); parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--root', type=Path, default=PROJECT)
    args = parser.parse_args(argv); files = Files(args.root)
    report = verify_task(files, args.policy, args.phase, args.task_id, args.run_dir, physical=True)
    report.pop('_parsed')
    m.require(not args.output.exists(), 'Fresh verifier output required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('passed', 'phase', 'task_id', 'rows')}))


if __name__ == '__main__':
    try: main()
    except (ValueError, KeyError, FileNotFoundError, AssertionError) as exc:
        print(json.dumps({'passed': False, 'error': type(exc).__name__ + ': ' + str(exc)}), file=sys.stderr)
        raise SystemExit(1)
