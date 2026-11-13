"""Synthetic verifier/failure guards; no model import or new GPU results."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bank_math as m
import verify_endpoint_scores as v
from test_bank_math import candidates


def policy_fixture():
    endpoints = {}
    base = {'directory': str(v.REMOTE / 'models/Qwen3-VL-2B-Instruct')}
    for name in v.ENDPOINTS:
        path = ('exploratory/principled_ordering_20261105/full_sft/train/checkpoint-768'
                if name == 'initial_sft768' else
                'exploratory/coverage_20261106/stage_v1/' + name[:-4] + '/train/checkpoint-256')
        endpoints[name] = {'adapter_path': path, 'weight_identity': {'base': base,
                           'adapter': {'directory': str(v.REMOTE / path)}}}
    historical = {'reference_outcome': 'exploratory/coverage_20261106/stage_v1/reference/reference_only_outcome.json',
                  'sha256': 'a' * 64, 'rows': 1024, 'elapsed_seconds': 2560.}
    return {'status': 'locked_before_endpoint_bank', 'endpoints': endpoints,
            'banks': {b: {'source_manifest': name, 'source_manifest_sha256': digest}
                      for b, (name, digest) in v.ORIGINAL_BANKS.items()},
            'tasks': v.fixed_tasks(), 'input': v.INPUT, 'beta': m.BETA,
            'candidate_policy': 'canonical24_full_sequence_plusEOS_temperature1', 'tie_policy': m.TIE_POLICY,
            'formal_rows': 2024, 'pilot_rows': 18, 'pilot_rows_per_task': 2,
            'wall_limit_seconds': 10800, 'budget_cny_cap': 4.8,
            'first_preflight_started_utc': '2026-11-07T00:00:00Z', 'absolute_deadline_utc': '2026-11-07T03:00:00Z',
            'test_access': False, 'confirmation_access': False, 'resume': False, 'automatic_retry': False,
            'optimizer_steps': 0, 'backward_calls': 0, 'generation_calls': 0,
            'sources_sha256': {p: v.OLD_SOURCES.get(p, '0' * 64) for p in v.SOURCE_PATHS},
            'artifacts_sha256': {historical['reference_outcome']: historical['sha256']},
            'historical_timing': historical,
            'timing_gate_policy': 'first_row_once_remaining_second_row_with_historical_floor_1.5_plus600',
            'required_versions': {'python': 'synthetic CPU-only fixture'},
            'revision': '89644892e4d85e24eaac8bacfd4f463576704203'}


def row_fixture(phase='pilot'):
    records = []; rows = []
    payload = {'endpoint': 'initial_sft768', 'bank': 'dev200', 'model_weights_sha256': 'cpu-model',
               'weight_identity': {'base': 'cpu', 'adapter': 'cpu'},
               'pixel_sha256': {f'{i}.jpg': str(i) * 64 for i in range(4)}}
    signature = {'sha256': 'cpu-signature', 'payload': payload}
    for i in range(2):
        record = {'qid': f'q{i}', 'recipe_id': f'r{i}', 'presentation_id': f'p{i}', 'source_split': 'val',
                  'manifest_set': 'val', 'display_perm': [3, 2, 1, 0], 'target_order': [0, 1, 2, 3],
                  'target_labels': ['D', 'C', 'B', 'A'], 'steps': [1, 2, 3, 4],
                  'image_paths_display': [f'{j}.jpg' for j in range(4)]}
        audit = {k: record[k] for k in ('qid', 'recipe_id', 'presentation_id')}
        audit.update(steps=4, titles_retained=4, input_tokens=900, recipe_tokens_used=400,
                     prompt_text_hash='a' * 64, input_hash='b' * 64,
                     image_grid_thw=[[1, 16, 16]] * 4)
        row = {k: record[k] for k in ('qid', 'recipe_id', 'presentation_id', 'source_split', 'manifest_set',
                                    'display_perm', 'target_order', 'target_labels')}
        row.update(index=i, endpoint=payload['endpoint'], bank=payload['bank'],
                   configuration_sha256=signature['sha256'], model_weights_sha256='cpu-model',
                   optimizer_steps=0, backward_calls=0, generation_calls=0,
                   candidate_forwards=0 if phase == 'cached' else 25, input_audit=audit,
                   pixel_sha256=payload['pixel_sha256'], candidates=candidates(), token_logps=[[-1.] * 5 for _ in range(24)],
                   numerical_checks=[{'candidate_index': 0, 'literal_all_position_logits': i == 0,
                                      'reference_token_logps': [-1.] * 5, 'max_token_logp_abs_difference': 0.}],
                   elapsed_seconds=0. if phase == 'cached' else 1.)
        records.append(record); rows.append(row)
    return records, rows, signature


def completion_fixture(phase='pilot'):
    _, _, signed = row_fixture(phase)
    fingerprint = {'sha256': 'a' * 64, 'tensors': 392, 'numel': 17432576,
                   'selection': 'all language-model lora_ parameters regardless of requires_grad'}
    result = {'status': 'complete_reused_initial_train_scores' if phase == 'cached' else 'complete_zero_update_endpoint_scores',
              'signature_sha256': signed['sha256'], 'rows': 2, 'expected_rows': 2,
              'optimizer_steps': 0, 'backward_calls': 0, 'generation_calls': 0,
              'candidate_forwards': 0 if phase == 'cached' else 50,
              'weight_identity_before': signed['payload']['weight_identity'],
              'weight_identity_after': signed['payload']['weight_identity'],
              'runtime_lora_before': None if phase == 'cached' else fingerprint,
              'runtime_lora_after': None if phase == 'cached' else fingerprint,
              'runtime_visual_before': None if phase == 'cached' else 'd' * 64,
              'runtime_visual_after': None if phase == 'cached' else 'd' * 64,
              'all_modules_eval': True, 'all_parameters_frozen': True, 'trainable_parameters_numel': 0,
              'model_was_loaded': phase != 'cached', 'runtime_model_reloaded': False,
              'new_model_calls': 0, 'elapsed_seconds': 3.}
    return result, signed


class IndependentVerifierTests(unittest.TestCase):
    def test_fixed_policy_source_bank_clock_and_task_guards(self):
        p = policy_fixture(); v.check_policy(p)
        self.assertEqual((len(v.fixed_tasks()), sum(t['rows'] for t in v.fixed_tasks())), (9, 2024))
        for change in ('draft', 'tasks', 'source', 'bank', 'clock', 'endpoint', 'tie'):
            bad = deepcopy(p)
            if change == 'draft': bad['status'] = 'draft_endpoint_bank'
            if change == 'tasks': bad['tasks'].pop()
            if change == 'source': bad['sources_sha256'].pop(next(iter(bad['sources_sha256'])))
            if change == 'bank': bad['banks']['train256']['source_manifest_sha256'] = 'f' * 64
            if change == 'clock': bad['absolute_deadline_utc'] = '2026-11-07T04:00:00Z'
            if change == 'endpoint': bad['endpoints']['flow44_raw_sft_256']['adapter_path'] += '-best'
            if change == 'tie': bad['tie_policy'] = 'favor_gold'
            with self.subTest(change=change), self.assertRaises(ValueError): v.check_policy(bad)

    def test_rows_reconstruct_complete_EOS_without_producer_math(self):
        records, rows, signed = row_fixture(); parsed, residual = v.check_rows(records, rows, signed, 'pilot')
        self.assertEqual(len(parsed), 2); self.assertEqual(residual, 0)
        self.assertAlmostEqual(parsed[0]['policy']['q'][0], 1 / 24)

    def test_missing_reordered_mapping_and_false_literal_policy_rejected(self):
        for change in ('missing', 'reordered', 'mapping', 'literal', 'image', 'update'):
            records, rows, signed = row_fixture()
            if change == 'missing': rows.pop()
            if change == 'reordered': rows.reverse()
            if change == 'mapping': rows[0]['display_perm'] = [0, 1, 2, 3]
            if change == 'literal': rows[1]['numerical_checks'][0]['literal_all_position_logits'] = True
            if change == 'image': rows[0]['pixel_sha256'] = {'other.jpg': 'a' * 64}
            if change == 'update': rows[0]['optimizer_steps'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError): v.check_rows(records, rows, signed, 'pilot')

    def test_recorded_numerical_residual_must_equal_actual_token_difference(self):
        records, rows, signed = row_fixture(); rows[0]['numerical_checks'][0]['reference_token_logps'][0] -= .02
        with self.assertRaises(ValueError): v.check_rows(records, rows, signed, 'pilot')
        rows[0]['numerical_checks'][0]['max_token_logp_abs_difference'] = .02
        with self.assertRaises(ValueError): v.check_rows(records, rows, signed, 'pilot')

    def test_zero_update_frozen_nonempty_inference_fingerprint_required(self):
        done, signed = completion_fixture(); v.check_completion(done, signed, 2, 'pilot')
        for change in ('empty', 'changed', 'trainable', 'backward'):
            bad = deepcopy(done)
            if change == 'empty': bad['runtime_lora_before']['tensors'] = 0
            if change == 'changed': bad['weight_identity_after'] = {'different': True}
            if change == 'trainable': bad['all_parameters_frozen'] = False
            if change == 'backward': bad['backward_calls'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError): v.check_completion(bad, signed, 2, 'pilot')

    def test_cached_reuse_cannot_pretend_new_model_measurements(self):
        done, signed = completion_fixture('cached'); v.check_completion(done, signed, 2, 'cached')
        done['model_was_loaded'] = True
        with self.assertRaises(ValueError): v.check_completion(done, signed, 2, 'cached')

    def test_path_scope_and_byte_stability(self):
        with tempfile.TemporaryDirectory() as temp:
            f = v.Files(temp); path = Path(temp) / 'safe.json'; path.write_text('{}')
            self.assertEqual(f.json('safe.json'), {})
            for name in ('../outside', 'data/manifests/test.jsonl', 'confirmation200/manifest.jsonl'):
                with self.subTest(name=name), self.assertRaises(ValueError): f.relative(name)
            path.write_text('{"changed":true}')
            with self.assertRaises(ValueError): f.stable()

    def test_actual_CPU_cache_policy_source_and_time_must_precede_GPU_authorization(self):
        policy = policy_fixture(); policy['output_directory'] = 'exploratory/endpoint_bank_20261106/cpu_fixture'
        policy['expected_cpu_tests'] = 99; stage = Path(policy['output_directory'])
        task = v.fixed_tasks()[0]['id']; sha = 'synthetic-policy'; verifier = policy['sources_sha256']['research/endpoint_bank_20261106/verify_endpoint_scores.py']
        class Memory:
            def __init__(self): self.data = {}
            def json(self, path): return deepcopy(self.data[str(path)])
            def sha(self, path): return v.obj_sha(self.data[str(path)])
        for change in ('valid', 'cpu_policy', 'cache_after_authorization', 'detached', 'source'):
            files = Memory()
            cpu = {'passed': True, 'tests': 99, 'expected_tests': 99, 'policy_sha256': sha,
                   'sources_sha256': policy['sources_sha256'], 'first_preflight_started_utc': policy['first_preflight_started_utc'],
                   'absolute_deadline_utc': policy['absolute_deadline_utc'], 'failures': 0, 'errors': 0, 'skipped': 0,
                   'expected_failures': 0, 'unexpected_successes': 0, 'torch_imported': False, 'model_gpu_calls': 0,
                   'native_watchdog_probe': {'passed': True}, 'completed_utc': '2026-11-07T00:01:00Z'}
            cached = {'passed': True, 'phase': 'cached', 'task_id': 'initial_sft768__train256', 'rows': 256,
                      'policy_sha256': sha, 'verifier_sha256': verifier, 'completed_utc': '2026-11-07T00:02:00Z'}
            if change == 'cpu_policy': cpu['policy_sha256'] = 'old-policy'
            if change == 'cache_after_authorization': cached['completed_utc'] = '2026-11-07T00:04:00Z'
            if change == 'source': cached['verifier_sha256'] = 'wrong-source'
            files.data[str(stage / 'cpu_preflight.json')] = cpu
            files.data[str(stage / 'cached_initial_train/independent_verification.json')] = cached
            auth_path = stage / 'authorizations' / ('pilot_' + task + '.json')
            auth = {'policy_sha256': sha, 'phase': 'pilot', 'task_id': task, 'rows': 2, 'pgid': 100,
                    'cpu_preflight_sha256': files.sha(stage / 'cpu_preflight.json'),
                    'cached_initial_train_audit_sha256': files.sha(stage / 'cached_initial_train/independent_verification.json'),
                    'issued_utc': '2026-11-07T00:03:00Z', 'whole_gate_sha256': None}
            files.data[str(auth_path)] = auth
            payload = {'authorization_sha256': files.sha(auth_path), 'pgid': 101 if change == 'detached' else 100}
            args = (files, policy, sha, payload, 'pilot', task, 2, v.utc('2026-11-07T00:03:01Z'))
            if change == 'valid': v.check_authorization(*args)
            else:
                with self.subTest(change=change), self.assertRaises(ValueError): v.check_authorization(*args)

    def test_large_physical_hash_is_streamed_and_stability_rechecked(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.safetensors'; data = b'x' * (2 * 1024 * 1024 + 3)
            path.write_bytes(data); files = v.Files(temp)
            actual = files.physical('model.safetensors')
            self.assertEqual(actual, {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
            self.assertNotIn('model.safetensors', files.contents); files.stable()
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError): files.stable()

    def test_resource_gate_first_once_historical_floor_and_all_nine_are_recomputed(self):
        policy = policy_fixture(); policy['output_directory'] = 'exploratory/endpoint_bank_20261106/fixture'
        spec = policy['historical_timing']; pilot_inputs = {}; parts = {}; proofs = {}
        for task in v.fixed_tasks():
            pilot_inputs[task['id']] = {'passed': True, 'rows': 2, 'row_seconds': [4., 1.], 'raw_bytes': 100, 'command_wall_seconds': 8.}
            parts[task['id']] = {'first_row_once_seconds': 4., 'second_row_observed_seconds': 1.,
                                 'later_row_forecast_seconds': 2.5, 'formal_score_forecast_seconds': 4 + (task['rows'] - 1) * 2.5,
                                 'model_load_hash_fingerprint_io_allowance_seconds': 3.}
            proofs[task['id']] = {'rows': 2, 'row_seconds': [4., 1.], 'raw_bytes': 100, 'command_wall_seconds': 7.}
        class Memory:
            def json(self, path):
                if str(path) == spec['reference_outcome']:
                    return {'rows': 1024, 'elapsed_seconds': 2560., 'updates': 0, 'optimizer_constructed': False}
                return proofs[Path(path).parent.name]
            def sha(self, path): return spec['sha256']
        gate = {'historical_reference_timing': spec, 'historical_per_row_floor_seconds': 2.5,
                'pilot_timing_inputs': pilot_inputs, 'task_components': parts, 'formal_scoring_seconds': 5073.5,
                'nine_model_load_allowance_seconds': 27., 'estimated_seconds': 8250.75,
                'required_free_bytes': 1342379680, 'actual_free_bytes': 10**10,
                'safety_factor': 1.5, 'validation_allowance_seconds': 600, 'passed': True, 'remaining_seconds': 10000}
        self.assertEqual(v.verify_resource_gate(Memory(), policy, gate)['estimated_seconds'], 8250.75)
        bad = deepcopy(gate); bad['estimated_seconds'] -= 10
        with self.assertRaises(ValueError): v.verify_resource_gate(Memory(), policy, bad)
        bad = deepcopy(gate); bad['pilot_timing_inputs'].pop(next(iter(pilot_inputs)))
        with self.assertRaises(ValueError): v.verify_resource_gate(Memory(), policy, bad)


if __name__ == '__main__': unittest.main(verbosity=2)
