"""Fail-closed complete-stage and descriptive endpoint-analysis guards."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_endpoint_bank as a
import bank_math as m
import verify_endpoint_scores as v
from test_verify_endpoint_scores import policy_fixture


def complete_fixture():
    policy = policy_fixture()
    state = {'status': 'completed_endpoint_bank_zero_update', 'active_child_pid': None,
             'started_utc': '2026-11-07T00:00:01Z', 'updated_utc': '2026-11-07T02:00:00Z',
             'first_preflight_started_utc': policy['first_preflight_started_utc'],
             'absolute_deadline_utc': policy['absolute_deadline_utc'], 'pgid': 123,
             'commands': [{'stage': name, 'returncode': 0, 'pgid': 123,
                           'started_utc': '2026-11-07T00:01:00Z', 'finished_utc': '2026-11-07T01:00:00Z'}
                          for name in sorted(a.required_commands())],
             'formal_tasks': 9, 'pilot_tasks': 9, 'formal_rows': 2024, 'pilot_rows': 18,
             'cached_initial_train_rows': 256, 'optimizer_steps': 0, 'backward_calls': 0, 'generation_calls': 0,
             'candidate_forwards': 51050, 'model_loads': 18,
             'automatic_retry': False, 'resume': False, 'test_access': False, 'confirmation_access': False}
    return policy, state


class AnalysisGuards(unittest.TestCase):
    def test_complete38_command_gate_rejects_missing_failed_detached_or_late_work(self):
        policy, state = complete_fixture(); a.complete_gate(policy, state)
        for change in ('missing', 'duplicate', 'failed', 'detached', 'active', 'after_commit', 'cached_count'):
            bad = deepcopy(state)
            if change == 'missing': bad['commands'].pop()
            if change == 'duplicate': bad['commands'][0] = deepcopy(bad['commands'][1])
            if change == 'failed': bad['commands'][0]['returncode'] = 1
            if change == 'detached': bad['commands'][0]['pgid'] = 124
            if change == 'active': bad['active_child_pid'] = 999
            if change == 'after_commit': bad['commands'][0]['finished_utc'] = '2026-11-07T02:01:00Z'
            if change == 'cached_count': bad['cached_initial_train_rows'] = 255
            with self.subTest(change=change), self.assertRaises(ValueError): a.complete_gate(policy, bad)

    def test_partial_stage_refuses_before_any_new_scores_or_greedy_read(self):
        policy = policy_fixture(); policy['output_directory'] = 'exploratory/endpoint_bank_20261106/stage_v1'
        class OnlyPolicyState:
            def __init__(self): self.reads = []
            def json(self, path):
                self.reads.append(str(path))
                return policy if str(path) == 'policy.json' else {'status': 'still_running'}
            def sha(self, path): return 'fake-policy'
        files = OnlyPolicyState()
        with self.assertRaisesRegex(ValueError, 'Incomplete endpoint bank'): a.analyze_all(files, 'policy.json')
        self.assertEqual(files.reads, ['policy.json', policy['output_directory'] + '/state.json'])

    def test_empty_stratum_is_retained_not_dropped_or_divided_by_zero(self):
        result = a.describe([])
        self.assertEqual(result['n_questions'], 0); self.assertEqual(result['MAP_correct'], 0)
        self.assertIsNone(result['Jstar_minus_J'])
        self.assertEqual(len(a.STRATA), 3)

    def test_qstar_MAP_is_not_accuracy_upper_bound(self):
        from test_bank_math import policy
        reference = policy([2.] + [0.] * 23); actual = policy([0., 4.] + [0.] * 22)
        result = m.metrics(actual, reference, 1)
        self.assertEqual(result['qstar_map']['map_exact'], 0)
        self.assertEqual(result['map']['map_exact'], 1)
        self.assertGreater(result['Jstar_minus_J'], 0)

    def test_realization_uses_ratio_of_sums_not_tiny_per_question_denominators(self):
        from test_bank_math import policy
        reference = policy([0.] * 24)
        row = m.metrics(reference, reference, 0); row['recipe_id'] = 'synthetic'
        first = deepcopy(row); first.update(available_objective_gain=1e-12, achieved_objective_gain=-1e-3)
        second = deepcopy(row); second.update(available_objective_gain=1., achieved_objective_gain=.1)
        summary = a.describe([first, second])['objective_gain_realization']
        self.assertAlmostEqual(summary['ratio_of_sums_not_mean_per_question_ratio'], .099 / (1 + 1e-12))
        first.update(available_objective_gain=0., achieved_objective_gain=-.5)
        summary = a.describe([first])['objective_gain_realization']
        self.assertIsNone(summary['ratio_of_sums_not_mean_per_question_ratio'])
        self.assertEqual(summary['sum_achieved'], -.5)

    def test_remote_physical_claim_requires_core_and_all_locked_weight_receipts(self):
        directory = Path('exploratory/endpoint_bank_20261106/fixture/formal/task')
        identity = {'base': {'directory': str(v.REMOTE / 'models/base'),
                             'weight_files': [{'name': 'model.safetensors', 'sha256': 'a' * 64, 'bytes': 10}]},
                    'adapter': {'directory': str(v.REMOTE / 'adapters/a'),
                                'weight_files': [{'name': 'adapter_model.safetensors', 'sha256': 'b' * 64, 'bytes': 5}]}}
        signed = {'payload': {'pixel_sha256': {'p.jpg': 'c' * 64}, 'weight_identity': identity}}
        checked = {'phase': 'formal', 'task_id': 'task', 'rows': 2, 'signature_sha256': 'sig', 'output_sha256': 'out',
                   'candidate_forwards': 50, 'optimizer_steps': 0, 'backward_calls': 0, 'generation_calls': 0, 'policy_sha256': 'policy'}
        receipt = dict(checked, passed=True, verifier_sha256='verifier', physical_pixels_and_model_files_verified=True)
        policy = {'sources_sha256': {'research/endpoint_bank_20261106/verify_endpoint_scores.py': 'verifier'}, 'image_root': 'data/images'}
        class Memory:
            def __init__(self): self.report = deepcopy(receipt)
            def json(self, path): return signed if Path(path).name == 'signature.json' else self.report
            def relative(self, path):
                p = Path(path); return p.relative_to(v.REMOTE) if p.is_absolute() else p
        for items in ({}, {'policy.json': {'sha256': 'x', 'bytes': 1}}):
            f = Memory(); f.report.update(input_receipts=items, input_files_verified=len(items))
            with self.assertRaisesRegex(ValueError, 'omits core'):
                a.bind_remote_verification(f, policy, directory, checked, 'policy.json')

    def test_old_real_dev_greedy_matches200_and_raw_sampling_change_is_rejected(self):
        project = Path(__file__).resolve().parents[2]
        name = 'exploratory/principled_ordering_20261105/full_sft/dev768/predictions.jsonl'
        root = project if (project / name).is_file() else project / '.state/remote_principled_20261105'
        files = v.Files(root); config = files.json(name + '.config.json'); raw = files.rows(name)
        records = files.rows('data/manifests/val.jsonl')
        by_key = {(r['qid'], r['presentation_id']): r for r in raw}
        parsed = [{'record': r, 'row': by_key[(r['qid'], r['presentation_id'])]} for r in records]
        policy = {'endpoints': {'initial_sft768': {'weight_identity': config['weight_identity']}},
                  'dev_greedy': {'initial_sft768': {'predictions': name, 'config': name + '.config.json'}},
                  'artifacts_sha256': {name: files.sha(name), name + '.config.json': files.sha(name + '.config.json')},
                  'banks': {'dev200': {'source_manifest_sha256': files.sha('data/manifests/val.jsonl')}},
                  'input': config['input']}
        result = a.existing_greedy(files, policy, 'initial_sft768', records, parsed)
        self.assertEqual((len(result), sum(x['exact'] for x in result.values())), (200, 143))
        class Mutated:
            def json(self, path): return files.json(path)
            def rows(self, path):
                rows = deepcopy(files.rows(path)); rows[0]['decoding']['do_sample'] = True; return rows
        with self.assertRaises(ValueError): a.existing_greedy(Mutated(), policy, 'initial_sft768', records, parsed)


if __name__ == '__main__': unittest.main(verbosity=2)
