"""Meaningful CPU checks for coordinate leakage, fixed cohort and scoring gates."""
from copy import deepcopy
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import unittest
from datetime import datetime, timedelta, timezone
import signal
import time

spec = spec_from_file_location('orbit_diagnostic', Path(__file__).with_name('orbit_diagnostic.py'))
m = module_from_spec(spec)
spec.loader.exec_module(m)


def source():
    rows = []
    for i in range(200):
        display, target = [2, 0, 3, 1], [1, 3, 0, 2]
        paths = [f'{i}_{j}.jpg' for j in range(4)]
        rows.append({'qid': f'q{i}', 'recipe_id': f'r{i}', 'presentation_id': f'q{i}::p2031',
                     'source_split': 'val', 'manifest_set': 'val', 'display_perm': display, 'target_order': target,
                     'target_labels': [chr(65 + display.index(s)) for s in target],
                     'image_paths_display': [paths[s] for s in display], 'image_paths_chronological': paths,
                     'image_status_display': ['ok']*4, 'image_status_chronological': ['ok']*4,
                     'steps': [{'body': 'preserve recipe text'}]})
    return rows


class OrbitTests(unittest.TestCase):
    def test_full_orbit_nonidentity_target_and_image_roundtrip(self):
        rows = source()
        before = deepcopy(rows)
        orbit = m.make_orbit(rows)
        self.assertEqual(rows, before)
        self.assertEqual(len(orbit), 384)
        self.assertEqual({r['qid'] for r in orbit}, {f'q{i}' for i in range(16)})
        for row in orbit:
            self.assertEqual(row['target_order'], [1, 3, 0, 2])
            self.assertEqual(row['steps'], rows[0]['steps'])
            self.assertEqual(m.independent_score(' '.join(row['target_labels']), row['display_perm'], row['target_order'])['exact'], 1)
            self.assertEqual(row['image_paths_display'], [row['image_paths_chronological'][s] for s in row['display_perm']])

    def test_reject_wrong_original_gold_and_image_map(self):
        for field, value in [('target_labels', ['A','B','C','D']), ('image_paths_chronological', ['x','y','z','w'])]:
            rows = source()
            rows[0][field] = value
            with self.assertRaises(RuntimeError):
                m.make_orbit(rows)

    def test_reject_alternate_cohort_or_repeated_recipe(self):
        with self.assertRaises(RuntimeError):
            m.make_orbit(source()[:16])
        rows = source()
        rows[1]['recipe_id'] = rows[0]['recipe_id']
        with self.assertRaises(RuntimeError):
            m.make_orbit(rows)

    def test_independent_parser_and_six_integer_relations_match_frozen(self):
        from recipeqa.metrics import score_order
        for display in m.PERMS:
            for output_order in m.PERMS:
                response = ' '.join(chr(65+p) for p in output_order)
                a = m.independent_score(response, display, [1,3,0,2])
                b = score_order(response, display, [1,3,0,2])
                for key in ('exact', 'valid', 'pair', 'semantic_order'):
                    self.assertEqual(a[key], b[key])
        for response in ('Answer: A,B,C,D', 'ABCD', 'a b c d', 'ABCA', 'A B C', 'A B C D.', 'reason: A B C D', '', None):
            a = m.independent_score(response, [2,0,3,1], [1,3,0,2])
            b = score_order(response, [2,0,3,1], [1,3,0,2])
            for key in ('exact','valid','pair','semantic_order'):
                self.assertEqual(a[key], b[key])

    def test_consistently_wrong_semantic_order_is_not_correct(self):
        orbit = m.make_orbit(source())
        raw = []
        for row in orbit:
            wrong_semantic = [0,1,2,3]
            response = ' '.join(chr(65+row['display_perm'].index(s)) for s in wrong_semantic)
            raw.append({**{k: row[k] for k in ('qid','recipe_id','presentation_id','display_perm','target_order','target_labels')},
                        'raw_response': response, 'score': __import__('recipeqa.metrics', fromlist=['score_order']).score_order(response, row['display_perm'], row['target_order']),
                        'model_weights_sha256': 'known', 'prompt_kind': 'answer', 'sample_index': 0,
                        'decoding': {'do_sample': False,'max_new_tokens':32}, 'technical_error':None})
        scored = m.rescore(orbit, raw, 'known')
        summary = m.summarize(scored)
        self.assertEqual(summary['recipe_macro']['semantic_modal_consistency'], 1)
        self.assertEqual(summary['recipe_macro']['K_over24'], 0)
        self.assertEqual(summary['recipe_macro']['all24_exact'], 0)
        with self.assertRaises(RuntimeError):
            m.rescore(orbit, raw[:-1], 'known')
        with self.assertRaises(RuntimeError):
            m.rescore(orbit, raw + [raw[0]], 'known')
        raw[0]['technical_error'] = 'injected technical failure'
        with self.assertRaises(RuntimeError):
            m.rescore(orbit, raw, 'known')

    def test_deadline_uses_earlier_cap_and_rejects_expiry(self):
        now = datetime(2026,11,6,tzinfo=timezone.utc)
        self.assertEqual(m.remaining_budget(200, now+timedelta(seconds=50), monotonic_now=100, utc_now=now), 50)
        self.assertEqual(m.remaining_budget(120, now+timedelta(seconds=50), monotonic_now=100, utc_now=now), 20)
        for phase, absolute in [(100, now+timedelta(seconds=50)), (200, now)]:
            with self.assertRaises(RuntimeError):
                m.remaining_budget(phase, absolute, monotonic_now=100, utc_now=now)

    def test_normal_termination_raises_and_restores_handlers(self):
        before = {number:signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGHUP)}
        previous = m.install_termination_handlers()
        try:
            for number in before:
                self.assertTrue(callable(signal.getsignal(number)))
            with self.assertRaises(InterruptedError):
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            for number in before:
                self.assertEqual(signal.getsignal(number), signal.SIG_IGN)
        finally:
            m.restore_termination_handlers(previous)
        for number, handler in before.items():
            self.assertEqual(signal.getsignal(number), handler)

    def test_real_short_budget_alarm_interrupts_cpu_wait(self):
        before = signal.getsignal(signal.SIGALRM)
        previous = m.install_budget_alarm(.01)
        try:
            with self.assertRaises(TimeoutError):
                time.sleep(.2)
        finally:
            m.cancel_budget_alarm(previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))
        self.assertEqual(signal.getsignal(signal.SIGALRM), before)

    def test_timing_gate_requires_four_complete_pilots(self):
        entries = [{'wall_seconds':12, 'prediction_emissions':[{'seconds_since_start':9+i} for i in range(4)]}]*4
        rows = [[{'generation_seconds':1}]*4 for _ in range(4)]
        estimate = m.estimate_remaining(entries, rows)
        self.assertEqual(estimate['estimated_seconds'], 1.5*(36+1536)+300)
        with self.assertRaises(RuntimeError):
            m.estimate_remaining(entries[:3], rows[:3])


if __name__ == '__main__':
    unittest.main()
