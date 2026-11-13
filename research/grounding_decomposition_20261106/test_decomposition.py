#!/usr/bin/env python3
"""CPU-only tests for new matching/oracle protocol and evidence completeness.

No torch, tokenizer/model download, SSH or GPU. Real processor/pixel equality is
an additional launch-time preflight, not simulated by these tests.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'src'))
from recipeqa.modeling import InputConfig,_render_recipe,build_messages,hash_json,read_jsonl,sha256_file
from recipeqa.metrics import score_order
from decomposition import (b_messages,oracle_table,parse_b,prepare_metadata,rebuild_visible_recipe,
                           score_b,score_oracle,source_mapping,validate_legal_b_format)
from run_decomposition import (CONDITIONS,DEV_SHA,MODEL_ORDER,estimate_remaining,metadata_receipt,summarize)
from run_decomposition import run_command
from run_decomposition import cpu_test_receipt,EXPECTED_CPU_TESTS
from launch_stage import remaining_seconds
from build_draft_policy import build


def example():
    # IDs deliberately start at zero and contain gaps. Correct visible steps
    # A/B/C/D=5/4/3/1, never backend source values8/6/5/0.
    return {'qid':'synthetic-cpu-boundary','recipe_id':'cpu-only','presentation_id':'p3210',
        'source_split':'val','manifest_set':'val','display_perm':[3,2,1,0],
        'target_order':[0,1,2,3],'target_labels':['D','C','B','A'],
        'steps':[{'source_step_id':n,'title':f'title{i}','body':'abcdefghijk'} for i,n in enumerate((0,2,5,6,8))],
        'source_step_ids_chronological':[0,5,6,8],
        'image_paths_display':['a.jpg','b.jpg','c.jpg','d.jpg']}


def answer(values=(5,4,3,1),order='D C B A'):
    return '\n'.join(f'{label}: step {value}' for label,value in zip('ABCD',values))+f'\nOrder: {order}'


class CharTokenizer:
    """Tiny deterministic boundary fixture; not an actual Qwen tokenizer."""
    def encode(self,text,add_special_tokens=False):return list(text)
    def decode(self,tokens,skip_special_tokens=True):return ''.join(tokens)


class WordTokenizer:
    def encode(self,text,add_special_tokens=False):return text.replace('\n',' ').split()


class MappingAndPromptTests(unittest.TestCase):
    def test_source_id_zero_and_gaps_map_to_visible_position(self):
        visible,chronological=source_mapping(example())
        self.assertEqual(visible,[5,4,3,1]);self.assertEqual(chronological,[1,3,4,5])

    def test_invalid_or_duplicate_source_mapping_is_data_error(self):
        row=example();row['steps'][1]['source_step_id']=0
        with self.assertRaises(ValueError):source_mapping(row)
        row=example();row['source_step_ids_chronological'][0]=99
        with self.assertRaises(ValueError):source_mapping(row)

    def test_test_or_confirmation_split_rejected(self):
        row=example();row['manifest_set']='confirmation'
        with self.assertRaises(ValueError):source_mapping(row)

    def test_metadata_target_must_round_trip(self):
        row=example();row['target_labels']=['A','B','C','D']
        with self.assertRaises(ValueError):source_mapping(row)

    def test_b_preserves_visible_recipe_and_four_images(self):
        recipe='Step 1: Keep this exact recipe.\n\nStep 2: 完整 UTF8.'
        messages=b_messages(recipe);first=messages[0]['content'][0]['text']
        self.assertEqual(first.split('Ordered recipe:\n',1)[1].encode(),recipe.encode())
        self.assertEqual(sum(x['type']=='image' for x in messages[0]['content']),4)
        self.assertNotIn('Output only four labels separated by spaces.',first)
        self.assertNotIn('source_step',str(messages));self.assertNotIn('.jpg',str(messages))

    def test_old_cap_rebuild_is_exact_and_hash_changes_fail(self):
        row=example();tok=CharTokenizer();recipe=_render_recipe(row['steps'],tok,3)
        messages=build_messages(recipe,process=False)
        audit={'uniform_body_cap':3,'recipe_tokens_used':len(recipe),'prompt_text_hash':hash_json(messages),
            'input_hash':hash_json({'messages':messages,'display_perm':row['display_perm'],
                'image_paths_display':row['image_paths_display'],'preprocessing':asdict(InputConfig())})}
        actual,original,b=rebuild_visible_recipe(row,audit,tok)
        self.assertEqual(actual,recipe);self.assertNotIn('defgh',actual)
        bad=deepcopy(audit);bad['uniform_body_cap']=4
        with self.assertRaises(ValueError):rebuild_visible_recipe(row,bad,tok)

    def test_legal_b_budget_is_checked_not_silently_extended(self):
        receipt=validate_legal_b_format(WordTokenizer(),14,64)
        self.assertLessEqual(receipt['plain_answer_plus_eos_tokens'],64)
        with self.assertRaises(ValueError):validate_legal_b_format(CharTokenizer(),14,64)
        with self.assertRaises(ValueError):validate_legal_b_format(WordTokenizer(),14,128)


class ScoringTests(unittest.TestCase):
    def test_full_correct_mapping_and_order(self):
        metrics=score_b(answer(),example())
        self.assertEqual(metrics['step_match_count'],4);self.assertEqual(metrics['mapping_pair_correct_count'],6)
        self.assertEqual(metrics['final_exact'],1);self.assertEqual(metrics['mapping_order_consistent'],1)

    def test_absolute_numbers_wrong_but_relative_order_correct(self):
        # A/B/C/D=4/3/2/1 all in range; only D is exact, but all6relations correct.
        metrics=score_b(answer((4,3,2,1)),example())
        self.assertEqual(metrics['step_match_count'],1);self.assertEqual(metrics['mapping_pair_correct_count'],6)
        self.assertEqual(metrics['derived_exact'],1)

    def test_out_of_range_pairs_count_wrong_fixed_six(self):
        metrics=score_b(answer((99,4,3,1)),example())
        self.assertEqual(metrics['out_of_range_count'],1);self.assertEqual(metrics['mapping_pair_correct_count'],3)
        self.assertEqual(metrics['mapping_pair'],.5);self.assertEqual(metrics['unique_mapping'],0)
        self.assertIsNone(metrics['derived_order']);self.assertEqual(metrics['final_exact'],1)

    def test_ties_are_wrong_and_never_broken_by_label(self):
        metrics=score_b(answer((4,4,3,1)),example())
        self.assertEqual(metrics['mapping_pair_correct_count'],5)
        self.assertEqual(metrics['in_range_tied_pair_count'],1);self.assertIsNone(metrics['derived_order'])
        self.assertEqual(metrics['mapping_order_comparable'],0)

    def test_unclear_and_missing_remain_denominator(self):
        metrics=score_b(answer(('unclear',4,3,1)),example())
        self.assertEqual(metrics['unclear_or_missing_count'],1)
        self.assertEqual(metrics['step_match_count'],3);self.assertEqual(metrics['mapping_pair_correct_count'],3)
        self.assertEqual(metrics['format_valid'],1)
        missing=score_b('\n'.join(answer().splitlines()[1:]),example())
        self.assertEqual(missing['format_valid'],0);self.assertEqual(missing['mapping_pair_correct_count'],3)

    def test_duplicate_field_cannot_cherry_pick_correct_number(self):
        metrics=score_b('A: step 1\n'+answer(),example())
        self.assertEqual(metrics['step_match_count'],3);self.assertEqual(metrics['format_valid'],0)
        self.assertTrue(metrics['parsed']['mapping']['A']['duplicate'])

    def test_format_and_final_order_are_independent(self):
        metrics=score_b('Some extra prose\n'+answer(),example())
        self.assertEqual(metrics['format_valid'],0);self.assertEqual(metrics['final_exact'],1)
        metrics=score_b(answer(order='A B C D'),example())
        self.assertEqual(metrics['all_steps_match'],1);self.assertEqual(metrics['final_exact'],0)
        self.assertEqual(metrics['mapping_order_comparable'],1);self.assertEqual(metrics['mapping_order_consistent'],0)

    def test_order_field_prose_and_duplicates_invalid(self):
        self.assertFalse(parse_b(answer(order='I think D C B A'))['order_valid'])
        self.assertFalse(parse_b(answer()+'\nOrder: D C B A')['order_valid'])
        self.assertFalse(parse_b(answer(order='D C B B'))['order_valid'])

    def test_cyclic_table_has_its_own_answer_and_no_shared_context(self):
        values,expected,c=oracle_table([5,4,3,1],'C')
        shifted,other,cp=oracle_table([5,4,3,1],'Cpi')
        self.assertEqual(shifted,[4,3,1,5]);self.assertNotEqual(expected,other)
        self.assertEqual(score_oracle('D C B A',expected)['exact'],1)
        self.assertEqual(score_oracle('D C B A',other)['exact'],0)
        self.assertEqual(len(c),len(cp),1);self.assertNotIn('image',cp[0]['content'][0]['text'].lower().replace('image labels',''))


class TimingAndCompletenessTests(unittest.TestCase):
    def test_cpu_gate_requires_exact_nonzero_count_and_clean_terminal_result(self):
        def completed(count=25,terminal='OK',code=0):
            return SimpleNamespace(stdout='',stderr=f'\nRan {count} tests in 0.1s\n\n{terminal}\n',returncode=code)
        good=cpu_test_receipt(completed(),EXPECTED_CPU_TESTS)
        self.assertTrue(good['passed']);self.assertEqual(good['tests_run'],25)
        self.assertEqual((good['skipped'],good['failures'],good['errors']),(0,0,0))
        for bad in (completed(0),completed(24),completed(26),completed(25,'OK (skipped=1)'),
                    completed(25,'FAILED (failures=1)',1),completed(25,'FAILED (errors=1)',1),
                    completed(25,'OK',1),SimpleNamespace(stdout='',stderr='',returncode=0),
                    SimpleNamespace(stdout='',stderr='Ran 25 tests in 0.1s\nRan 25 tests in 0.1s\nOK\n',returncode=0)):
            with self.subTest(stderr=bad.stderr):self.assertFalse(cpu_test_receipt(bad,EXPECTED_CPU_TESTS)['passed'])
        self.assertEqual(cpu_test_receipt(completed(25,'OK (skipped=1)'),25)['skipped'],1)
        self.assertEqual(cpu_test_receipt(completed(25,'FAILED (failures=1)',1),25)['failures'],1)
        self.assertEqual(cpu_test_receipt(completed(25,'FAILED (errors=1)',1),25)['errors'],1)

    @staticmethod
    def pilots():
        return [{'model':m,'condition':c,'cohort_index':i,'total_seconds':2+i}
                for m in MODEL_ORDER for i in range(2) for c in CONDITIONS]

    def test_gate_counts_240_remaining_and_reload_cost(self):
        gate=estimate_remaining(self.pilots(),[10,20])
        self.assertEqual(gate['remaining_predictions'],240)
        self.assertEqual(gate['components_seconds']['predictions'],30*8*3)
        self.assertEqual(gate['estimated_remaining_seconds'],1.5*(720+30)+120)

    def test_gate_rejects_missing_condition_bad_duration_or_replacement(self):
        with self.assertRaises(ValueError):estimate_remaining(self.pilots()[:-1],[10,20])
        bad=self.pilots();bad[0]['total_seconds']=float('nan')
        with self.assertRaises(ValueError):estimate_remaining(bad,[10,20])
        replaced=self.pilots();replaced[0]['cohort_index']=31
        with self.assertRaises(ValueError):estimate_remaining(replaced,[10,20])

    def test_outer_watchdog_uses_elapsed_original_clock(self):
        start=datetime(2026,11,6,8,0,tzinfo=timezone.utc)
        policy={'first_preflight_started_utc':start.isoformat(),
            'absolute_deadline_utc':(start+timedelta(minutes=30)).isoformat(),'wall_limit_seconds':1800}
        self.assertEqual(remaining_seconds(policy,start+timedelta(minutes=11)),19*60)
        with self.assertRaises(ValueError):remaining_seconds(policy,start+timedelta(minutes=30))
        with self.assertRaises(ValueError):remaining_seconds(policy,start-timedelta(seconds=1))

    def test_draft_refused_before_environment_or_model_access(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'draft.json';path.write_text(json.dumps({'status':'draft_not_authorized_for_execution'}))
            with self.assertRaisesRegex(ValueError,'DRAFT ONLY'):
                run_command(SimpleNamespace(policy=str(path)))

    @staticmethod
    def complete_rows():
        result=[];record=example()
        for model in MODEL_ORDER:
            for i in range(32):
                for c in CONDITIONS:
                    metrics=score_order('D C B A',record['display_perm'],record['target_order']) if c=='A' else score_b(answer(),record) if c=='B' else score_oracle('D C B A',oracle_table([5,4,3,1],c)[1])
                    result.append({'model':model,'condition':c,'cohort_index':i,'qid':f'q{i}','metrics':metrics,
                        'body_clipped':i<3,'historical_A_raw_equal':True,'historical_A_parsed_equal':True,
                        'completion_tokens':5,'reached_token_limit':False,'total_seconds':1,'technical_error':None})
        return result

    def test_full_summary_uses_fixed_denominators_and_clipping_strata(self):
        summary=summarize(self.complete_rows())
        result=summary['results']['sft44']
        self.assertEqual(result['B']['source_step_match'],1)
        self.assertEqual(result['B']['mapping_order_comparable_n'],32)
        self.assertEqual(result['body_clipping_strata']['clipped']['n'],3)
        self.assertEqual(result['body_clipping_strata']['unclipped']['n'],29)

    def test_incomplete_duplicate_or_technical_failure_never_overall_summary(self):
        rows=self.complete_rows()
        with self.assertRaises(ValueError):summarize(rows[:-1])
        duplicate=deepcopy(rows);duplicate[-1]=duplicate[0]
        with self.assertRaises(ValueError):summarize(duplicate)
        failed=deepcopy(rows);failed[0]['technical_error']='encoding failed'
        with self.assertRaises(ValueError):summarize(failed)

    def test_known_development_cohort_and_historical_inputs_match(self):
        data_root=ROOT if (ROOT/'data/manifests/val.jsonl').is_file() else ROOT/'.state/remote_principled_20261105'
        manifest=data_root/'data/manifests/val.jsonl'
        self.assertEqual(sha256_file(manifest),DEV_SHA)
        prefix=data_root/'exploratory/principled_ordering_20261105/finite_trials_v2/flow44'
        matched=prepare_metadata(read_jsonl(manifest),{
            'sft44':read_jsonl(prefix/'raw_sft/dev64/predictions.jsonl'),
            'exactkl44':read_jsonl(prefix/'exact_kl/dev64/predictions.jsonl')})
        receipt=metadata_receipt(matched)
        self.assertEqual(receipt['total_new_calls'],256)
        self.assertEqual(sum(item['body_clipped'] for item in receipt['cohort']),3)
        self.assertEqual(receipt['cohort'][0]['gold_visible_steps'],[8,6,5,1])

    def test_builder_remains_draft_and_binds_whole_execution_sources(self):
        data_root=ROOT if (ROOT/'data/manifests/val.jsonl').is_file() else ROOT/'.state/remote_principled_20261105'
        rows=read_jsonl(data_root/'data/manifests/val.jsonl')[:32]
        # Synthetic digest fixture only; no image-byte proof is claimed. Runtime
        # must independently hash actual remote bytes, so these cannot execute.
        image_fixture={path:'0'*64 for row in rows for path in row['image_paths_display']}
        draft=build(data_root,ROOT,image_fixture)
        self.assertEqual(draft['status'],'draft_not_authorized_for_execution')
        self.assertIsNone(draft['first_preflight_started_utc']);self.assertIsNone(draft['budget_authorization'])
        self.assertEqual(draft['new_calls'],256);self.assertEqual(draft['remaining_calls_after_pilot'],240)
        self.assertIn('research/grounding_decomposition_20261106/launch_stage.py',draft['sources_sha256'])


if __name__=='__main__':
    require_no_torch='torch' not in sys.modules
    if not require_no_torch:raise RuntimeError('CPU metadata suite unexpectedly imported torch')
    unittest.main(verbosity=2)
