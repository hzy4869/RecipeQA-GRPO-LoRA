"""Adversarial CPU-only scope, timing, authorization and no-update boundaries."""
from __future__ import annotations
import ast
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import tempfile
import unittest
from endpoint_bank_contract import *


def policy_fixture():
    initial='exploratory/principled_ordering_20261105/full_sft/train/checkpoint-768'
    endpoints={}
    for key in ENDPOINTS:
        path=initial if key=='initial_sft768' else 'exploratory/coverage_20261106/stage_v1/'+key.removesuffix('_256')+'/train/checkpoint-256'
        endpoints[key]={'adapter_path':path,'weight_identity':{'base':{'directory':str(BASE/'models/Qwen3-VL-2B-Instruct'),'weight_files':[{'name':'model.safetensors','bytes':1,'sha256':'0'*64}]},
            'adapter':{'directory':str(BASE/path),'weight_files':[{'name':'adapter_model.safetensors','bytes':1,'sha256':'1'*64}]}}}
    source={n:'reference/'+n for n in ('source_cache','source_config','source_completion','source_manifest','source_signature','source_audit')}
    source.update(rows=256,output_directory='cached_initial_train')
    source['source_receipt_sha256']={source[k]:'0'*64 for k in ('source_cache','source_config','source_completion','source_manifest','source_signature','source_audit')}
    return {'status':'draft_endpoint_bank','wall_limit_seconds':10800,'budget_cny_cap':4.8,'reference_cny_per_hour':1.6,
        'first_preflight_started_utc':None,'absolute_deadline_utc':None,'output_directory':'exploratory/endpoint_bank_20261106/stage_v1',
        'model_path':'models/Qwen3-VL-2B-Instruct','image_root':'data/images','revision':REVISION,'beta':BETA,'input':deepcopy(INPUT),
        'required_versions':{'python':'synthetic_cpu_fixture_only'},
        'endpoints':endpoints,'banks':{b:{'manifest':'prepared/'+b+'.jsonl','manifest_sha256':'0'*64,'selected_records_sha256':'1'*64,'rows':n,
            'source_manifest':FULL_TRAIN if b=='train256' else DEV,'source_manifest_sha256':FULL_SHA if b=='train256' else DEV_SHA} for b,n in BANKS.items()},
        'tasks':fixed_tasks(),'formal_rows':2024,'pilot_rows':18,'pilot_rows_per_task':2,'cached_initial_train':source,
        'candidate_policy':'canonical24_full_sequence_plusEOS_temperature1','tie_policy':'exact_maximum_then_lowest_canonical_index',
        'expected_cpu_tests':1,'sources_sha256':{k:OLD_SOURCES.get(k,'0'*64) for k in SOURCE_PATHS},
        'artifacts_sha256':{HISTORICAL_OUTCOME:'0'*64},
        'historical_timing':{'reference_outcome':HISTORICAL_OUTCOME,'sha256':'0'*64,'rows':1024,'elapsed_seconds':2485.4406706159934},
        'timing_gate_policy':'first_row_once_remaining_second_row_with_historical_floor_1.5_plus600',
        'test_access':False,'confirmation_access':False,'resume':False,'automatic_retry':False,'optimizer_steps':0,'backward_calls':0,'generation_calls':0}


def records_fixture(bank):
    return [{'qid':str(i),'presentation_id':str(i)+'p','recipe_id':str(i%(254 if bank=='train256' else 200)),
        'source_split':'train' if bank=='train256' else 'val','manifest_set':'train_all7558' if bank=='train256' else 'val',
        'technical_status':'ok','image_paths_display':['a','b','c','d'],'display_perm':[0,1,2,3],
        'target_order':[0,1,2,3],'target_labels':['A','B','C','D']} for i in range(BANKS[bank])]


class ContractTests(unittest.TestCase):
    def reject(self,mutate):
        p=policy_fixture();mutate(p)
        with self.assertRaises((ValueError,KeyError,TypeError)):validate_policy(p,locked=False)

    def test_fixed_tasks_accounting(self):
        t=fixed_tasks();self.assertEqual(len(t),9);self.assertEqual(sum(x['rows'] for x in t),2024)
        self.assertNotIn('initial_sft768__train256',[x['id'] for x in t])

    def test_candidates_all24_no_gold_filter(self):
        self.assertEqual(len(CANDIDATES),len(set(CANDIDATES)));self.assertEqual(len(CANDIDATES),24)
        self.assertTrue(all(set(x.split())==set('ABCD') for x in CANDIDATES))

    def test_complete_draft_accepted(self):validate_policy(policy_fixture(),locked=False)

    def test_draft_rejected_as_execution(self):
        with self.assertRaises(ValueError):validate_policy(policy_fixture())

    def test_missing_endpoint_rejected(self):self.reject(lambda p:p['endpoints'].pop('flow45_exact_kl_256'))
    def test_extra_endpoint_rejected(self):self.reject(lambda p:p['endpoints'].update(other=p['endpoints']['initial_sft768']))
    def test_missing_bank_rejected(self):self.reject(lambda p:p['banks'].pop('train256'))
    def test_short_task_list_rejected(self):self.reject(lambda p:p['tasks'].pop())
    def test_different_task_order_rejected(self):self.reject(lambda p:p['tasks'].reverse())
    def test_test_access_rejected(self):self.reject(lambda p:p.update(test_access=True))
    def test_confirmation_access_rejected(self):self.reject(lambda p:p.update(confirmation_access=True))
    def test_any_update_or_generation_rejected(self):
        for k in ('optimizer_steps','backward_calls','generation_calls'):
            with self.subTest(field=k):self.reject(lambda p:p.update({k:1}))

    def test_different_beta_rejected(self):self.reject(lambda p:p.update(beta=1.0))
    def test_input_budget_change_rejected(self):self.reject(lambda p:p['input'].update(max_input_tokens=2048))
    def test_checkpoint64_rejected(self):self.reject(lambda p:p['endpoints']['flow44_raw_sft_256'].update(adapter_path='bad/checkpoint-64'))
    def test_missing_cache_receipt_rejected(self):self.reject(lambda p:p['cached_initial_train']['source_receipt_sha256'].pop('reference/source_audit'))
    def test_source_closure_cannot_shrink(self):self.reject(lambda p:p['sources_sha256'].pop(SOURCE_PATHS[0]))
    def test_historical_scoring_source_frozen(self):self.reject(lambda p:p['sources_sha256'].update({next(iter(OLD_SOURCES)):'0'*64}))

    def test_original_train_metadata_accepted(self):validate_records(records_fixture('train256'),'train256')
    def test_original_dev_metadata_accepted(self):validate_records(records_fixture('dev200'),'dev200')
    def test_test_row_rejected_regardless_of_bankname(self):
        r=records_fixture('dev200');r[0]['source_split']='test'
        with self.assertRaises(ValueError):validate_records(r,'dev200')

    def test_confirmation_metadata_rejected(self):
        r=records_fixture('dev200');r[0]['manifest_set']='confirmation'
        with self.assertRaises(ValueError):validate_records(r,'dev200')

    def test_missing_and_duplicate_rows_rejected(self):
        r=records_fixture('train256')
        with self.assertRaises(ValueError):validate_records(r[:-1],'train256')
        r[1]=deepcopy(r[0])
        with self.assertRaises(ValueError):validate_records(r,'train256')

    def test_paths_and_image_traversal_rejected(self):
        for bad in ('/tmp/file','../../system'):
            with self.assertRaises(ValueError):confined(bad)
        for bad in ('../../file','/root/file'):
            with self.assertRaises(ValueError):image_path('data/images',bad)

    def test_one_shared_exact_clock(self):
        t=datetime(2026,11,6,tzinfo=timezone.utc);p={'first_preflight_started_utc':t.isoformat(),'absolute_deadline_utc':(t+timedelta(seconds=10800)).isoformat()}
        self.assertEqual(remaining_seconds(p,at=t+timedelta(seconds=900)),9900)
        p['absolute_deadline_utc']=(t+timedelta(seconds=10801)).isoformat()
        with self.assertRaises(ValueError):remaining_seconds(p,at=t)

    def test_expired_and_future_clock_rejected(self):
        t=datetime(2026,11,6,tzinfo=timezone.utc);p={'first_preflight_started_utc':t.isoformat(),'absolute_deadline_utc':(t+timedelta(seconds=10800)).isoformat()}
        for at in (t-timedelta(seconds=1),t+timedelta(seconds=10800)):
            with self.assertRaises(ValueError):remaining_seconds(p,at=at)

    def test_gate_requires_all_nine_pilots(self):
        with self.assertRaises(ValueError):whole_gate({},fixed_tasks(),10000,10**10,policy_fixture()['historical_timing'])

    def test_gate_charges_separate_pilot_and_all_formal_work(self):
        t=fixed_tasks();r={x['id']:{'passed':True,'rows':2,'row_seconds':[2.,3.],'command_wall_seconds':20.,'raw_bytes':20000} for x in t}
        history=policy_fixture()['historical_timing']
        g=whole_gate(r,t,10000,10**10,history);self.assertTrue(g['passed']);self.assertEqual(g['formal_scoring_seconds'],2024*3-9)
        self.assertFalse(whole_gate(r,t,1,10**10,history)['passed']);self.assertFalse(whole_gate(r,t,10000,1,history)['passed'])

    def test_first_literal_row_is_charged_once(self):
        t=fixed_tasks();r={x['id']:{'passed':True,'rows':2,'row_seconds':[100.,3.],'command_wall_seconds':120.,'raw_bytes':20000} for x in t}
        g=whole_gate(r,t,20000,10**10,policy_fixture()['historical_timing'])
        self.assertEqual(g['formal_scoring_seconds'],9*100+(2024-9)*3)

    def test_historical_floor_prevents_optimistic_second_row(self):
        t=fixed_tasks();r={x['id']:{'passed':True,'rows':2,'row_seconds':[3.,.1],'command_wall_seconds':20.,'raw_bytes':20000} for x in t}
        history=policy_fixture()['historical_timing'];g=whole_gate(r,t,20000,10**10,history)
        self.assertAlmostEqual(g['formal_scoring_seconds'],9*3+(2024-9)*history['elapsed_seconds']/1024)
        self.assertEqual(set(g['pilot_timing_inputs']),{x['id'] for x in t})

    def test_scorer_has_no_generate_backward_or_optimizer_call(self):
        tree=ast.parse((HERE/'score_endpoint_bank.py').read_text());calls=[n.func for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in ('generate','backward','step') for n in calls))
        self.assertNotIn('torch.optim',(HERE/'score_endpoint_bank.py').read_text())

    def test_no_detached_scientific_children(self):
        source=(HERE/'run_endpoint_bank.py').read_text();self.assertNotIn('start_new_session',source);self.assertNotIn('setsid',source)

    def test_authorization_missing_rejects_even_known_task(self):
        with tempfile.TemporaryDirectory() as d:
            p=policy_fixture();path=Path(d)/'policy.json';path.write_text('{}')
            with self.assertRaises(FileNotFoundError):validate_authorization(p,path,p['tasks'][0]['id'],'pilot',Path(d))

    def make_authorization_fixture(self,root,phase='pilot'):
        p=policy_fixture();tick=datetime.now(timezone.utc)-timedelta(seconds=60)
        p.update(first_preflight_started_utc=tick.isoformat(),absolute_deadline_utc=(tick+timedelta(seconds=10800)).isoformat())
        policy=root/'policy.json';policy.write_text(json.dumps(p));out=root/p['output_directory']
        def put(name,value):
            q=out/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(value));return q
        cpu=put('cpu_preflight.json',{'passed':True,'tests':1,'expected_tests':1,'failures':0,'errors':0,'skipped':0,
            'expected_failures':0,'unexpected_successes':0,'model_gpu_calls':0,'torch_imported':False,
            'native_watchdog_probe':{'passed':True},'completed_utc':(tick+timedelta(seconds=1)).isoformat(),
            'policy_sha256':sha256_file(policy),'sources_sha256':p['sources_sha256'],
            'first_preflight_started_utc':p['first_preflight_started_utc'],'absolute_deadline_utc':p['absolute_deadline_utc']})
        cache=put('cached_initial_train/independent_verification.json',{'passed':True,'rows':256,
            'phase':'cached','task_id':'initial_sft768__train256','policy_sha256':sha256_file(policy),
            'verifier_sha256':p['sources_sha256']['research/endpoint_bank_20261106/verify_endpoint_scores.py'],
            'completed_utc':(tick+timedelta(seconds=1)).isoformat()})
        task=p['tasks'][0];authorization={'policy_sha256':sha256_file(policy),'task_id':task['id'],'phase':phase,
            'rows':2 if phase=='pilot' else task['rows'],'pgid':os.getpgrp(),'issued_utc':(tick+timedelta(seconds=2)).isoformat(),
            'cpu_preflight_sha256':sha256_file(cpu),'cached_initial_train_audit_sha256':sha256_file(cache),'whole_gate_sha256':None}
        path=put('authorizations/'+phase+'_'+task['id']+'.json',authorization)
        return p,policy,task,path,cpu,cache

    def test_pilot_requires_actual_cpu_and_cache_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p,policy,t,path,cpu,cache=self.make_authorization_fixture(root)
            self.assertEqual(validate_authorization(p,policy,t['id'],'pilot',root),path.resolve())
            data=json.loads(cpu.read_text());data['skipped']=1;cpu.write_text(json.dumps(data))
            with self.assertRaises(ValueError):validate_authorization(p,policy,t['id'],'pilot',root)

    def test_formal_cannot_load_model_before_nine_task_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p,policy,t,path,_,_=self.make_authorization_fixture(root,'formal')
            with self.assertRaises(FileNotFoundError):validate_authorization(p,policy,t['id'],'formal',root)

    def test_authorization_cannot_predate_cached_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p,policy,t,path,cpu,cache=self.make_authorization_fixture(root)
            a=json.loads(path.read_text());c=json.loads(cache.read_text());c['completed_utc']=(stamp(a['issued_utc'])+timedelta(seconds=1)).isoformat();cache.write_text(json.dumps(c))
            a['cached_initial_train_audit_sha256']=sha256_file(cache);path.write_text(json.dumps(a))
            with self.assertRaises(ValueError):validate_authorization(p,policy,t['id'],'pilot',root)


if __name__=='__main__':unittest.main()
