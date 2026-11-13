"""CPU policy/budget/coverage checks; no SSH, GPU, training or actual data reads."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import projection_pipeline as p


def stage():
    return {'status':'locked_before_projection_stage','wall_limit_seconds':14400,'method':'forward_projection',
       'flow_order':['flow44','flow45'],'primary_checkpoint_step':64,'dev_manifest':'data/manifests/val.jsonl',
       'dev_manifest_sha256':p.DEV_SHA,'orbit_first_recipes':16,'orbit_layouts':24,
       'selection':'fixed_endpoint64_no_development_selection','test_access':False,'confirmation_access':False,
       'first_preflight_started_utc':'2026-11-06T10:00:00+00:00','absolute_deadline_utc':'2026-11-06T14:00:00+00:00',
       'child_policies':{'smoke44':{},'formal44':{},'formal45':{}},'historical':{arm:{} for arm in p.HISTORICAL_ARMS}}


def timing_commands():
    return [{'stage':'formal_'+arm,'returncode':0,
             'prediction_emissions':[{'seconds_since_start':10+i*.8} for i in range(384)]}
            for arm in p.HISTORICAL_ARMS]


class ProjectionPipelineTests(unittest.TestCase):
    def test_stage_rejects_draft_overbudget_and_selection_changes(self):
        now=datetime(2026,11,6,10,0,tzinfo=timezone.utc)
        p.validate_stage_policy(stage(),utc_now=now)
        for field,value in [('status','draft'),('wall_limit_seconds',14401),('primary_checkpoint_step',48),
            ('selection','best_dev'),('confirmation_access',True),('flow_order',['flow44'])]:
            value_stage=stage(); value_stage[field]=value
            with self.assertRaises(RuntimeError): p.validate_stage_policy(value_stage,utc_now=now)
        value_stage=stage(); value_stage['absolute_deadline_utc']='2026-11-06T14:00:01+00:00'
        with self.assertRaises(RuntimeError): p.validate_stage_policy(value_stage,utc_now=now)
        with self.assertRaises(RuntimeError): p.validate_stage_policy(stage(),utc_now=now-timedelta(seconds=1))

    def test_smoke_and_formal_require_distinct_exposure_and_fresh_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)
            child={'status':'locked_before_projection_training','method':'forward_projection','reward':'unique_complete_order_exact_0_1',
                'mode':'formal','cli':{'mode':'formal','seed':44,'beta':p.BETA,'learning_rate':5e-6,
                    'limit':256,'max_steps':64,'save_steps':16,'output_dir':str(dest/'flow44/train')}}
            p.validate_child_policy(child,'formal44',dest)
            bad=deepcopy(child); bad['cli']['limit']=4
            with self.assertRaises(RuntimeError): p.validate_child_policy(bad,'formal44',dest)
            smoke=deepcopy(child); smoke['mode']=smoke['cli']['mode']='smoke'
            smoke['cli'].update(limit=4,max_steps=1,save_steps=1,output_dir=str(dest/'smoke44/train'))
            p.validate_child_policy(smoke,'smoke44',dest)
            (dest/'smoke44/train').mkdir(parents=True)
            with self.assertRaises(RuntimeError): p.validate_child_policy(smoke,'smoke44',dest)

    def test_historical_control_must_match_start_exposure_flow_and_reference(self):
        cli={'method':'exact_kl','manifest':'train.jsonl','expected_manifest_sha256':'manifesthash','seed':44,
             'limit':256,'max_steps':64,'learning_rate':5e-6}
        payload={'method':'exact_kl','cli':cli,'initial_identity':{'same':'start'},'actual_updates':64,
                 'accumulation':4,'resolved_beta':p.BETA,'reference_policy_sha256':'referencehash'}
        signature=p.hash_json(payload)
        signed={'sha256':signature,'payload':payload}
        outcome={'status':'completed','actual_prompt_visits':256,'global_step':64,'run_signature_sha256':signature}
        child={'cli':dict(cli,reference_cache='ref.jsonl'),'artifacts_sha256':{'ref.jsonl':'referencehash'}}
        p.validate_historical_training(signed,outcome,child,'exact_kl',{'same':'start'})
        bad=deepcopy(child); bad['cli']['seed']=45
        with self.assertRaises(RuntimeError): p.validate_historical_training(signed,outcome,bad,'exact_kl',{'same':'start'})
        bad=deepcopy(child); bad['artifacts_sha256']['ref.jsonl']='different_reference'
        with self.assertRaises(RuntimeError): p.validate_historical_training(signed,outcome,bad,'exact_kl',{'same':'start'})

    def test_timing_accounts_both_arms_all_evaluations_and_all_historical_timings(self):
        estimate=p.estimate_whole_stage(60,20,timing_commands())
        expected=1.5*(128*20+10*40+1168*.8+4*10)+600
        self.assertAlmostEqual(estimate['estimated_seconds'],expected,places=7)
        commands=timing_commands(); commands[0]['prediction_emissions']=commands[0]['prediction_emissions'][:-1]
        with self.assertRaises(RuntimeError): p.estimate_whole_stage(60,20,commands)
        with self.assertRaises(RuntimeError): p.estimate_whole_stage(60,20,timing_commands()[:3])
        duplicates=timing_commands(); duplicates[-1]=deepcopy(duplicates[0])
        with self.assertRaises(RuntimeError): p.estimate_whole_stage(60,20,duplicates)

    def test_historical_max_interval_includes_preprocessing_and_no_score_filter(self):
        commands=timing_commands()
        # A slow record in one historical arm raises the gate; no score exists
        # in the cost model, and it cannot remove that arm or replace the cohort.
        events=commands[-1]['prediction_emissions']
        for index in range(10,len(events)): events[index]['seconds_since_start']+=3
        estimate=p.estimate_whole_stage(60,20,commands)
        self.assertAlmostEqual(estimate['max_historical_preprocess_generation_write_interval_seconds'],3.8,places=10)

    def test_cpu_receipt_requires_zero_skips_and_no_cuda(self):
        report={'tests_run':15,'successful':True,'failures':0,'errors':0,'skipped':0,'cuda_initialized':False}
        self.assertTrue(p.validate_report(report,15))
        for field,value in [('skipped',3),('cuda_initialized',True),('tests_run',14),('errors',1)]:
            bad=dict(report); bad[field]=value
            self.assertFalse(p.validate_report(bad,15))

    def test_greedy_and_orbit_summaries_refuse_denominator_loss(self):
        rows=[{'recipe_id':str(i),'exact':i%2,'pair':.5,'valid':1} for i in range(200)]
        self.assertEqual(p.greedy_summary(rows),{'exact':.5,'pair':.5,'valid':1})
        with self.assertRaises(RuntimeError): p.greedy_summary(rows[:-1])
        example={'per_recipe':[{'recipe_id':str(i),'K_over24':.5,'all24_exact':0,'semantic_modal_consistency':.7,'pair':.8,'valid':1} for i in range(16)]}
        self.assertEqual(p.paired_orbit(example,example)['recipe_macro']['K_over24'],0)
        missing=deepcopy(example); missing['per_recipe'].pop()
        with self.assertRaises(RuntimeError): p.paired_orbit(example,missing)


if __name__=='__main__': unittest.main()
