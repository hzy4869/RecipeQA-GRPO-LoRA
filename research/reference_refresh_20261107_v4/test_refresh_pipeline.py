"""CPU contract tests for completeness, budget accounting and fixed sequencing."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from refresh_contract import ARMS,ARM_KINDS,FLOWS,PATH_SMOKES,TRAIN_SOURCE_PATHS,WALL_SECONDS,DEV_SHA
from refresh_pipeline import (TECHNICAL_KEYS,validate_stage,disk_estimate,historical_prediction_timing,
    historical_training_timing,estimate_whole_stage,require_science_commands_complete,FINAL_GUARD_BOOTSTRAP)
from build_draft_policy import command_schedule,HISTORICAL_ARMS


class RefreshPipelineTests(unittest.TestCase):
    @staticmethod
    def policy():
        start=datetime.fromisoformat('2026-11-07T04:18:24.421606+00:00')
        return {'status':'locked_before_reference_refresh_stage','wall_limit_seconds':WALL_SECONDS,'budget_cny_cap':38.4,
            'flow_order':list(FLOWS),'methods':list(ARM_KINDS),'formal_order':list(ARMS),'technical_order':list(TECHNICAL_KEYS),
            'primary_step':128,'boundary_step':64,'command_schedule':command_schedule(),
            'dev_manifest':'data/manifests/val.jsonl','dev_manifest_sha256':DEV_SHA,'orbit_first_recipes':16,'orbit_layouts':24,
            'test_access':False,'confirmation_access':False,'no_resume':True,
            'selection':'fixed128_primary_step64_boundary_no_dev_selection',
            'children':{k:{} for k in (*ARMS,*TECHNICAL_KEYS)},'sources_sha256':{p:'0'*64 for p in TRAIN_SOURCE_PATHS},
            'historical_training_runs':{k:'prior/'+k for k in HISTORICAL_ARMS},'expected_cpu_tests':1,'continuation':True,'newly_executed_commands':54,'historical_reused_commands':20,'technical_gpu_reexecution':False,
            'first_preflight_started_utc':start.isoformat(),'absolute_deadline_utc':(start+timedelta(seconds=WALL_SECONDS)).isoformat()}

    @staticmethod
    def smokes():
        return {k:{'verified':True,'steps':2 if k in PATH_SMOKES else 1,'visits':8 if k in PATH_SMOKES else 4,
            'step_seconds':[2.,2.] if k in PATH_SMOKES else [2.],'wall_seconds':20.,'verify_seconds':3.,
            'effective_adapter_update':True,'actual_parameter_update_l2':[.1,.1] if k in PATH_SMOKES else [.1],
            'final_bank_rows':4,'final_bank_seconds':4.,'refresh_rows':4 if k in PATH_SMOKES else 0,
            'refresh_seconds':4. if k in PATH_SMOKES else 0.} for k in TECHNICAL_KEYS}

    @staticmethod
    def history():
        return {'full_run_step_floor_seconds':{'raw_sft':10.,'exact_kl':40.},
            'canonical_score_per_row_floor_seconds':2.5,'training_load_and_checkpoint_floor_seconds':10.,
            'verifier_wall_floor_seconds':4.,'historical_q0_forward_cost':25600,'new_q0_forwards':0}

    def test_locked_complete_policy_is_accepted(self):
        validate_stage(self.policy())

    def test_rejects_draft_shortened_endpoint_or_budget(self):
        for key,value in [('status','draft_not_authorized_for_execution'),('primary_step',64),('boundary_step',32),
                          ('wall_limit_seconds',57600),('budget_cny_cap',25.6),('expected_cpu_tests',0)]:
            p=self.policy();p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_stage(p)

    def test_rejects_missing_path_smoke_or_scientific_arm(self):
        for key in (ARMS[0],TECHNICAL_KEYS[0],PATH_SMOKES[1]):
            p=self.policy();p['children'].pop(key)
            with self.subTest(key=key),self.assertRaises(ValueError):validate_stage(p)

    def test_rejects_closure_loss_or_changed_order(self):
        p=self.policy();p['sources_sha256'].pop('scripts/validate_run.py')
        with self.assertRaises(ValueError):validate_stage(p)
        p=self.policy();p['formal_order'].reverse()
        with self.assertRaises(ValueError):validate_stage(p)

    def test_no_test_confirmation_or_checkpoint_selection(self):
        for key,value in [('test_access',True),('confirmation_access',True),('no_resume',False),('selection','best_dev')]:
            p=self.policy();p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_stage(p)

    def test_all74_commands_are_distinct_fixed_and_complete(self):
        rows=command_schedule();self.assertEqual(len(rows),74);self.assertEqual(len(set(rows)),74)
        self.assertEqual(rows[0],'cpu_preflight');self.assertEqual(sum(r.startswith('q0_') for r in rows),4)
        self.assertEqual(rows[-1],'stage_global_verify')
        first_formal=rows.index(ARMS[0]+'_train')
        self.assertGreater(first_formal,max(rows.index(k+'_verify') for k in TECHNICAL_KEYS))
        self.assertFalse(any('step64_dev' in r for r in rows))

    def test_science_commit_cannot_skip_final_audit_or_claim_incomplete_work(self):
        schedule=command_schedule();state={'active_child_pid':None,
            'commands':[{'stage':s,'returncode':0} for s in schedule[:-1]]}
        require_science_commands_complete(state,schedule)
        for mode in ('missing','failed','active','already_audited','no_audit_schedule'):
            current=deepcopy(state);expected=list(schedule)
            if mode=='missing':current['commands'].pop()
            elif mode=='failed':current['commands'][-1]['returncode']=1
            elif mode=='active':current['active_child_pid']=123
            elif mode=='already_audited':current['commands'].append({'stage':schedule[-1],'returncode':0})
            else:expected.pop()
            with self.subTest(mode=mode),self.assertRaises(ValueError):require_science_commands_complete(current,expected)

    def test_final_guard_barrier_preserves_pid_group_and_requires_parent_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            script=Path(tmp)/'guard.py';receipt=Path(tmp)/'receipt.json'
            script.write_text("import os,sys,json; from pathlib import Path; "
                "Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp()}))")
            command=[sys.executable,'-u','-c',FINAL_GUARD_BOOTSTRAP,str(script),str(receipt)]
            for release in (False,True):
                child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                try:
                    with self.assertRaises(subprocess.TimeoutExpired):child.wait(timeout=.1)
                    self.assertFalse(receipt.exists())
                    if release:child.stdin.write(b'1');child.stdin.flush()
                    child.stdin.close();child.wait(timeout=5)
                    if release:
                        self.assertEqual(child.returncode,0)
                        self.assertEqual(json.loads(receipt.read_text()),{'pid':child.pid,'pgid':os.getpgrp()})
                    else:
                        self.assertNotEqual(child.returncode,0);self.assertFalse(receipt.exists())
                finally:
                    if child.poll() is None:child.kill();child.wait(timeout=5)
                    child.stdout.close();child.stderr.close()

    def test_cannot_remove_or_insert_schedule_command(self):
        for rows in (command_schedule()[:-1],['cpu_preflight']+command_schedule()):
            p=self.policy();p['command_schedule']=rows
            with self.assertRaises(ValueError):validate_stage(p)

    def test_storage_accounts28_checkpoints_and8_continuity_files(self):
        r=disk_estimate(200,70,10**12)
        self.assertEqual(r['remaining_checkpoint_count'],28);self.assertEqual(r['technical_checkpoints'],12)
        self.assertEqual(r['remaining_continuity_count'],8);self.assertEqual(r['final_adapter_exports'],0)
        self.assertEqual(r['continuity_file_bound_bytes'],270);self.assertTrue(r['passed'])
        self.assertFalse(disk_estimate(200,70,1000)['passed'])

    def test_post_smoke_storage_does_not_double_charge_written_technical_files(self):
        r=disk_estimate(200,70,10**12,technical_complete=True,continuity_bytes=500)
        self.assertEqual(r['remaining_checkpoint_count'],16);self.assertEqual(r['remaining_continuity_count'],4)
        self.assertEqual(r['continuity_file_bound_bytes'],500)
        with self.assertRaises(ValueError):disk_estimate(50,70,1000)

    def test_gate_uses_full_historical_floor_and_all_targets(self):
        g=estimate_whole_stage(self.smokes(),self.history(),.5,3.)
        self.assertEqual(g['components_seconds']['eight_128update_trainings'],128*(2*10+6*40))
        self.assertEqual(g['components_seconds']['two_actual256_reference_refreshes'],512*2.5)
        self.assertEqual(g['components_seconds']['eight_fixed256_final_train_banks'],2048*2.5)
        self.assertEqual(g['components_seconds']['4672_predictions_and_16_model_loads'],4672*.5+16*3)
        self.assertEqual(g['formal_updates'],1024);self.assertEqual(g['new_q0_forwards'],0)
        self.assertEqual(g['historical_shared_q0_forwards_charged_once'],25600)
        self.assertTrue(g['all_eight_formal_arms_required'])

    def test_gate_requires_each_verified_technical_path_and_actual_refresh(self):
        for change in ('missing','unverified','wrong_visits','missing_refresh'):
            s=self.smokes()
            if change=='missing':s.pop(PATH_SMOKES[0])
            elif change=='unverified':s[TECHNICAL_KEYS[0]]['verified']=False
            elif change=='wrong_visits':s[PATH_SMOKES[0]]['visits']=4
            else:s[PATH_SMOKES[0]]['refresh_rows']=0
            with self.subTest(change=change),self.assertRaises(ValueError):estimate_whole_stage(s,self.history(),.5,3)

    def test_slower_technical_step_and_refresh_increase_estimate(self):
        s=self.smokes();base=estimate_whole_stage(s,self.history(),.5,3)
        s['path_smoke_flow44_refresh_beta']['step_seconds'][1]=60
        s['path_smoke_flow44_refresh_beta']['refresh_seconds']=20
        g=estimate_whole_stage(s,self.history(),.5,3)
        self.assertEqual(g['formal_step_seconds_by_arm']['flow44_refresh_beta'],60)
        self.assertEqual(g['canonical_score_per_row_seconds'],5)
        self.assertGreater(g['estimated_seconds'],base['estimated_seconds'])

    def test_technical_gate_rejects_no_update_and_zero_path_step(self):
        for kind in ('no_adapter_change','zero_second_step','nan_update','missing_step'):
            s=self.smokes();target=s[PATH_SMOKES[0]]
            if kind=='no_adapter_change':target['effective_adapter_update']=False
            elif kind=='zero_second_step':target['actual_parameter_update_l2'][1]=0.
            elif kind=='nan_update':target['actual_parameter_update_l2'][1]=float('nan')
            else:target['actual_parameter_update_l2'].pop()
            with self.subTest(kind=kind),self.assertRaises(ValueError):estimate_whole_stage(s,self.history(),.5,3.)

    def test_prediction_floor_requires_complete_four384_traces(self):
        rows=[{'stage':'formal_'+a,'returncode':0,'prediction_emissions':[{'seconds_since_start':i+2} for i in range(384)]} for a in HISTORICAL_ARMS]
        state={'status':'complete_four_endpoint_orbits_development_only','commands':rows}
        self.assertEqual(historical_prediction_timing(state),(1,2))
        for change in ('missing','duplicate','failed','nonmonotonic'):
            s=deepcopy(state)
            if change=='missing':s['commands'][0]['prediction_emissions'].pop()
            elif change=='duplicate':s['commands'][0]['stage']=s['commands'][1]['stage']
            elif change=='failed':s['commands'][0]['returncode']=1
            else:s['commands'][0]['prediction_emissions'][2]['seconds_since_start']=1
            with self.subTest(change=change),self.assertRaises(ValueError):historical_prediction_timing(s)

    def test_training_floor_is_complete_outer_wall_not_hot_step(self):
        commands=[];outcomes={};steps={}
        for a in HISTORICAL_ARMS:
            method='raw_sft' if a.endswith('raw_sft') else 'exact_kl'
            commands += [{'stage':a+'_train','returncode':0,'wall_seconds':2560 if method=='raw_sft' else 10240},
                         {'stage':a+'_verify','returncode':0,'wall_seconds':40}]
            outcomes[a]={'status':'completed','global_step':256,'actual_prompt_visits':1024,'method':method}
            steps[a]=[{'global_step':i+1,'elapsed_seconds':1.} for i in range(256)]
        ref={'status':'completed_reference_only','rows':1024,'updates':0,'reference_candidate_forwards_charged_this_run':25600,'elapsed_seconds':2560.}
        state={'status':'completed_four_coverage_endpoints_development_only','commands':commands}
        h=historical_training_timing(state,outcomes,steps,ref)
        self.assertEqual(h['full_run_step_floor_seconds'],{'raw_sft':10.,'exact_kl':40.})
        self.assertEqual(h['canonical_score_per_row_floor_seconds'],2.5)
        bad=deepcopy(steps);bad[HISTORICAL_ARMS[0]].pop()
        with self.assertRaises(ValueError):historical_training_timing(state,outcomes,bad,ref)


if __name__=='__main__':unittest.main(verbosity=2)
