"""Focused resource/completeness controls, no SSH/model/GPU calls."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from coverage_contract import ARMS,FLOWS,METHODS,TRAIN_SOURCE_PATHS,WALL_SECONDS,DEV_SHA,estimate_whole_stage
from coverage_pipeline import validate_stage,disk_estimate,historical_prediction_timing


class CoveragePipelineTests(unittest.TestCase):
    @staticmethod
    def policy():
        start=datetime.now(timezone.utc)-timedelta(seconds=1)
        return {'status':'locked_before_coverage_stage','wall_limit_seconds':WALL_SECONDS,'budget_cny_cap':25.6,
            'flow_order':list(FLOWS),'methods':list(METHODS),'primary_step':256,'diagnostic_step':64,
            'dev_manifest':'data/manifests/val.jsonl','dev_manifest_sha256':DEV_SHA,'orbit_first_recipes':16,'orbit_layouts':24,
            'test_access':False,'confirmation_access':False,'no_resume':True,
            'selection':'fixed256_primary_fixed64_diagnostic_no_dev_selection',
            'children':{key:{} for key in ('reference',*ARMS,*('smoke_'+arm for arm in ARMS))},
            'sources_sha256':{p:'0'*64 for p in TRAIN_SOURCE_PATHS},'expected_cpu_tests':1,
            'first_preflight_started_utc':start.isoformat(),'absolute_deadline_utc':(start+timedelta(seconds=WALL_SECONDS)).isoformat()}

    def test_draft_or_early_endpoint_not_formal_policy(self):
        validate_stage(self.policy())
        for key,value in [('status','draft'),('primary_step',64),('wall_limit_seconds',43200),('expected_cpu_tests',0)]:
            p=self.policy();p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_stage(p)

    def test_allfour_smokes_and_formal_policies_required(self):
        p=self.policy();p['children'].pop(ARMS[0])
        with self.assertRaises(ValueError):validate_stage(p)
        p=self.policy();p['sources_sha256'].pop('scripts/validate_run.py')
        with self.assertRaises(ValueError):validate_stage(p)

    def test_storage_retains_twenty_checkpoints_and_never_deletes_old(self):
        gate=disk_estimate(200,70,100000000000)
        self.assertEqual(gate['formal_checkpoints'],16);self.assertEqual(gate['smoke_checkpoints'],4)
        self.assertTrue(gate['passed']);self.assertFalse(disk_estimate(200,70,1000)['passed'])

    def test_whole_gate_uses_all_four_256steps_and3136predictions(self):
        smokes={arm:{'verified':True,'steps':1,'visits':4,'step_seconds':i+1,'wall_seconds':i+11} for i,arm in enumerate(ARMS)}
        gate=estimate_whole_stage(smokes,2,3)
        self.assertEqual(gate['components_seconds']['four_256step_trainings'],256*10)
        self.assertEqual(gate['components_seconds']['3136_fixed_predictions_and_12loads'],3136*2+12*3)
        bad=deepcopy(smokes);bad.pop(ARMS[0])
        with self.assertRaises(ValueError):estimate_whole_stage(bad,2,3)
        bad=deepcopy(smokes);bad[ARMS[0]]['verified']=False
        with self.assertRaises(ValueError):estimate_whole_stage(bad,2,3)

    def test_prediction_timing_requires_all384_for_allfour_controls(self):
        commands=[{'stage':'formal_'+arm,'returncode':0,'prediction_emissions':[{'seconds_since_start':i+2} for i in range(384)]} for arm in ARMS]
        state={'status':'complete_four_endpoint_orbits_development_only','commands':commands}
        self.assertEqual(historical_prediction_timing(state),(1,2))
        bad=deepcopy(state);bad['commands'][0]['prediction_emissions'].pop()
        with self.assertRaises(ValueError):historical_prediction_timing(bad)
        bad=deepcopy(state);bad['commands'][0]['stage']=bad['commands'][1]['stage']
        with self.assertRaises(ValueError):historical_prediction_timing(bad)


if __name__=='__main__':unittest.main(verbosity=2)
