"""CPU-only adversarial and numerical checks for the independent refresh verifier."""
from copy import deepcopy
from itertools import permutations
import math
import random
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import verify_refresh_run as v


def candidates():
    return [{'text':text,'token_ids':[32+i//6,300+i,400+i,500+i,151645],
             'token_count':5,'eos_token_id':151645,'post_eos_whitespace_token_ids':[198]}
            for i,text in enumerate(v.ACTIONS)]


def policy_and_tokens(weights=None):
    weights=weights or [i+1 for i in range(24)]
    z=math.fsum(weights)
    tokens=[[math.log(.9*w/z)/5]*5 for w in weights]
    return v.reconstruct(tokens,candidates()),tokens


def record(i):
    d=[2,0,3,1]
    return {'qid':'q'+str(i),'recipe_id':'r'+str(i),'presentation_id':'p'+str(i),
            'source_split':'train','manifest_set':'train_all7558','technical_status':'ok',
            'display_perm':d,'target_order':[0,1,2,3],
            'target_labels':['B','D','A','C'],'visit_id':'old-'+str(i)}


def visit_fixture():
    record0=record(0);gold,rewards=v.coordinate_gold(record0)
    ref_policy,ref_tokens=policy_and_tokens(list(range(24,0,-1)))
    policy,tokens=policy_and_tokens()
    audit={'input_hash':'a'*64}
    reference={key:record0[key] for key in ('qid','recipe_id','presentation_id','source_split',
                                          'display_perm','target_order','target_labels')}
    reference.update(candidates=candidates(),token_logps=ref_tokens,input_audit=audit,
                     exact_rewards=rewards,gold_candidate_index=gold,
                     **{key:ref_policy[key] for key in ('q','logq','logZ')})
    terms=v.independent_terms('exact_kl',policy,gold,ref_policy['logq'],v.BETA)
    visit={key:record0[key] for key in ('qid','recipe_id','presentation_id','source_split',
                                      'display_perm','target_order','target_labels')}
    selected=[i for i,c in enumerate(terms['coefficients']) if c!=0]
    visit.update(visit_index=4,pass_index=1,index_within_pass=0,reference_row_index=0,
                 global_step_before_update=1,reference_generation=1,
                 q0_reference_policy_sha256='q0',active_reference_policy_sha256='fresh',
                 run_signature_sha256='signed',input_audit=audit,input_hash='a'*64,
                 candidate_texts=list(v.ACTIONS),candidate_token_ids=[c['token_ids'] for c in candidates()],
                 token_logps_including_eos=tokens,sequence_logps=policy['scores'],reference_logq=ref_policy['logq'],
                 canonical_mass=math.exp(policy['logZ']),completion_token_count=5,exact_rewards=rewards,
                 numerical_reference_checks=[{'candidate_index':0,'literal_all_position_logits':False,
                                              'reference_token_logps':tokens[0],'max_token_logp_abs_difference':0.}],
                 sampled_indices=None,sampling_uniforms=None,
                 sampling_rng_state_sha256=v.object_sha(random.Random(44).getstate()),
                 nonzero_coefficients=len(selected),group_active=True,
                 backward_checks=[{'candidate_index':i,'max_token_logp_difference':0.} for i in selected],
                 consumption_point='all_selected_candidate_backward_calls_returned',
                 **{key:policy[key] for key in ('q','logq','logZ')},**terms)
    return record0,reference,visit


class RawPolicyTests(unittest.TestCase):
    def test_raw_sequence_probabilities_and_gold_coordinates(self):
        p,t=policy_and_tokens()
        self.assertAlmostEqual(math.exp(p['logZ']),.9,places=13)
        self.assertAlmostEqual(math.fsum(p['q']),1,places=13)
        self.assertAlmostEqual(p['q'][23],24/300,places=13)
        gold,reward=v.coordinate_gold(record(0))
        self.assertEqual(v.ACTIONS[gold],'B D A C')
        self.assertEqual(reward.count(1),1)

    def test_missing_eos_or_reordered_candidates_rejected(self):
        p,t=policy_and_tokens(); c=candidates()
        bad=deepcopy(c);bad[2]['token_ids'][-1]=198
        with self.assertRaises(ValueError):v.reconstruct(t,bad)
        bad=deepcopy(c);bad[0],bad[1]=bad[1],bad[0]
        with self.assertRaises(ValueError):v.reconstruct(t,bad)

    def test_bad_probability_mass_and_tokens_rejected(self):
        _,t=policy_and_tokens()
        t[0][0]=float('nan')
        with self.assertRaises(ValueError):v.reconstruct(t,candidates())
        with self.assertRaises(ValueError):v.reconstruct([[-.001]*5]*24,candidates())

    def test_complete_sequence_nll_differs_from_conditional(self):
        p,_=policy_and_tokens(); gold,_=v.coordinate_gold(record(0))
        raw=-p['scores'][gold]; conditional=-p['logq'][gold]
        self.assertAlmostEqual(raw-conditional,-math.log(.9),places=13)

    def test_wrong_gold_mapping_rejected(self):
        r=record(0);r['target_labels']=['A','B','C','D']
        with self.assertRaises(ValueError):v.coordinate_gold(r)


class ObjectiveTests(unittest.TestCase):
    def test_exact_kl_coefficients_match_finite_difference(self):
        p,_=policy_and_tokens(); ref,_=policy_and_tokens(list(range(24,0,-1)))
        gold=10; beta=v.BETA
        terms=v.independent_terms('exact_kl',p,gold,ref['logq'],beta)
        logits=p['scores']
        def objective(values):
            top=max(values);z=top+math.log(math.fsum(math.exp(a-top) for a in values))
            logq=[a-z for a in values];q=[math.exp(a) for a in logq]
            return q[gold]-beta*math.fsum(a*(b-c) for a,b,c in zip(q,logq,ref['logq']))
        h=1e-5
        for i,expected in enumerate(terms['coefficients']):
            plus=logits.copy();minus=logits.copy();plus[i]+=h;minus[i]-=h
            observed=(objective(plus)-objective(minus))/(2*h)
            self.assertAlmostEqual(observed,expected,places=9)

    def test_raw_sft_is_raw_gold_sequence_coefficient(self):
        p,_=policy_and_tokens(); terms=v.independent_terms('raw_sft',p,10,p['logq'],0.)
        self.assertEqual(terms['coefficients'],[float(i==10) for i in range(24)])
        self.assertEqual(terms['active_beta'],0.)
        with self.assertRaises(ValueError):v.independent_terms('raw_sft',p,10,p['logq'],v.BETA)

    def test_two_round_ideal_equals_fixed_half_beta(self):
        rng=random.Random(818)
        for gold in (0,10,23):
            ref,_=policy_and_tokens([.01+rng.random() for _ in range(24)])
            once=v.analytic_target(ref,gold,v.BETA)
            twice=v.analytic_target(once,gold,v.BETA)
            half=v.analytic_target(ref,gold,v.BETA/2)
            for a,b in zip(twice['q'],half['q']):self.assertAlmostEqual(a,b,places=14)

    def test_two_endpoint_objectives_satisfy_gap_identity(self):
        p,_=policy_and_tokens();ref,_=policy_and_tokens(list(range(24,0,-1)))
        for beta in (v.BETA,v.BETA/2):
            result=v.endpoint_objective(p,ref,10,beta)
            self.assertGreaterEqual(result['gap'],0)
            self.assertAlmostEqual(result['gap'],beta*result['KL_q_qstar'],places=13)
            self.assertAlmostEqual(result['gap'],result['available_gain']-result['achieved_gain'],places=13)

    def test_original_beta_stratum_not_reclassified_by_half_beta(self):
        weights=[.01]*24;weights[0]=.2;weights[1]=.5
        ref,_=policy_and_tokens(weights)
        self.assertEqual(v.fixed_stratum(ref,0),'initial_wrong_qstar_MAP_wrong')
        half=v.analytic_target(ref,0,v.BETA/2)
        self.assertEqual(max(range(24),key=lambda i:half['q'][i]),0)
        self.assertEqual(v.fixed_stratum(ref,0),'initial_wrong_qstar_MAP_wrong')

    def test_realization_is_ratio_of_sums_and_negative_unclipped(self):
        result=v.gain_realization([{'available_gain':1.,'achieved_gain':.5},
                                   {'available_gain':3.,'achieved_gain':-3.}])
        self.assertEqual(result['ratio_of_sums'],-.625)
        self.assertIsNone(v.gain_realization([{'available_gain':0.,'achieved_gain':-1.}])['ratio_of_sums'])


class ExposureContinuityTests(unittest.TestCase):
    def test_technical_effective_updates_required_but_formal_negative_result_retained(self):
        self.assertFalse(v.verify_effective_update('formal','same','same',False,'no_effective_adapter_update',[0.,0.]))
        self.assertTrue(v.verify_effective_update('path_smoke','before','after',True,'effective_training',[.1,.2]))
        with self.assertRaises(ValueError):v.verify_effective_update('smoke','same','same',False,'no_effective_adapter_update',[0.])
        with self.assertRaises(ValueError):v.verify_effective_update('path_smoke','before','after',True,'effective_training',[.1,0.])
        with self.assertRaises(ValueError):v.verify_effective_update('formal','same','same',True,'effective_training',[.1])

    def test_actual_visit_consumes_second_pass_fresh_reference_and_rejects_swaps(self):
        r,ref,visit=visit_fixture()
        def verify(row):
            return v.verify_visit(row,r,ref,4,4,'exact_kl',v.BETA,1,'fresh','q0','signed',44)
        result=verify(visit)
        self.assertEqual(result['current_forwards'],25)
        self.assertEqual(result['backwards'],24)
        for key,bad_value in [('reference_generation',0),('active_reference_policy_sha256','q0'),
                              ('pass_index',0),('global_step_before_update',0),('reference_row_index',1),
                              ('completion_token_count',4),('consumption_point','not_completed')]:
            bad=deepcopy(visit);bad[key]=bad_value
            with self.assertRaises(ValueError):verify(bad)
        bad=deepcopy(visit);bad['coefficients'][2]+=.01
        with self.assertRaises(ValueError):verify(bad)
        bad=deepcopy(visit);bad['backward_checks'].pop()
        with self.assertRaises(ValueError):verify(bad)

    def test_two_passes_repeat_same_unique_records_without_new_visit_ids(self):
        records=[record(i) for i in range(4)]
        visits=v.verify_two_pass_records(records,4,2)
        self.assertEqual(visits[:4],visits[4:])
        self.assertEqual(visits[4]['visit_id'],'old-0')
        with self.assertRaises(ValueError):v.verify_two_pass_records(records*2,8,1)

    def test_heldout_record_rejected(self):
        records=[record(i) for i in range(4)];records[1]['source_split']='val'
        with self.assertRaises(ValueError):v.verify_two_pass_records(records,4,2)

    def test_literal_check_origin_survives_q0_reordering(self):
        _,t=policy_and_tokens();checks=[{'candidate_index':0,'literal_all_position_logits':False,
                                        'reference_token_logps':t[0], 'max_token_logp_abs_difference':0.}]
        v.numeric_check(checks,t,False)
        with self.assertRaises(ValueError):v.numeric_check(checks,t,True)
        checks[0]['reference_token_logps']=checks[0]['reference_token_logps'].copy()
        checks[0]['reference_token_logps'][1]=float('nan')
        with self.assertRaises(ValueError):v.numeric_check(checks,t,False)

    def test_logical_state_is_order_independent_but_type_and_value_sensitive(self):
        a={'optimizer':{0:{'step':64,'moment':[.1,.2]}},'scheduler':None,'rng':(7,b'123')}
        b={'rng':(7,b'123'),'scheduler':None,'optimizer':{0:{'moment':[.1,.2],'step':64}}}
        self.assertEqual(v.logical_fingerprint(a),v.logical_fingerprint(b))
        b['optimizer']['0']=b['optimizer'].pop(0)
        self.assertNotEqual(v.logical_fingerprint(a),v.logical_fingerprint(b))
        b=deepcopy(a);b['optimizer'][0]['step']=0
        self.assertNotEqual(v.logical_fingerprint(a),v.logical_fingerprint(b))

    def test_continuity_rejects_every_changed_or_missing_component(self):
        components=('optimizer','scheduler','gradients','module_modes','rng','global_step')
        before={k:'same-'+k for k in components};after=before.copy()
        v.verify_unchanged_components(before,after,components)
        for key in components:
            bad=after.copy();bad[key]='changed'
            with self.assertRaises(ValueError):v.verify_unchanged_components(before,bad,components)
        bad=after.copy();del bad['gradients']
        with self.assertRaises(ValueError):v.verify_unchanged_components(before,bad,components)

    def test_actual_snapshot_summary_rejects_reset_optimizer_and_missing_rng(self):
        payload={'schema_version':1,'global_step':64,
                 'optimizer':{'state':{i:{'step':64,'exp_avg':[.1],'exp_avg_sq':[.01]} for i in range(392)}},
                 'scheduler':{'last_epoch':64},'gradients':{str(i):[.2] for i in range(392)},
                 'model_modes':{'':False,'layer':False},
                 'rng':{'python_global':(1,2),'sampling_rng':(3,4),'numpy':(5,6),'cpu':[7],'cuda':[[8]]}}
        summary=v.continuity_summary(payload)
        self.assertEqual(summary['optimizer_steps'],[64]*392)
        bad=deepcopy(payload);bad['optimizer']['state'][13]['step']=0
        with self.assertRaises(ValueError):v.continuity_summary(bad)
        bad=deepcopy(payload);del bad['rng']['cuda']
        with self.assertRaises(ValueError):v.continuity_summary(bad)
        bad=deepcopy(payload);bad['optimizer']['state'][13]['exp_avg'][0]=.15
        changed=v.continuity_summary(bad)
        with self.assertRaises(ValueError):v.verify_unchanged_components(
            summary['component_sha256'],changed['component_sha256'],summary['component_sha256'])
        bad=deepcopy(payload);bad['rng']['cpu'][0]=11
        changed=v.continuity_summary(bad)
        self.assertNotEqual(summary['component_sha256']['rng'],changed['component_sha256']['rng'])


class PrimaryStatisticsTests(unittest.TestCase):
    def test_both_primary_estimands_equal_weight_flows_not_independent400(self):
        columns={f+'_'+a:[0]*200 for f in ('flow44','flow45') for a in v.ARMS}
        columns['flow44_refresh_beta']=[1]*200
        columns['flow45_fixed_half_beta']=[1]*200
        effects=v.primary_differences(columns)
        self.assertEqual(effects['fixed_half_beta'],[0.]*200)
        self.assertEqual(effects['raw_sft'],[.5]*200)

    def test_joint_bootstrap_common_recipe_draws_and_exact_frozen_seed(self):
        columns={f+'_'+a:[0]*200 for f in ('flow44','flow45') for a in v.ARMS}
        for f in ('flow44','flow45'):columns[f+'_refresh_beta']=[i%2 for i in range(200)]
        result=v.joint_bootstrap(columns)
        self.assertEqual(result['independent_recipe_units'],200)
        self.assertEqual(len(result['samples']['raw_sft']),5000)
        self.assertEqual(result['samples']['raw_sft'],result['samples']['fixed_half_beta'])
        rng=random.Random(20261107);idx=[rng.randrange(200) for _ in range(200)]
        self.assertEqual(result['samples']['raw_sft'][0],sum(i%2 for i in idx)/200)
        with self.assertRaises(ValueError):v.joint_bootstrap(columns,draws=10)

    def test_nonconstant_columns_mapping_flow_weights_units_and_linear_percentiles(self):
        columns={f+'_'+a:[0]*200 for f in ('flow44','flow45') for a in v.ARMS}
        columns['flow44_refresh_beta']=[int(i%5<3) for i in range(200)]
        columns['flow45_refresh_beta']=[int(i%7<2) for i in range(200)]
        columns['flow44_fixed_half_beta']=[int(i%3==0) for i in range(200)]
        columns['flow45_fixed_half_beta']=[int(i%4<2) for i in range(200)]
        columns['flow44_raw_sft']=[int(i%11<5) for i in range(200)]
        columns['flow45_raw_sft']=[int(i%13<8) for i in range(200)]
        result=v.joint_bootstrap(columns)
        rng=random.Random(20261107);first=[rng.randrange(200) for _ in range(200)]
        for control in ('fixed_half_beta','raw_sft'):
            direct=sum((columns['flow44_refresh_beta'][i]-columns['flow44_'+control][i]
                        +columns['flow45_refresh_beta'][i]-columns['flow45_'+control][i])/2 for i in range(200))/200
            self.assertAlmostEqual(result['point_estimates_pp'][control],100*direct,places=13)
            resampled=sum((columns['flow44_refresh_beta'][i]-columns['flow44_'+control][i]
                           +columns['flow45_refresh_beta'][i]-columns['flow45_'+control][i])/2 for i in first)/200
            self.assertAlmostEqual(result['samples'][control][0],resampled,places=13)
            flows=[result['per_flow_samples'][f+'_refresh_minus_'+control][0] for f in ('flow44','flow45')]
            self.assertAlmostEqual(result['samples'][control][0],sum(flows)/2,places=13)
            sorted_draws=sorted(result['samples'][control]);position=4999*.025;lo=int(position)
            expected=100*(sorted_draws[lo]+(position-lo)*(sorted_draws[lo+1]-sorted_draws[lo]))
            self.assertAlmostEqual(result['intervals95_pp'][control][0],expected,places=13)
        self.assertEqual(v.linear_percentile([0.,10.,20.,30.],.25),7.5)


class PhysicalReceiptTests(unittest.TestCase):
    def test_nested_root_inside_remote_prefers_actual_root_for_absolute_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            remote=Path(temporary).resolve()/'remote_project'
            nested=remote/'tmp'/'fixture_root';nested.mkdir(parents=True)
            file=nested/'checkpoint'/'weight.bin';file.parent.mkdir()
            raw=b'nested-root regression';file.write_bytes(raw)
            with patch.object(v,'REMOTE',remote):
                files=v.Files(nested)
                self.assertEqual(files.relative(file),'checkpoint/weight.bin')
                self.assertEqual(files.path(file),file)
                self.assertEqual(files.physical(file),{'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
                self.assertEqual(files.read(file),raw)
                self.assertEqual(set(files.inputs),{'checkpoint/weight.bin'})
                # A remote-absolute path outside this actual root still maps
                # into an exported root; production root==REMOTE is unchanged.
                self.assertEqual(files.relative(remote/'data/images/example.jpg'),'data/images/example.jpg')
                production=v.Files(remote)
                self.assertEqual(production.relative(file),'tmp/fixture_root/checkpoint/weight.bin')

    def test_safe_tensor_raw_fingerprint_and_unchecked_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'checkpoint';path.mkdir()
            name='base_model.model.language_model.layers.0.lora_A.default.weight'
            saved_name=name.replace('.default.','.')
            data=struct.pack('<ff',1.,-.5)
            header=json.dumps({saved_name:{'dtype':'F32','shape':[1,2],'data_offsets':[0,8]}}).encode()
            file=path/'adapter_model.safetensors'
            file.write_bytes(struct.pack('<Q',len(header))+header+data)
            inventory=[{'name':name,'shape':[1,2],'numel':2}]
            files=v.Files(root)
            self.assertEqual(v.safetensor_fingerprint(files,'checkpoint',inventory),hashlib.sha256(name.encode()+data).hexdigest())
            file.write_bytes(file.read_bytes()+b'x')
            with self.assertRaises(ValueError):v.safetensor_fingerprint(v.Files(root),'checkpoint',inventory)

    def test_safe_tensor_nonfinite_values_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'adapter_model.safetensors'
            header=json.dumps({'x':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
            path.write_bytes(struct.pack('<Q',len(header))+header+struct.pack('<f',float('nan')))
            with self.assertRaises(ValueError):v.safetensor_fingerprint(v.Files(root),'.',[{'name':'x','shape':[1],'numel':1}])

    def test_path_and_sha_receipts_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            files=v.Files(root)
            for name in ('../escape','data/manifests/test.jsonl','confirmation200/raw.jsonl'):
                with self.assertRaises((ValueError,FileNotFoundError)):files.read(name)
            path=Path(root)/'evidence.json';path.write_text('{}')
            files.read('evidence.json');path.write_text('{"changed":1}')
            with self.assertRaises(ValueError):files.stable()


class WholeStageGateTests(unittest.TestCase):
    def fixture(self):
        arms,technical,_=v.fixed_stage_schedule()
        observed={key:{'verified':True,'steps':2 if key.startswith('path_smoke_') else 1,
                       'visits':8 if key.startswith('path_smoke_') else 4,
                       'step_seconds':[2.,3.] if key.startswith('path_smoke_') else [2.],
                       'final_bank_rows':4,'final_bank_seconds':8.,'wall_seconds':30.,'verify_seconds':5.,
                       'refresh_rows':4 if key.startswith('path_smoke_') else 0,
                       'refresh_seconds':12. if key.startswith('path_smoke_') else 0.}
                  for key in technical}
        history={'full_run_step_floor_seconds':{'raw_sft':10.,'exact_kl':20.},
                 'training_load_and_checkpoint_floor_seconds':40.,'verifier_wall_floor_seconds':6.,
                 'canonical_score_per_row_floor_seconds':1.}
        return observed,history

    def test_resource_bound_uses_all_arms_full_history_and_fresh_final_costs(self):
        observed,history=self.fixture();forecast=v.resource_forecast(observed,history,.5,4.)
        self.assertEqual(forecast['canonical_score_per_row_seconds'],3.)
        self.assertEqual(forecast['components_seconds']['eight_128update_trainings'],128*(2*10+6*20))
        self.assertEqual(forecast['components_seconds']['two_actual256_reference_refreshes'],512*3)
        self.assertEqual(forecast['components_seconds']['eight_fixed256_final_train_banks'],2048*3)
        self.assertEqual(forecast['components_seconds']['4672_predictions_and_16_model_loads'],4672*.5+16*4.)
        expected=1.5*sum(forecast['components_seconds'].values())+600
        self.assertEqual(forecast['estimated_seconds'],expected)

    def test_resource_gate_rejects_missing_path_smoke_and_partial_rows(self):
        observed,history=self.fixture();del observed['path_smoke_flow45_refresh_beta']
        with self.assertRaises(ValueError):v.resource_forecast(observed,history,.5,4.)
        observed,history=self.fixture();observed['smoke_flow44_fixed_beta']['final_bank_rows']=3
        with self.assertRaises(ValueError):v.resource_forecast(observed,history,.5,4.)

    def test_storage_covers_28_checkpoints_8_snapshots_then_16_and4_remaining(self):
        checkpoint=220000000;adapter=70000000;continuity=300000000
        first=v.disk_requirement(checkpoint,adapter,continuity,False)
        later=v.disk_requirement(checkpoint,adapter,continuity,True)
        self.assertEqual(first,math.ceil(1.2*(28*checkpoint+8*continuity+512*1024**2)+1024**3))
        self.assertEqual(later,math.ceil(1.2*(16*checkpoint+4*continuity+512*1024**2)+1024**3))
        self.assertGreater(first,later)

    def test_partial_stage_guard_refuses_before_any_raw_score_read(self):
        arms,technical,schedule=v.fixed_stage_schedule()
        self.assertEqual(len(schedule),74)
        self.assertEqual(schedule[-1],'stage_global_verify')
        class MemoryFiles:
            def __init__(self):self.opened=[];self.mutable_inputs=set()
            def relative(self,path):return str(path)
            def json(self,path):
                self.opened.append(str(path))
                if str(path)=='parent.json':
                    return {'output_directory':'stage','status':'locked_before_reference_refresh_stage',
                            'command_schedule':schedule,'formal_order':arms,'technical_order':technical,
                            'children':{name:{} for name in arms+technical}}
                if str(path)=='stage/state.json':return {'status':'still_running','active_child_pid':123,'commands':[]}
                raise AssertionError('Opened score/raw evidence before completion: '+str(path))
        files=MemoryFiles()
        with self.assertRaises(ValueError):v.audit_stage_completion(files,'parent.json')
        self.assertEqual(files.opened,['parent.json','stage/state.json'])

    def test_live_stage_journal_receipt_is_not_a_persisted_immutable_input(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'state.json';path.write_text('{"status":"running"}')
            files=v.Files(root);files.read('state.json');files.mutable_inputs.add('state.json')
            immutable,observed=files.report_receipts()
            self.assertNotIn('state.json',immutable)
            self.assertIn('state.json',observed)
            # The journal is still required to remain stable during this audit;
            # it may transition only after the verifier subprocess has returned.
            path.write_text('{"status":"completed"}')
            with self.assertRaises(ValueError):files.stable()

    def test_only_actual_self_live_74th_guard_or_terminal_all74_can_open_scores(self):
        _,_,schedule=v.fixed_stage_schedule()
        commands=[{'stage':name,'returncode':0,'pid':123,'pgid':v.os.getpgrp()} for name in schedule]
        commands[-1].update(returncode=None,pid=v.os.getpid())
        state={'status':'running_stage_global_verify','active_child_pid':v.os.getpid(),
               'pgid':v.os.getpgrp(),'commands':commands}
        self.assertTrue(v.validate_stage_command_window(state,schedule))
        for mutation in ('wrong_pid','wrong_pgid','failed_science','wrong_stage'):
            bad=deepcopy(state)
            if mutation=='wrong_pid':bad['commands'][-1]['pid']+=1
            elif mutation=='wrong_pgid':bad['commands'][-1]['pgid']+=1
            elif mutation=='failed_science':bad['commands'][20]['returncode']=1
            else:bad['commands'][-1]['stage']='unplanned_guard'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):v.validate_stage_command_window(bad,schedule)
        state.update(status='completed_eight_reference_refresh_endpoints_independently_verified',active_child_pid=None)
        state['commands'][-1]['returncode']=0
        self.assertFalse(v.validate_stage_command_window(state,schedule))
        state['commands'][-1]['returncode']=1
        with self.assertRaises(ValueError):v.validate_stage_command_window(state,schedule)


if __name__=='__main__':
    unittest.main()
