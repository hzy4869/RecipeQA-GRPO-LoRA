"""Focused stdlib CPU contracts: no model/torch import or silent skip."""
import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random
import tempfile
from unittest.mock import patch
import sys
import unittest

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import refresh_contract as c
import refresh_train as t


def record(i):
    return {'qid':f'q{i}','recipe_id':f'r{i%254}','presentation_id':f'p{i}', 'visit_id':f'v{i}',
            'source_split':'train','technical_status':'ok','manifest_set':'train_all7558',
            'target_labels':['A','B','C','D'],'target_order':[0,1,2,3],'display_perm':[0,1,2,3],
            'image_paths_display':[f'{i}_{j}.jpg' for j in range(4)]}


def child(mode='formal',arm='refresh_beta',flow='flow44'):
    budget=c.budget_for(mode); beta=0. if arm=='raw_sft' else c.BETA/2 if arm=='fixed_half_beta' else c.BETA
    cli={**budget,'seed':int(flow[-2:]),'manifest':str(c.BASE/f'research/reference_refresh_20261107_v3/prepared_v1/{"formal" if mode=="formal" else "smoke"}_{flow}.jsonl'),
         'model_path':str(c.BASE/c.MODEL_PATH),'adapter_path':str(c.BASE/c.INITIAL_ADAPTER),'revision':c.REVISION,
         'method':'raw_sft' if arm=='raw_sft' else 'exact_kl','beta':beta,'refresh_boundary_step':budget['max_steps']//2 if arm=='refresh_beta' and mode!='smoke' else None,
         'learning_rate':5e-6,'final_bank_count':budget['unique_count'],'reference_cache':'q0.jsonl','prepare_reference_only':False,
         'min_pixels':65536,'max_pixels':262144,'recipe_token_budget':1536,'max_input_tokens':4096,'logp_atol':.01,'wall_seconds':72000}
    return {'status':'locked_before_reference_refresh_training','mode':mode,'arm':arm,'flow':flow,'cli':cli,
            'sources_sha256':dict.fromkeys(c.TRAIN_SOURCE_PATHS,'0'*64),'reward':'unique_complete_order_exact_0_1',
            'selection':'fixed128_primary_step64_boundary_no_dev_selection',
            'optimizer':{'accumulation':4,'clip_norm':1.,'canonical_C_including_eos':5,'fresh_adamw':True},
            'first_preflight_started_utc':'2026-11-07T00:00:00+00:00','batch_clock_started_utc':'2026-11-07T00:00:00+00:00'}


class RefreshContracts(unittest.TestCase):
    def test_all_eighteen_predeclared_child_roles(self):
        count=0
        for flow in c.FLOWS:
            for arm in c.ARM_KINDS:
                for mode in ('smoke','formal'):
                    p=child(mode,arm,flow);self.assertEqual(c.validate_child(p),p['cli']);count+=1
            p=child('path_smoke','refresh_beta',flow);c.validate_child(p);count+=1
        self.assertEqual(count,18)

    def test_method_beta_boundary_budget_changes_refused(self):
        for key,value in [('beta',c.BETA/2),('refresh_boundary_step',63),('passes',1),('max_steps',64),('unique_count',512),('save_steps',16),('seed',45),('final_bank_count',200),('prepare_reference_only',True)]:
            p=child();p['cli'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_child(p)
        with self.assertRaises(ValueError):c.validate_child(child('path_smoke','fixed_beta'))

    def test_source_closure_reward_and_clock_not_optional(self):
        for key,value in [('reward','pair_bonus'),('selection','best_dev'),('sources_sha256',{}),('batch_clock_started_utc','later')]:
            p=child();p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_child(p)

    def test_fixed_pool256_254recipes_and_same_second_pass(self):
        full=[record(i) for i in range(7558)];pool=c.fixed_pool(full);orders=c.ordered_pools(pool)
        expected=deepcopy(pool);random.Random(45).shuffle(expected)
        self.assertEqual(orders['flow44'],full[:256]);self.assertEqual(orders['flow45'],expected)
        self.assertEqual(len({r['recipe_id'] for r in pool}),254)
        for flow,records in orders.items():
            t.validate_budget(records,512,128,256,2)
            visits=[records[i%256] for i in range(512)]
            self.assertEqual(visits[:256],visits[256:]);self.assertEqual(len({c.identity(r) for r in visits}),256)
        self.assertEqual(full,[record(i) for i in range(7558)])

    def test_unique_input_cannot_be_faked_as512_or_repeatedqid(self):
        data=[record(i) for i in range(256)]
        with self.assertRaises(ValueError):t.validate_budget(data*2,512,128,256,2)
        data[-1]=data[0]
        with self.assertRaises(ValueError):t.validate_budget(data,512,128,256,2)
        data=[record(i) for i in range(256)];data[0]['source_split']='val'
        with self.assertRaises(ValueError):t.validate_budget(data,512,128,256,2)

    def test_reference_reindex_preserves_all_numeric_payload_source_literal(self):
        records=[record(i) for i in range(1024)]
        spec={'pixel_sha256':{p:'0'*64 for r in records for p in r['image_paths_display']},'manifest_sha256':'old','selected_records_sha256':c.hash_json(records),'other_fixed':'untouched'}
        rows=[{'index':i,'qid':r['qid'],'presentation_id':r['presentation_id'],'reference_spec_sha256':c.hash_json(spec),'token_logps':[[float(-i-j-1)]*5 for j in range(24)],'numerical_checks':[{'literal':i==0}],'q':[1/24]*24,'input_audit':{'input_hash':str(i)}} for i,r in enumerate(records)]
        target=c.ordered_pools(records[:256])['flow45'];new,newspec,lineage=c.reference_reindex(rows,spec,records,target,'new')
        self.assertEqual(len(new),256);self.assertEqual(newspec['other_fixed'],'untouched')
        for actual,entry in zip(new,lineage['rows']):
            original=rows[entry['source_index']]
            self.assertEqual({k:v for k,v in actual.items() if k not in c.GENERATED_REFERENCE_ALLOWED_ROW_FIELDS},{k:v for k,v in original.items() if k not in c.GENERATED_REFERENCE_ALLOWED_ROW_FIELDS})
            self.assertEqual(entry['reference_literal_check_was_source_index_zero'],entry['source_index']==0)
        self.assertNotEqual(lineage['rows'][0]['source_index'],0)

    def test_cache_reindex_rejects_record_changes(self):
        rs=[record(i) for i in range(1024)];spec={'pixel_sha256':{p:'0'*64 for r in rs for p in r['image_paths_display']}}
        rows=[{'index':i,'qid':r['qid'],'presentation_id':r['presentation_id'],'reference_spec_sha256':c.hash_json(spec)} for i,r in enumerate(rs)]
        target=deepcopy(rs[:256]);target[0]['target_labels']=['D','C','B','A']
        with self.assertRaises(ValueError):c.reference_reindex(rows,spec,rs,target,'new')

    def test_actual6144_backward_seen_proof_not_preparation(self):
        pool=[record(i) for i in range(256)]
        visits=[{**record(i),'global_step_before_update':i//8,'consumption_point':'training_step_forward_backward_returned','run_signature_sha256':'sig'} for i in range(6144)]
        result=c.validate_prior_exposure(pool,visits,'sig');self.assertEqual(result['already_consumed_pool_questions'],256)
        visits[0]['consumption_point']='prepared_only'
        with self.assertRaises(ValueError):c.validate_prior_exposure(pool,visits,'sig')

    def test_common20h_clock_never_restarts(self):
        start=datetime(2026,11,7,tzinfo=timezone.utc);policy={'first_preflight_started_utc':start.isoformat(),'absolute_deadline_utc':(start+timedelta(seconds=72000)).isoformat()}
        self.assertEqual(c.remaining_seconds(policy,start+timedelta(seconds=100)),71900)
        with self.assertRaises(ValueError):c.remaining_seconds(policy,start+timedelta(seconds=72000))
        policy['absolute_deadline_utc']=(start+timedelta(seconds=72001)).isoformat()
        with self.assertRaises(ValueError):c.remaining_seconds(policy,start)

    def test_objective_math_ast_matches_frozen_coverage(self):
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef)}
        old=functions(ROOT/'research/coverage_20261106/coverage_train.py');new=functions(HERE/'refresh_train.py')
        for name in ['log_policy','categorical_draws','policy_coefficients','append','pixel_identities','cpu_gradient_norm','prepare']:
            with self.subTest(name=name):self.assertEqual(old[name],new[name])

    def test_complete_eos24_policy_math_normalization_and_rawce(self):
        values=[-12-i/10 for i in range(24)];q=t.log_policy(values);gold=7;reward=[int(i==gold) for i in range(24)]
        ref=t.log_policy([-14-i/15 for i in range(24)])['logq']
        for beta in (c.BETA,c.BETA/2):
            terms=t.policy_coefficients('exact_kl',q,reward,ref,beta)
            self.assertAlmostEqual(sum(terms['coefficients']),0);self.assertAlmostEqual(terms['regularized_expected_reward'],q['q'][gold]-beta*terms['kl_q_reference'])
        terms=t.policy_coefficients('raw_sft',q,reward,ref,0)
        self.assertEqual(terms['coefficients'],reward);self.assertEqual(terms['active_beta'],0.)

    def test_logical_state_schema_preserves_types_and_order_independence(self):
        x={1:[1,2], '1':(1,2)};y={'1':(1,2),1:[1,2]}
        self.assertEqual(t.logical_content(x),t.logical_content(y))
        self.assertNotEqual(t.logical_content((1,2)),t.logical_content([1,2]))
        with self.assertRaises(ValueError):t.logical_content(float('nan'))

    def test_optimizer_moment_gradient_rng_mutations_change_fingerprints(self):
        payload={'schema_version':1,'global_step':64,'optimizer':{'state':{i:{'step':64.,'exp_avg':[.1,.2],'exp_avg_sq':[.3,.4]} for i in range(392)}},'scheduler':{'last_epoch':64},'gradients':{'weight':[1.,2.]},'model_modes':{'':False},'rng':{'python':(1,2,3)}}
        base=t.continuity_summary(payload)
        for key in ['optimizer','gradients','rng','model_modes']:
            changed=deepcopy(payload)
            if key=='optimizer':changed[key]['state'][0]['exp_avg'][0]=.11
            elif key=='gradients':changed[key]['weight'][0]=1.01
            elif key=='rng':changed[key]['python']=(1,2,4)
            else:changed[key]['']=True
            with self.subTest(key=key):self.assertNotEqual(base['logical_sha256'],t.continuity_summary(changed)['logical_sha256'])
        payload['optimizer']['state'][0]['step']=1.
        with self.assertRaises(ValueError):t.continuity_summary(payload)

    def test_premodel_authorization_requires_actual_policy_and_all_distinct_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();p=child();start=datetime.now(timezone.utc)-timedelta(seconds=1)
            p.update(authorization_path='auth.json',first_preflight_started_utc=start.isoformat(),batch_clock_started_utc=start.isoformat(),absolute_deadline_utc=(start+timedelta(seconds=72000)).isoformat())
            cp=root/'child.json';cp.write_text(json.dumps(p));key='flow44_refresh_beta'
            keys=list(c.ARMS)+['smoke_'+x for x in c.ARMS]+list(c.PATH_SMOKES)
            parent={'expected_cpu_tests':1,'sources_sha256':p['sources_sha256'],'first_preflight_started_utc':p['first_preflight_started_utc'],'absolute_deadline_utc':p['absolute_deadline_utc'],
                    'children':{k:{'path':'child.json' if k==key else k+'.json','sha256':c.sha256_file(cp) if k==key else str(i%10)*64} for i,k in enumerate(keys)}}
            pp=root/'parent.json';pp.write_text(json.dumps(parent));psha=c.sha256_file(pp)
            def put(name,value):
                path=root/(name+'.json');path.write_text(json.dumps(value));return {'path':str(path),'sha256':c.sha256_file(path)}
            cpu=put('cpu',{'passed':True,'tests_run':1,'skipped':0,'failures':0,'errors':0,'cuda_initialized':False,'policy_sha256':psha,'sources_sha256':p['sources_sha256'],
                           'first_preflight_started_utc':p['first_preflight_started_utc'],'absolute_deadline_utc':p['absolute_deadline_utc'],'native_watchdog_probe':{'passed':True}})
            qrefs={}
            for mode in ('formal','smoke'):
                for flow in c.FLOWS:
                    qkey=('smoke_' if mode=='smoke' else '')+flow+'_raw_sft'
                    qrefs[mode+'_'+flow]=put(mode+'_'+flow,{'passed':True,'status':'complete_verified_reference_refresh_q0','new_q0_candidate_forwards':0,'mode':mode,'flow':flow,'arm':'raw_sft','unique':4 if mode=='smoke' else 256,'locked_policy_sha256':parent['children'][qkey]['sha256']})
            tech={}
            for name in ['smoke_'+x for x in c.ARMS]+list(c.PATH_SMOKES):
                mode='path_smoke' if name.startswith('path_smoke_') else 'smoke';flow,arm=name[len(mode)+1:].split('_',1)
                tech[name]=put(name,{'passed':True,'status':'complete_verified_reference_refresh_run','mode':mode,'flow':flow,'arm':arm,'locked_policy_sha256':parent['children'][name]['sha256']})
            auth={'schema_version':1,'phase':'formal','child_key':key,'child_policy_sha256':c.sha256_file(cp),'stage_policy_path':'parent.json','stage_policy_sha256':psha,'issued_utc':datetime.now(timezone.utc).isoformat(),
                  'first_preflight_started_utc':p['first_preflight_started_utc'],'absolute_deadline_utc':p['absolute_deadline_utc'],'cpu_receipt':cpu,'q0_receipts':qrefs,'technical_receipts':tech,
                  'whole_gate_receipt':put('gate',{'passed':True,'all_eight_formal_arms_required':True})}
            ap=root/'auth.json';ap.write_text(json.dumps(auth))
            with patch.object(t,'ROOT',root),patch.dict(t.os.environ,{'RECIPEQA_REFRESH_OUTER_WATCHDOG':psha}):
                self.assertEqual(t.validate_authorization(p,cp)['child_key'],key)
                for role in ('q0','technical'):
                    bad=deepcopy(auth)
                    if role=='q0':bad['q0_receipts']['formal_flow44']=bad['q0_receipts']['formal_flow45']
                    else:bad['technical_receipts']['smoke_flow44_raw_sft']=bad['technical_receipts']['smoke_flow45_raw_sft']
                    ap.write_text(json.dumps(bad))
                    with self.subTest(role=role),self.assertRaisesRegex(ValueError,'receipt role'):t.validate_authorization(p,cp)
                bad=deepcopy(auth);bad['technical_receipts'].pop('path_smoke_flow45_refresh_beta');ap.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError,'All ten'):t.validate_authorization(p,cp)

    def test_initial_cpu_suite_does_not_import_torch(self):
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main()
