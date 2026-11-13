"""Focused stdlib contracts; realautograd is a separate mandatory remote suite."""
from __future__ import annotations
import ast
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import random
import sys
import unittest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'src'))
from recipeqa.modeling import hash_json,read_jsonl,sha256_file
from coverage_contract import *


def data_root():return ROOT if (ROOT/FULL_TRAIN).is_file() else ROOT/'.state/remote_principled_20261105'


def child_fixture(mode='formal',flow='flow44',method='exact_kl'):
    cli={'method':method,'model_path':str(BASE/MODEL_PATH),'adapter_path':str(BASE/INITIAL_ADAPTER),'revision':REVISION,
        'manifest':str(BASE/f'research/coverage_20261106/prepared/{"smoke" if mode=="smoke" else "formal"}_{flow}.jsonl'),
        'limit':4 if mode=='smoke' else 1024,'max_steps':1 if mode=='smoke' else 256,'save_steps':1 if mode=='smoke' else 64,
        'seed':int(flow[-2:]),'learning_rate':5e-6,'beta':BETA if method=='exact_kl' else 0.,
        'prepare_reference_only':mode=='reference','reference_cache':None if mode=='reference' else str(BASE/'new_reference.jsonl'),
        'min_pixels':65536,'max_pixels':262144,'recipe_token_budget':1536,'max_input_tokens':4096,'logp_atol':.01,'wall_seconds':WALL_SECONDS}
    return {'status':'locked_before_coverage_training','mode':mode,'flow':flow,'method':method,'cli':cli,
        'sources_sha256':{p:'0'*64 for p in TRAIN_SOURCE_PATHS},'reward':'unique_complete_order_exact_0_1',
        'selection':'fixed256_primary_fixed64_diagnostic_no_dev_selection',
        'optimizer':{'accumulation':4,'clip_norm':1.,'canonical_C_including_eos':5,'fresh_adamw':True},
        'first_preflight_started_utc':'2026-11-06T12:00:00Z','batch_clock_started_utc':'2026-11-06T12:00:00Z'}


class CoverageContractTests(unittest.TestCase):
    def test_actual_known_pool_order_and_completed_prior_exposure(self):
        root=data_root();self.assertEqual(sha256_file(root/FULL_TRAIN),FULL_SHA)
        pool=fixed_pool(read_jsonl(root/FULL_TRAIN));orders=ordered_pools(pool)
        self.assertEqual(len({r['recipe_id'] for r in pool}),963)
        self.assertEqual(orders['flow44'],pool)
        shuffled=deepcopy(pool);random.Random(45).shuffle(shuffled);self.assertEqual(orders['flow45'],shuffled)
        self.assertEqual({hash_json(r) for r in orders['flow44']},{hash_json(r) for r in orders['flow45']})
        folder=root/'exploratory/principled_ordering_20261105/full_sft/train'
        signed=json.loads((folder/'run_signature.json').read_text());self.assertEqual(signed['sha256'],hash_json(signed['payload']))
        visits=read_jsonl(folder/'actual_prompt_visits.jsonl')
        receipt=validate_prior_exposure(pool,visits,signed['sha256']);self.assertEqual(receipt['already_consumed_pool_questions'],1024)
        broken=deepcopy(visits);broken[0]['consumption_point']='prepared_not_consumed'
        with self.assertRaises(ValueError):validate_prior_exposure(pool,broken,signed['sha256'])

    def test_fixed_pool_rejects_heldout_and_duplicate_without_resampling(self):
        rows=read_jsonl(data_root()/FULL_TRAIN);bad=deepcopy(rows);bad[0]['source_split']='val'
        with self.assertRaises(ValueError):fixed_pool(bad)
        bad=deepcopy(rows);bad[1]=bad[0]
        with self.assertRaises(ValueError):fixed_pool(bad)

    def test_child_budget_and_seed_path_contract(self):
        validate_child(child_fixture())
        validate_child(child_fixture('smoke'))
        validate_child(child_fixture('reference',method='raw_sft'))
        for field,value in [('limit',256),('max_steps',64),('save_steps',16),('seed',45),('beta',.1),('learning_rate',1e-4)]:
            bad=child_fixture();bad['cli'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_child(bad)

    def test_child_cannot_reset_clock_change_scoring_or_drop_source(self):
        bad=child_fixture();bad['batch_clock_started_utc']='2026-11-06T13:00:00Z'
        with self.assertRaises(ValueError):validate_child(bad)
        bad=child_fixture();bad['sources_sha256'].pop('scripts/predict.py')
        with self.assertRaises(ValueError):validate_child(bad)
        bad=child_fixture();bad['cli']['manifest']=str(BASE/'formal_flow45.jsonl')
        with self.assertRaises(ValueError):validate_child(bad)

    @staticmethod
    def reference_fixture():
        records=[{'qid':f'q{i}','presentation_id':f'p{i}','display_perm':[0,1,2,3],
            'image_paths_display':[f'{i}-{j}.jpg' for j in range(4)]} for i in range(1024)]
        spec={'manifest_sha256':'original','selected_records_sha256':hash_json(records),
            'pixel_sha256':{p:'pixels' for r in records for p in r['image_paths_display']},
            'candidate_texts':['opaque fixed24'],'initial_identity':{'fixed':'weights'}}
        rows=[{'index':i,'qid':r['qid'],'presentation_id':r['presentation_id'],
            'reference_spec_sha256':hash_json(spec),'token_logps':[['opaque unchanged scores']],
            'numerical_checks':[{'literal_all_position_logits':i==0}],'input_audit':{'fixed':i}}
            for i,r in enumerate(records)]
        return rows,spec,records

    def test_reorder_preserves_raw_numeric_checks_and_only_two_row_fields_change(self):
        rows,spec,records=self.reference_fixture();target=deepcopy(records);random.Random(45).shuffle(target)
        new,new_spec,lineage=reference_reindex(rows,spec,records,target,'targetmanifest')
        self.assertEqual(len(new),1024);self.assertEqual(lineage['allowed_row_changes'],['index','reference_spec_sha256'])
        bank={identity(r):r for r in rows}
        for i,row in enumerate(new):
            original=bank[identity(row)]
            self.assertEqual({k:v for k,v in row.items() if k not in GENERATED_REFERENCE_ALLOWED_ROW_FIELDS},
                {k:v for k,v in original.items() if k not in GENERATED_REFERENCE_ALLOWED_ROW_FIELDS})
            self.assertEqual(row['index'],i)
        self.assertEqual(sum(r['numerical_checks'][0]['literal_all_position_logits'] for r in new),1)
        self.assertEqual({k for k in spec if spec[k]!=new_spec[k]}, {'manifest_sha256','selected_records_sha256'})

    def test_smoke_subset_pixels_and_lineage_use_actual_source_index(self):
        rows,spec,records=self.reference_fixture();target=[records[i] for i in (8,2,0,10)]
        new,new_spec,lineage=reference_reindex(rows,spec,records,target,'smokemanifest')
        self.assertEqual([r['source_index'] for r in lineage['rows']],[8,2,0,10])
        self.assertEqual([r['numerical_checks'][0]['literal_all_position_logits'] for r in new],[False,False,True,False])
        self.assertEqual(len(new_spec['pixel_sha256']),16)

    def test_reindex_forbids_record_display_modification_or_duplicate_target(self):
        rows,spec,records=self.reference_fixture();bad=deepcopy(records[:4]);bad[0]['display_perm']=[3,2,1,0]
        with self.assertRaises(ValueError):reference_reindex(rows,spec,records,bad,'bad')
        with self.assertRaises(ValueError):reference_reindex(rows,spec,records,[records[0]]*4,'bad')
        rows[0]['index']=17
        with self.assertRaises(ValueError):reference_reindex(rows,spec,records,records[:4],'bad')

    def test_no_fresh_clock_on_retry(self):
        start=datetime(2026,11,6,12,tzinfo=timezone.utc)
        policy={'first_preflight_started_utc':start.isoformat(),'absolute_deadline_utc':(start+timedelta(hours=16)).isoformat()}
        self.assertEqual(remaining_seconds(policy,start+timedelta(hours=15)),3600)
        with self.assertRaises(ValueError):remaining_seconds(policy,start+timedelta(hours=16))
        with self.assertRaises(ValueError):remaining_seconds(policy,start-timedelta(seconds=1))

    def test_training_math_and_entire_update_loop_identical_to_frozen_source(self):
        old=ast.parse((ROOT/'exploratory/principled_ordering_20261105/finite_policy_train.py').read_text())
        new=ast.parse((HERE/'coverage_train.py').read_text())
        for name in ('log_policy','categorical_draws','policy_coefficients','append','pixel_identities','cpu_gradient_norm','prepare','save_checkpoint'):
            left=[n for n in ast.walk(old) if isinstance(n,ast.FunctionDef) and n.name==name]
            right=[n for n in ast.walk(new) if isinstance(n,ast.FunctionDef) and n.name==name]
            self.assertEqual(len(left),1);self.assertEqual(ast.dump(left[0],include_attributes=False),ast.dump(right[0],include_attributes=False),name)
        def loop(tree):return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='step' and ast.unparse(n.iter)=='range(max_steps)')
        self.assertEqual(ast.dump(loop(old),include_attributes=False),ast.dump(loop(new),include_attributes=False))


if __name__=='__main__':unittest.main(verbosity=2)
