"""Focused independent coverage-audit boundary tests; no Torch/trainer/GPU."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from verify_coverage_run import (ROOT,INITIAL_ADAPTER,REVISION,BETA,WALL_SECONDS,LEGACY_SHA,OLD_VERIFIER_SHA,
    REQUIRED_SOURCE_PATHS,audit_child,audit_lineage_rows,audit_numerical_checks,audit_state_file,
    audit_effective_update,tensor_fingerprint,hash_json)


def child(mode='formal',flow='flow44',method='raw_sft'):
    now=datetime(2026,11,6,8,tzinfo=timezone.utc)
    cli={'method':method,'seed':int(flow[-2:]),'limit':4 if mode=='smoke' else 1024,
        'max_steps':1 if mode=='smoke' else 256,'save_steps':1 if mode=='smoke' else 64,
        'learning_rate':5e-6,'beta':BETA if method=='exact_kl' else 0.,'prepare_reference_only':mode=='reference',
        'reference_cache':None if mode=='reference' else str(ROOT/'exploratory/coverage_20261106/fixture/ref.jsonl'),
        'revision':REVISION,'logp_atol':.01,'min_pixels':65536,'max_pixels':262144,'recipe_token_budget':1536,
        'max_input_tokens':4096,'wall_seconds':WALL_SECONDS,'adapter_path':str(ROOT/INITIAL_ADAPTER)}
    sources={p:'0'*64 for p in REQUIRED_SOURCE_PATHS};sources.update(LEGACY_SHA)
    sources['exploratory/principled_ordering_20261105/verify_finite_run.py']=OLD_VERIFIER_SHA
    return {'status':'locked_before_coverage_training','mode':mode,'method':method,'flow':flow,'cli':cli,
        'sources_sha256':sources,'reward':'unique_complete_order_exact_0_1',
        'selection':'fixed256_primary_fixed64_diagnostic_no_dev_selection',
        'optimizer':{'accumulation':4,'clip_norm':1.,'canonical_C_including_eos':5,'fresh_adamw':True},
        'batch_clock_started_utc':now.isoformat(),'first_preflight_started_utc':now.isoformat(),
        'absolute_deadline_utc':(now+timedelta(seconds=WALL_SECONDS)).isoformat(),
        'initial_identity':{'adapter':{'directory':str(ROOT/INITIAL_ADAPTER)}},
        'reference_lineage':None if mode=='reference' or (mode=='formal' and flow=='flow44') else 'fixture.lineage.json'}


def lineage_fixture(indices):
    records=[{'qid':f'q{i}','presentation_id':f'p{i}','recipe_id':f'r{i}',
        'image_paths_display':[f'{i}_{j}.jpg' for j in range(4)],'display_perm':[3,2,1,0],
        'target_labels':['D','C','B','A'],'visit_id':f'unchanged{i}'} for i in range(1024)]
    spec={'candidate_texts':['fixture full canonical grammar'],'numerical_reference_atol':.01,
        'manifest_sha256':'source_manifest','selected_records_sha256':hash_json(records),
        'pixel_sha256':{p:'a'*64 for r in records for p in r['image_paths_display']}}
    spec_sha=hash_json(spec)
    source=[{'index':i,'qid':r['qid'],'presentation_id':r['presentation_id'],'reference_spec_sha256':spec_sha,
        'scores':[-float(i),-.1],'input_audit':{'opaque_actual_input_hash':f'hash{i}'},
        'numerical_checks':[{'candidate_index':0,'literal_all_position_logits':i==0,
            'max_token_logp_abs_difference':0.,'reference_token_logps':[-1.,-2.,-3.,-4.,-5.]}]}
        for i,r in enumerate(records)]
    target_records=[deepcopy(records[i]) for i in indices]
    target_spec=deepcopy(spec);target_spec.update(manifest_sha256='target_manifest',selected_records_sha256=hash_json(target_records),
        pixel_sha256={p:spec['pixel_sha256'][p] for r in target_records for p in r['image_paths_display']})
    target_spec_sha=hash_json(target_spec)
    target=[{**deepcopy(source[i]),'index':n,'reference_spec_sha256':target_spec_sha} for n,i in enumerate(indices)]
    entries=[]
    for n,i in enumerate(indices):
        unchanged={k:v for k,v in source[i].items() if k not in ('index','reference_spec_sha256')}
        entries.append({'index':n,'source_index':i,'qid':records[i]['qid'],'presentation_id':records[i]['presentation_id'],
            'record_sha256':hash_json(records[i]),'source_row_sha256':hash_json(source[i]),'target_row_sha256':hash_json(target[n]),
            'unchanged_payload_sha256':hash_json(unchanged),'reference_literal_check_was_source_index_zero':i==0})
    lineage={'source_spec_sha256':hash_json(spec),'target_spec_sha256':hash_json(target_spec),
        'source_selected_records_sha256':hash_json(records),'target_selected_records_sha256':hash_json(target_records),
        'source_rows':1024,'target_rows':len(indices),'target_manifest_sha256':'target_manifest',
        'allowed_row_changes':['index','reference_spec_sha256'],
        'allowed_spec_changes':['manifest_sha256','pixel_sha256','selected_records_sha256'],'rows':entries}
    return source,spec,records,target,target_spec,target_records,lineage


class ChildContractTests(unittest.TestCase):
    def test_all_four_formal_and_smoke_budgets(self):
        for flow in ('flow44','flow45'):
            for method in ('raw_sft','exact_kl'):
                self.assertEqual(audit_child(child('formal',flow,method)),(1024,256,64))
                self.assertEqual(audit_child(child('smoke',flow,method)),(4,1,1))
        self.assertEqual(audit_child(child('reference')),(1024,256,64))

    def test_old256_or_old_save16_or_truncated_update_rejected(self):
        for key,value in [('limit',256),('max_steps',64),('save_steps',16)]:
            bad=child();bad['cli'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):audit_child(bad)

    def test_seed_beta_lr_C_clip_or_selection_cannot_change(self):
        cases=[]
        for key,value in [('seed',45),('learning_rate',1e-5),('beta',.1)]:
            bad=child();bad['cli'][key]=value;cases.append(bad)
        for key,value in [('canonical_C_including_eos',4),('clip_norm',2.),('fresh_adamw',False)]:
            bad=child();bad['optimizer'][key]=value;cases.append(bad)
        bad=child();bad['selection']='best_dev';cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):audit_child(bad)

    def test_missing_or_changed_dependency_rejected(self):
        bad=child();bad['sources_sha256'].pop('research/coverage_20261106/test_verify_coverage_run.py')
        with self.assertRaises(ValueError):audit_child(bad)
        bad=child();bad['sources_sha256']['exploratory/principled_ordering_20261105/finite_policy_train.py']='f'*64
        with self.assertRaises(ValueError):audit_child(bad)
        bad=child();bad['sources_sha256']['exploratory/principled_ordering_20261105/verify_finite_run.py']='f'*64
        with self.assertRaises(ValueError):audit_child(bad)

    def test_draft_missing_lineage_or_clock_reset_rejected(self):
        bad=child();bad['status']='draft'
        with self.assertRaises(ValueError):audit_child(bad)
        bad=child(flow='flow45');bad['reference_lineage']=None
        with self.assertRaises(ValueError):audit_child(bad)
        bad=child();bad['first_preflight_started_utc']='2026-11-06T08:00:01+00:00'
        with self.assertRaises(ValueError):audit_child(bad)
        bad=child();bad['absolute_deadline_utc']='2026-11-07T01:00:00+00:00'
        with self.assertRaises(ValueError):audit_child(bad)


class ReferenceLineageTests(unittest.TestCase):
    def test_full1024_reorder_and_four_subset_preserve_original_index_checks(self):
        for indices in (list(reversed(range(1024))),[3,0,1023,12]):
            fixture=lineage_fixture(indices)
            self.assertEqual(audit_lineage_rows(*fixture),indices)
            for origin,row in zip(indices,fixture[3]):
                audit_numerical_checks(row['numerical_checks'],origin==0,.01,[[-1.,-2.,-3.,-4.,-5.]])
        fixture=lineage_fixture([3,0,1023,12])
        with self.assertRaises(ValueError):audit_numerical_checks(fixture[3][0]['numerical_checks'],True,.01,[[-1.,-2.,-3.,-4.,-5.]])

    def test_changed_score_or_candidate_check_payload_rejected(self):
        for field,value in [('scores',[999.,-.1]),('input_audit',{'opaque_actual_input_hash':'changed'}),('numerical_checks',[])]:
            fixture=lineage_fixture([3,0,1023,12]);fixture[3][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):audit_lineage_rows(*fixture)

    def test_changed_record_display_gold_or_visitid_rejected(self):
        for key,value in [('display_perm',[0,1,2,3]),('target_labels',['A','B','C','D']),('visit_id','new')]:
            fixture=lineage_fixture([3,0,1023,12]);fixture[5][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):audit_lineage_rows(*fixture)

    def test_changed_nonpermitted_spec_or_lineage_hash_rejected(self):
        fixture=lineage_fixture([3,0,1023,12]);fixture[4]['numerical_reference_atol']=.1
        with self.assertRaises(ValueError):audit_lineage_rows(*fixture)
        fixture=lineage_fixture([3,0,1023,12]);fixture[6]['rows'][0]['source_row_sha256']='f'*64
        with self.assertRaises(ValueError):audit_lineage_rows(*fixture)

    def test_duplicate_missing_or_extra_lineage_rows_rejected(self):
        fixture=lineage_fixture([3,0,1023,12]);fixture[5][-1]=deepcopy(fixture[5][0])
        with self.assertRaises(ValueError):audit_lineage_rows(*fixture)
        fixture=lineage_fixture([3,0,1023,12]);fixture[6]['rows'].pop()
        with self.assertRaises(ValueError):audit_lineage_rows(*fixture)


class PhysicalEvidenceTests(unittest.TestCase):
    def test_physical_safetensor_fingerprint_and_truncation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);name='base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight'
            inventory=[{'name':name,'shape':[1,2],'numel':2}];raw=struct.pack('<ff',1.,-2.)
            header=json.dumps({name.replace('.default.','.'): {'dtype':'F32','shape':[1,2],'data_offsets':[0,8]}}).encode()
            p=root/'adapter_model.safetensors';p.write_bytes(struct.pack('<Q',len(header))+header+raw)
            self.assertEqual(tensor_fingerprint(root,inventory),hashlib.sha256(name.encode()+raw).hexdigest())
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):tensor_fingerprint(root,inventory)

    def test_zip_crc_and_missing_state_payload_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)/'optimizer.pt'
            with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_STORED) as z:z.writestr('optimizer/data.pkl',b'valid_payload_fixture')
            audit_state_file(p)
            raw=p.read_bytes();offset=raw.index(b'valid_payload_fixture');raw=raw[:offset]+b'X'+raw[offset+1:];p.write_bytes(raw)
            with self.assertRaises(ValueError):audit_state_file(p)
            with zipfile.ZipFile(p,'w') as z:z.writestr('readme',b'not a Torch container')
            with self.assertRaises(ValueError):audit_state_file(p)

    def test_no_effective_update_is_valid_when_disclosed_honestly(self):
        self.assertFalse(audit_effective_update('same','same',False,'no_effective_adapter_update'))
        with self.assertRaises(ValueError):audit_effective_update('same','same',True,'effective_training')


if __name__=='__main__':
    if 'torch' in sys.modules:raise RuntimeError('Independent coverage verifier suite imported Torch')
    unittest.main(verbosity=2)
