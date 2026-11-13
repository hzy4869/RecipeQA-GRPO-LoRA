#!/usr/bin/env python3
"""Independent CPU acceptance math and persisted-state checks for reference refresh.

No producer imports, model construction, generation, optimizer execution or network.
File/schema integration is completed against CONTRACT_DRAFT before protocol freeze.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime
from itertools import permutations
from pathlib import Path
import argparse
from array import array
import hashlib
import json
import math
import os
import random
import re
import struct
import sys
import zipfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REMOTE = Path('/root/autodl-tmp/rl_recipeqa')
BETA = 1.6363096678310884
ACTIONS = tuple(' '.join(p) for p in permutations('ABCD'))
ARMS = ('raw_sft', 'fixed_beta', 'fixed_half_beta', 'refresh_beta')
MODES = {'smoke':(4,1,4,1,1), 'path_smoke':(4,2,8,2,1), 'formal':(256,2,512,128,64)}
INPUT = {'min_pixels':65536,'max_pixels':262144,'recipe_token_budget':1536,
         'max_input_tokens':4096,'max_completion_tokens':32,'prompt_version':'recipeqa-order-v1'}
OLD_SCORING = {
    'exploratory/principled_ordering_20261105/score_permutations.py':'affeda80416145d2b6a134172e812d15d48a86d77c67128bd2e5f84e0c341819',
    'exploratory/principled_ordering_20261105/gradient_alignment.py':'1adff082850e0b83a9a70c83565638c02b8efeebdf0fb0ade49af937df9fe41b',
    'src/recipeqa/modeling.py':'5fedb7a60877e965861aa9174179880754cb0555c4167265fdd8c3781af39771',
    'src/recipeqa/metrics.py':'5287d44a3bb4ede3a44ce0a6625aee399260afecd42c984441a8e07492a2423a'}
FULL_TRAIN = 'exploratory/principled_ordering_20261105/full_train_data/train_all7558.jsonl'
FULL_SHA = 'b9979ee0612679d2e9460dd3bb8a6f68e7dea6794bc6cdcc3e1defdc87d8d949'
REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
SOURCE_PATHS = {
    'research/reference_refresh_20261107/'+name for name in (
        'refresh_train.py','refresh_contract.py','refresh_pipeline.py','build_draft_policy.py',
        'launch_stage.py','run_cpu_preflight.py','test_refresh_contract.py','test_refresh_pipeline.py',
        'verify_refresh_run.py','test_verify_refresh_run.py')}
SOURCE_PATHS.update(OLD_SCORING)
SOURCE_PATHS.update({'research/grounding_equivariance_20261106/orbit_diagnostic.py','scripts/predict.py',
                     'scripts/validate_run.py','exploratory/principled_ordering_20261105/finite_policy_train.py',
                     'exploratory/principled_ordering_20261105/verify_finite_run.py',
                     'exploratory/principled_ordering_20261105/test_finite_policy_tensor_reference.py'})


def require(condition, label):
    if not condition:
        raise ValueError(label)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def object_sha(value):
    return sha(json.dumps(value,sort_keys=True,ensure_ascii=False).encode())


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def close(actual, expected, label, atol=1e-8):
    require(finite(actual) and finite(expected)
            and math.isclose(actual,expected,rel_tol=1e-8,abs_tol=atol), 'Numeric mismatch: '+label)


def vector_close(actual, expected, label, atol=1e-8):
    require(len(actual)==len(expected),'Vector length mismatch: '+label)
    for i,(a,b) in enumerate(zip(actual,expected)):
        close(a,b,label+'['+str(i)+']',atol)


def utc(value):
    stamp=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(stamp.tzinfo is not None and stamp.utcoffset().total_seconds()==0,'Explicit UTC required')
    return stamp


class Files:
    """Explicit-root reads, streaming physical hashing and complete receipts."""
    def __init__(self,root):
        self.root=Path(root).resolve();self.inputs={};self.content={};self.mutable_inputs=set()

    def relative(self,name):
        path=Path(name)
        if path.is_absolute():
            path=path.relative_to(REMOTE) if path.is_relative_to(REMOTE) else path.relative_to(self.root)
        require('..' not in path.parts and str(path) not in ('','.'),'Unsafe evidence path')
        require(not any(part.lower().startswith('confirmation') for part in path.parts),'Confirmation data forbidden')
        if path.parts[:2]==('data','manifests'):
            require(path.name=='val.jsonl','Only declared development manifest is allowed')
        return str(path)

    def path(self,name):
        path=(self.root/self.relative(name)).resolve()
        require(path.is_relative_to(self.root),'Evidence escaped root')
        return path

    def physical(self,name):
        path=self.path(name);h=hashlib.sha256();size=0
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(1024**2),b''):
                h.update(block);size+=len(block)
        receipt={'sha256':h.hexdigest(),'bytes':size};key=self.relative(name)
        require(key not in self.inputs or self.inputs[key]==receipt,'File changed during audit: '+key)
        self.inputs[key]=receipt
        return receipt

    def read(self,name):
        key=self.relative(name)
        if key not in self.content:
            raw=self.path(name).read_bytes();self.content[key]=raw
            receipt={'sha256':sha(raw),'bytes':len(raw)}
            require(key not in self.inputs or self.inputs[key]==receipt,'Input changed before read')
            self.inputs[key]=receipt
        return self.content[key]

    def json(self,name):return json.loads(self.read(name))
    def rows(self,name):return [json.loads(line) for line in self.read(name).splitlines() if line.strip()]
    def sha(self,name):return self.physical(name)['sha256']

    def stable(self):
        for name,receipt in list(self.inputs.items()):
            require(self.physical(name)==receipt,'Input changed during complete audit')

    def report_receipts(self):
        return ({name:item for name,item in self.inputs.items() if name not in self.mutable_inputs},
                {name:item for name,item in self.inputs.items() if name in self.mutable_inputs})


def verify_identity(files,identity):
    for kind in ('base','adapter'):
        bundle=identity[kind]
        root=files.path(bundle['directory'])
        patterns=('adapter_model*.safetensors','adapter_model*.bin') if kind=='adapter' else ('model*.safetensors','pytorch_model*.bin')
        actual_weights={path.name for pattern in patterns for path in root.glob(pattern) if path.is_file()}
        auxiliary=('config.json','adapter_config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
                   'special_tokens_map.json','vocab.json','merges.txt','chat_template.jinja','chat_template.json',
                   'preprocessor_config.json','processor_config.json')
        require(actual_weights=={item['name'] for item in bundle['weight_files']}
                and {name for name in auxiliary if (root/name).is_file()}=={item['name'] for item in bundle['auxiliary_files']},
                'Physical model directory has missing/additional load-relevant files')
        require(bundle['weights_sha256']==object_sha(bundle['weight_files'])
                and bundle['bundle_sha256']==object_sha({'weights':bundle['weight_files'],
                                                         'auxiliary':bundle['auxiliary_files']}),
                'Model identity structure hash differs')
        require(bundle['weight_files'],'Missing actual model weight files')
        for spec in bundle['weight_files']+bundle['auxiliary_files']:
            actual=files.physical(Path(bundle['directory'])/spec['name'])
            require(actual['sha256']==spec['sha256'] and actual['bytes']>0
                    and ('bytes' not in spec or actual['bytes']==spec['bytes']),'Physical model file differs')


def safetensor_fingerprint(files,directory,inventory):
    """Independent safe-byte reproduction of actual float32 LoRA fingerprint."""
    path=files.path(Path(directory)/'adapter_model.safetensors')
    files.physical(path)
    with path.open('rb') as handle:
        prefix=handle.read(8);require(len(prefix)==8,'Truncated safetensors')
        length=struct.unpack('<Q',prefix)[0]
        require(0<length<=16*1024**2,'Invalid safetensors header')
        header=json.loads(handle.read(length))
        tensors={name:item for name,item in header.items() if name!='__metadata__'}
        require(set(tensors)=={row['name'].replace('.default.','.') for row in inventory},
                'Missing/additional saved LoRA tensors')
        intervals=sorted(item['data_offsets'] for item in tensors.values())
        require(intervals and intervals[0][0]==0 and all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
                and intervals[-1][1]==path.stat().st_size-8-length,'Unchecked/overlapping safetensor bytes')
        digest=hashlib.sha256()
        for row in inventory:
            entry=tensors[row['name'].replace('.default.','.')]
            begin,end=entry['data_offsets'];dtype=entry['dtype']
            size={'F32':4,'BF16':2,'F16':2,'F64':8}.get(dtype)
            require(entry['shape']==row['shape'] and math.prod(row['shape'])==row['numel']
                    and size and end-begin==size*row['numel'],'Saved LoRA shape/dtype/bytes differ')
            handle.seek(8+length+begin);raw=handle.read(end-begin)
            require(len(raw)==end-begin,'Truncated tensor data')
            if dtype=='F32':converted=raw
            elif dtype=='BF16':converted=b''.join(struct.pack('<I',x[0]<<16) for x in struct.iter_unpack('<H',raw))
            else:converted=b''.join(struct.pack('<f',x[0]) for x in struct.iter_unpack('<e' if dtype=='F16' else '<d',raw))
            values=array('f');values.frombytes(converted)
            if sys.byteorder!='little':values.byteswap()
            require(all(math.isfinite(x) for x in values),'Nonfinite saved LoRA tensor')
            digest.update(row['name'].encode());digest.update(converted)
    return digest.hexdigest()


def torch_archive(files,name):
    path=files.path(name);files.physical(name)
    require(path.stat().st_size>0 and zipfile.is_zipfile(path),'Unrecognized Torch state archive')
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None and any(n.endswith('/data.pkl') for n in archive.namelist()),
                'Damaged/incomplete Torch state archive')


def audit_child_policy(files,policy_path):
    policy=files.json(policy_path);cli=policy['cli'];mode=policy['mode'];arm=policy['arm'];flow=policy['flow']
    require(policy['status']=='locked_before_reference_refresh_training' and mode in MODES
            and arm in ARMS and flow in ('flow44','flow45')
            and (mode!='path_smoke' or arm=='refresh_beta'),'Unknown/draft child policy')
    unique,passes,visits,steps,save=MODES[mode]
    for key,expected in {'unique_count':unique,'passes':passes,'limit':visits,'max_steps':steps,
                         'save_steps':save,'final_bank_count':unique,'seed':int(flow[-2:]),'learning_rate':5e-6}.items():
        require(cli[key]==expected,'Fixed child budget/optimizer differs: '+key)
    method='raw_sft' if arm=='raw_sft' else 'exact_kl'
    beta=0. if arm=='raw_sft' else BETA/2 if arm=='fixed_half_beta' else BETA
    boundary=None if arm!='refresh_beta' or mode=='smoke' else steps//2
    require(cli['method']==method and cli['beta']==beta and cli['refresh_boundary_step']==boundary,
            'Fixed arm objective/refresh differs')
    require(cli['revision']==REVISION and files.relative(cli['model_path'])=='models/Qwen3-VL-2B-Instruct'
            and files.relative(cli['adapter_path'])=='exploratory/principled_ordering_20261105/full_sft/train/checkpoint-768',
            'Wrong initialization/model revision')
    require(policy['optimizer']=={'accumulation':4,'clip_norm':1.,'canonical_C_including_eos':5,'fresh_adamw':True}
            and policy['reward']=='unique_complete_order_exact_0_1'
            and policy['selection']=='fixed128_primary_step64_boundary_no_dev_selection', 'Wrong optimizer/reward/selection')
    require(policy['batch_clock_started_utc']==policy['first_preflight_started_utc']
            and 0<(utc(policy['absolute_deadline_utc'])-utc(policy['first_preflight_started_utc'])).total_seconds()<=72000,
            'Original20h clock missing/reset')
    for field in ('min_pixels','max_pixels','recipe_token_budget','max_input_tokens'):
        require(cli[field]==INPUT[field],'Input budget differs')
    require(cli['logp_atol']==.01 and cli['prepare_reference_only'] is False
            and 0<cli['wall_seconds']<=72000,'Wrong tolerance/reference-only/runtime')
    require(set(policy['sources_sha256'])==SOURCE_PATHS,'Incomplete/additional frozen source closure')
    for group in ('sources_sha256','artifacts_sha256'):
        for name,digest in policy[group].items():
            require(files.sha(name)==digest,'Frozen source/artifact differs: '+name)
    for name,digest in OLD_SCORING.items():
        require(policy['sources_sha256'][name]==digest,'Old scoring helper pin differs')
    own='research/reference_refresh_20261107/verify_refresh_run.py'
    require(sha(Path(__file__).read_bytes())==policy['sources_sha256'][own],'Executing verifier is not frozen source')
    return policy


def audit_pool(files,policy,records):
    require(files.sha(FULL_TRAIN)==FULL_SHA,'Original7558 bytes differ')
    full=files.rows(FULL_TRAIN);require(len(full)==7558,'Original7558 denominator differs')
    ordered=full[:256]
    require(len({r['recipe_id'] for r in ordered})==254,'Train recipe clusters differ')
    if policy['flow']=='flow45':
        ordered=ordered.copy();random.Random(45).shuffle(ordered)
    require(records==ordered[:policy['cli']['unique_count']],'Not fixed same256 pool and declared flow order')
    spec=policy['pool_spec']
    visits=files.rows(spec['prior_visits_path']);signed=files.json(spec['prior_signature_path'])
    require(files.sha(spec['prior_visits_path'])==spec['prior_visits_sha256']
            and files.sha(spec['prior_signature_path'])==spec['prior_signature_sha256']
            and signed['sha256']==object_sha(signed['payload'])==spec['prior_run_signature_sha256'],
            'Initial SFT actual exposure provenance differs')
    consumed=[r for r in visits if r['global_step_before_update']<768]
    require(len(consumed)==6144 and len({(r['qid'],r['presentation_id']) for r in consumed})==6144
            and Counter(r['global_step_before_update'] for r in consumed)==Counter({i:8 for i in range(768)})
            and all(r['consumption_point']=='training_step_forward_backward_returned'
                    and r['run_signature_sha256']==signed['sha256'] for r in consumed), 'Initial SFT768 actual consumed6144 proof differs')
    require({(r['qid'],r['presentation_id']) for r in full[:256]} <=
            {(r['qid'],r['presentation_id']) for r in consumed},'Pool includes unseen initial-SFT question')


def audit_q0(files,policy,policy_path):
    cli=policy['cli'];records=files.rows(cli['manifest'])
    verify_two_pass_records(records,cli['unique_count'],cli['passes'])
    require(files.sha(cli['manifest'])==cli['expected_manifest_sha256'],'Manifest bytes differ')
    audit_pool(files,policy,records)
    cache=cli['reference_cache'];config=files.json(str(cache)+'.config.json')
    done=files.json(str(cache)+'.completion.json');lineage=files.json(str(cache)+'.lineage.json')
    target=files.rows(cache);source=files.rows(lineage['source_cache'])
    original_config=files.json(str(lineage['source_cache'])+'.config.json')
    original_done=files.json(str(lineage['source_cache'])+'.completion.json')
    source_records=files.rows(lineage['source_manifest'])
    original_signature=files.json(Path(lineage['source_cache']).parent/'run_signature.json')
    original_audit=files.json(Path(lineage['source_cache']).parent/'independent_verification.json')
    require(len(source)==len(source_records)==original_done['rows']==1024
            and original_done['status']=='complete_fixed_initial_reference'
            and original_done['cache_sha256']==files.sha(lineage['source_cache'])
            and original_done['reference_spec_sha256']==object_sha(original_config)
            and original_audit['passed'] is True and original_audit['updates']==0
            and original_audit['status']=='complete_verified_coverage_reference_only'
            and original_signature['sha256']==object_sha(original_signature['payload'])
            and original_audit['run_signature_sha256']==original_signature['sha256'], 'Old actual1024 reference proof incomplete')
    require(original_config['initial_identity']==policy['initial_identity']
            and original_config['input']==INPUT and original_config['revision']==REVISION
            and original_config['scoring_source_sha256']==OLD_SCORING,
            'q0 actual source model/input/scoring closure differs')
    source_payload=original_signature['payload']
    require(source_payload['reference_policy_sha256']==files.sha(lineage['source_cache'])
            and source_payload['reference_spec_sha256']==object_sha(original_config)
            and source_payload['selected_records_sha256']==object_sha(source_records), 'Old reference signature differs')
    require(len(target)==len(records)==done['rows']==lineage['target_rows']
            and done['status']=='complete_fixed_initial_reference' and done['cache_sha256']==files.sha(cache)
            and done['reference_spec_sha256']==object_sha(config), 'Derived q0 cache incomplete')
    allowed_row={'index','reference_spec_sha256'}
    allowed_spec={'manifest_sha256','selected_records_sha256','pixel_sha256'}
    wanted={**original_config,'manifest_sha256':files.sha(cli['manifest']),
            'selected_records_sha256':object_sha(records),
            'pixel_sha256':{name:original_config['pixel_sha256'][name]
                            for name in sorted({n for r in records for n in r['image_paths_display']})}}
    require(config==wanted and lineage['allowed_row_changes']==sorted(allowed_row)
            and lineage['allowed_spec_changes']==sorted(allowed_spec)
            and lineage['source_rows']==1024 and len(lineage['rows'])==len(records), 'Cache lineage permits unreviewed changes')
    require(lineage['source_cache_sha256']==files.sha(lineage['source_cache'])
            and lineage['source_config_sha256']==files.sha(str(lineage['source_cache'])+'.config.json')
            and lineage['source_completion_sha256']==files.sha(str(lineage['source_cache'])+'.completion.json')
            and lineage['source_manifest_sha256']==files.sha(lineage['source_manifest'])
            and lineage['target_cache_sha256']==files.sha(cache)
            and lineage['target_manifest_sha256']==files.sha(cli['manifest'])
            and lineage['source_spec_sha256']==object_sha(original_config)
            and lineage['target_spec_sha256']==object_sha(config), 'Cache lineage file receipts differ')
    index={(r['qid'],r['presentation_id']):i for i,r in enumerate(source_records)}
    require(len(index)==1024,'Duplicate original reference input')
    for i,(record,row,link) in enumerate(zip(records,target,lineage['rows'])):
        origin=index[(record['qid'],record['presentation_id'])];old=source[origin]
        require(record==source_records[origin] and row=={**old,'index':i,'reference_spec_sha256':object_sha(config)},
                'q0 row modified beyond identity-only index/spec wrapper')
        unchanged={k:value for k,value in old.items() if k not in allowed_row}
        require(link=={'index':i,'source_index':origin,'qid':record['qid'],'presentation_id':record['presentation_id'],
                       'record_sha256':object_sha(record),'source_row_sha256':object_sha(old),
                       'target_row_sha256':object_sha(row),'unchanged_payload_sha256':object_sha(unchanged),
                       'reference_literal_check_was_source_index_zero':origin==0}, 'q0 source-row lineage differs')
        p=reconstruct(row['token_logps'],row['candidates']);check_policy_fields(row,p)
        gold,rewards=coordinate_gold(record)
        require(row['exact_rewards']==rewards and row['gold_candidate_index']==gold,'q0 gold differs')
        numeric_check(row['numerical_checks'],row['token_logps'],origin==0)
        require(row['input_audit']['input_tokens']<=4096 and row['input_audit']['recipe_tokens_used']<=1536,
                'q0 input budget differs')
    verify_identity(files,policy['initial_identity'])
    for name,digest in config['pixel_sha256'].items():
        require(files.sha(Path(cli['image_root'])/name)==digest,'Actual q0 photograph bytes differ')
    return {'records':records,'rows':target,'config':config,'lineage':lineage,
            'sha256':files.sha(cache),'spec_sha256':object_sha(config),
            'historical_whole1024_reference_candidate_forwards':25600,'new_q0_candidate_forwards':0}


def audit_authorization(files,policy,policy_path,started):
    auth=files.json(policy['authorization_path'])
    key=policy['flow']+'_'+policy['arm']
    if policy['mode']!='formal':key=policy['mode']+'_'+key
    require(auth['schema_version']==1 and auth['child_key']==key and auth['phase']==policy['mode']
            and auth['child_policy_sha256']==files.sha(policy_path),'Authorization child/phase differs')
    parent=files.json(auth['stage_policy_path'])
    require(auth['stage_policy_sha256']==files.sha(auth['stage_policy_path'])
            and parent['children'][key]['sha256']==files.sha(policy_path)
            and files.relative(parent['children'][key]['path'])==files.relative(policy_path)
            and parent['sources_sha256']==policy['sources_sha256'],'Authorization parent/source differs')
    for field in ('first_preflight_started_utc','absolute_deadline_utc'):
        require(auth[field]==parent[field]==policy[field],'Authorization original clock differs')
    require(utc(policy['first_preflight_started_utc'])<=utc(auth['issued_utc'])<=started
            <utc(policy['absolute_deadline_utc']),'Authorization after model/run start or outside clock')
    def receipt(item):
        require(files.sha(item['path'])==item['sha256'],'Authorization receipt bytes differ')
        proof=files.json(item['path']);require(proof['passed'] is True,'Unpassed prerequisite receipt')
        return proof
    cpu=receipt(auth['cpu_receipt'])
    require(cpu['tests_run']==parent['expected_cpu_tests'] and cpu['skipped']==cpu['failures']==cpu['errors']==0
            and cpu['cuda_initialized'] is False and cpu['policy_sha256']==auth['stage_policy_sha256']
            and cpu['sources_sha256']==policy['sources_sha256']
            and cpu['first_preflight_started_utc']==policy['first_preflight_started_utc']
            and cpu['absolute_deadline_utc']==policy['absolute_deadline_utc']
            and cpu['native_watchdog_probe']['passed'] is True,'Actual zero-skip CPU admission missing')
    require(set(auth['q0_receipts'])=={'formal_flow44','formal_flow45','smoke_flow44','smoke_flow45'},
            'Missing4 actual q0 audits')
    for name,item in auth['q0_receipts'].items():
        proof=receipt(item)
        mode,flow=name.split('_',1);childkey=('smoke_' if mode=='smoke' else '')+flow+'_raw_sft'
        require(proof['status']=='complete_verified_reference_refresh_q0'
                and proof['new_q0_candidate_forwards']==0 and proof['mode']==mode and proof['flow']==flow
                and proof['arm']=='raw_sft' and proof['unique']==(4 if mode=='smoke' else 256)
                and proof['locked_policy_sha256']==parent['children'][childkey]['sha256'],'Wrong q0 audit role')
    if policy['mode']=='formal':
        required={'smoke_'+flow+'_'+arm for flow in ('flow44','flow45') for arm in ARMS}
        required|={'path_smoke_'+flow+'_refresh_beta' for flow in ('flow44','flow45')}
        require(set(auth['technical_receipts'])==required,'All10 technical smokes required')
        for name,item in auth['technical_receipts'].items():
            proof=receipt(item)
            kind='path_smoke' if name.startswith('path_smoke_') else 'smoke'
            flow,arm=name[len(kind)+1:].split('_',1)
            require(proof['status']=='complete_verified_reference_refresh_run'
                    and proof['mode']==kind and proof['flow']==flow and proof['arm']==arm
                    and proof['locked_policy_sha256']==parent['children'][name]['sha256'],
                    'Wrong technical receipt role')
        gate=receipt(auth['whole_gate_receipt'])
        require(gate['all_eight_formal_arms_required'] is True,'Reduced formal time/resource gate')
        audit_resource_gate(files,parent,gate)
    return auth


def fixed_stage_schedule():
    arms=[flow+'_'+arm for flow in ('flow44','flow45') for arm in ARMS]
    technical=['smoke_'+arm for arm in arms]+['path_smoke_'+flow+'_refresh_beta' for flow in ('flow44','flow45')]
    schedule=['cpu_preflight']+['q0_'+mode+'_'+flow+'_verify' for flow in ('flow44','flow45') for mode in ('smoke','formal')]
    schedule.extend(tag for key in technical for tag in (key+'_train',key+'_verify'))
    schedule.extend(tag for arm in arms for tag in (
        arm+'_train',arm+'_verify',arm+'_step128_dev200',arm+'_step128_dev200_coverage',
        arm+'_step128_orbit384',arm+'_step128_orbit384_coverage'))
    schedule.append('stage_global_verify')
    require(len(schedule)==len(set(schedule))==74,'Internal exact stage schedule error')
    return arms,technical,schedule


def resource_forecast(technical,history,prediction_per_row,prediction_load):
    arms,required,_=fixed_stage_schedule()
    require(set(technical)==set(required),'Need all10 actual technical timings')
    score=history['canonical_score_per_row_floor_seconds']
    verify=history['verifier_wall_floor_seconds'];load=history['training_load_and_checkpoint_floor_seconds']
    floors=history['full_run_step_floor_seconds']
    require(set(floors)=={'raw_sft','exact_kl'} and min(floors.values())>0
            and min(score,prediction_per_row,prediction_load)>0,'Missing full historical resource floors')
    for key,observed in technical.items():
        path=key.startswith('path_smoke_');expected_steps=2 if path else 1
        require(observed['verified'] is True and observed['steps']==expected_steps
                and observed['visits']==4*expected_steps and len(observed['step_seconds'])==expected_steps
                and min(observed['step_seconds'])>0 and observed['final_bank_rows']==4
                and observed['final_bank_seconds']>0 and observed['wall_seconds']>0,'Technical resource observation incomplete')
        require((observed['refresh_rows']==4 and observed['refresh_seconds']>0) if path else
                (observed['refresh_rows']==0 and observed['refresh_seconds']==0),'Wrong refresh-path cost')
        score=max(score,observed['final_bank_seconds']/4,
                  observed['refresh_seconds']/4 if path else 0)
        verify=max(verify,observed['verify_seconds'])
        load=max(load,observed['wall_seconds']-math.fsum(observed['step_seconds'])
                 -observed['final_bank_seconds']-observed['refresh_seconds'],0.)
    per_arm={}
    for arm in arms:
        method='raw_sft' if arm.endswith('_raw_sft') else 'exact_kl'
        measured=technical['smoke_'+arm]['step_seconds'][:]
        if arm.endswith('_refresh_beta'):measured+=technical['path_smoke_'+arm]['step_seconds']
        per_arm[arm]=max(floors[method],*measured)
    components={'eight_128update_trainings':128*math.fsum(per_arm.values()),
                'eight_training_loads_and_checkpoint_overhead':8*load,
                'two_actual256_reference_refreshes':512*score,
                'eight_fixed256_final_train_banks':2048*score,
                '4672_predictions_and_16_model_loads':4672*prediction_per_row+16*prediction_load,
                'eight_independent_run_verifications':8*verify}
    return {'estimated_seconds':1.5*math.fsum(components.values())+600,
            'components_seconds':components,'formal_step_seconds_by_arm':per_arm,
            'canonical_score_per_row_seconds':score,'prediction_per_record_seconds':prediction_per_row}


def disk_requirement(checkpoint,adapter,continuity,technical_complete):
    require(checkpoint>adapter>0 and continuity>=0,'Missing actual state/checkpoint byte bounds')
    count=16 if technical_complete else 28;snapshots=4 if technical_complete else 8
    boundary=max(checkpoint+adapter,continuity)
    return math.ceil(1.2*(count*checkpoint+snapshots*boundary+512*1024**2)+1024**3)


def directory_bytes(files,directory):
    paths=[path for path in files.path(directory).rglob('*') if path.is_file()]
    require(paths,'Empty physical checkpoint directory')
    return sum(files.physical(path)['bytes'] for path in paths)


def audit_resource_gate(files,parent,gate):
    """Rebuild the complete conservative gate from actual old/technical receipts."""
    arms,technical_keys,schedule=fixed_stage_schedule();root=Path(parent['output_directory'])
    require(parent['command_schedule']==schedule and parent['formal_order']==arms
            and parent['technical_order']==technical_keys,'Parent reduced/reordered full schedule')
    state=files.json(root/'state.json');commands={c['stage']:c for c in state['commands']}
    files.mutable_inputs.add(files.relative(root/'state.json'))
    technical={};largest_checkpoint=0;largest_continuity=0
    require(set(gate['technical_receipts'])==set(technical_keys),'Resource gate lacks10 receipt bindings')
    for key in technical_keys:
        child=files.json(parent['children'][key]['path']);out=Path(child['cli']['output_dir'])
        item=gate['technical_receipts'][key]
        require(files.relative(item['path'])==files.relative(out/'independent_verification.json')
                and files.sha(item['path'])==item['sha256'],'Technical receipt path/SHA differs')
        proof=files.json(item['path']);outcome=files.json(out/'train_outcome.json')
        require(proof['passed'] is True and proof['status']=='complete_verified_reference_refresh_run'
                and proof['locked_policy_sha256']==parent['children'][key]['sha256']
                and proof['verifier_sha256']==parent['sources_sha256']['research/reference_refresh_20261107/verify_refresh_run.py'],
                'Resource gate uses stale/unverified technical run')
        for name in ('step_times.jsonl','train_outcome.json'):
            bound=proof['required_artifacts'][files.relative(out/name)]
            require(files.physical(out/name)==bound,'Technical raw timing not bound to independent run proof')
        step_rows=files.rows(out/'step_times.jsonl');bank=files.json(out/'final_train_bank/scores.jsonl.completion.json')
        before=child['cli']['refresh_boundary_step']
        event=files.json(out/('refresh_at_step'+str(before))/'refresh_event.json') if before is not None else None
        training_command=commands[key+'_train'];verification_command=commands[key+'_verify']
        require(training_command['returncode']==verification_command['returncode']==0
                and utc(verification_command['finished_utc'])<=utc(gate['completed_utc']),
                'Technical command not completed before formal resource admission')
        technical[key]={'verified':True,'steps':proof['steps'],'visits':proof['visits'],
                        'step_seconds':[r['elapsed_seconds'] for r in step_rows],
                        'wall_seconds':training_command['wall_seconds'],'verify_seconds':verification_command['wall_seconds'],
                        'final_bank_rows':bank['rows'],'final_bank_seconds':bank['elapsed_seconds'],
                        'refresh_rows':event['rows'] if event else 0,'refresh_seconds':event['elapsed_seconds'] if event else 0}
        for checkpoint in outcome['checkpoints']:
            largest_checkpoint=max(largest_checkpoint,directory_bytes(files,checkpoint['path']))
        if event:
            largest_continuity=max(largest_continuity,*[files.physical(x['path'])['bytes']
                                                       for x in event['continuity_files'].values()])
    history_state=files.json(parent['historical_coverage_state'])
    require(history_state['status']=='completed_four_coverage_endpoints_development_only','Historical coverage incomplete')
    old_commands={c['stage']:c for c in history_state['commands']};floors={'raw_sft':0.,'exact_kl':0.}
    old_load=old_verify=0.
    expected_old={f+'_'+m for f in ('flow44','flow45') for m in ('raw_sft','exact_kl')}
    require(set(parent['historical_training_runs'])==expected_old,'Historical timing drops a flow/method')
    for key,path in parent['historical_training_runs'].items():
        outcome=files.json(Path(path)/'train_outcome.json');step_rows=files.rows(Path(path)/'step_times.jsonl')
        train=old_commands[key+'_train'];verify=old_commands[key+'_verify']
        require(outcome['status']=='completed' and outcome['global_step']==256 and outcome['actual_prompt_visits']==1024
                and len(step_rows)==256 and [s['global_step'] for s in step_rows]==list(range(1,257))
                and train['returncode']==verify['returncode']==0,'Historical training floor lacks complete256updates')
        floors[outcome['method']]=max(floors[outcome['method']],train['wall_seconds']/256)
        old_load=max(old_load,train['wall_seconds']-math.fsum(s['elapsed_seconds'] for s in step_rows),0.)
        old_verify=max(old_verify,verify['wall_seconds'])
    reference=files.json(parent['historical_reference_outcome'])
    require(reference['status']=='completed_reference_only' and reference['rows']==1024 and reference['updates']==0
            and reference['reference_candidate_forwards_charged_this_run']==25600,'Historical true q0 cost missing')
    history={'full_run_step_floor_seconds':floors,'training_load_and_checkpoint_floor_seconds':old_load,
             'verifier_wall_floor_seconds':old_verify,'canonical_score_per_row_floor_seconds':reference['elapsed_seconds']/1024,
             'historical_q0_forward_cost':25600,'new_q0_forwards':0,'historical_reference_rows':1024}
    orbit=files.json(parent['historical_orbit_state'])
    require(orbit['status']=='complete_four_endpoint_orbits_development_only','Historical orbit incomplete')
    old_predictions=[c for c in orbit['commands'] if c['stage'].startswith('formal_') and not c['stage'].endswith('_coverage')]
    require(len(old_predictions)==4 and {c['stage'] for c in old_predictions}=={'formal_'+x for x in expected_old},
            'Historical prediction timing missing controls')
    gaps=[];first=[]
    for c in old_predictions:
        emitted=c['prediction_emissions'];require(c['returncode']==0 and len(emitted)==384,'Incomplete384 prediction timing')
        times=[r['seconds_since_start'] for r in emitted]
        require(times[0]>0 and all(b>a for a,b in zip(times,times[1:])),'Nonmonotonic historical emission timing')
        first.append(times[0]);gaps.extend(b-a for a,b in zip(times,times[1:]))
    forecast=resource_forecast(technical,history,max(gaps),max(first))
    for key,value in forecast.items():
        if isinstance(value,dict):
            require(set(gate[key])==set(value),'Resource components/arm set differs')
            for sub,number in value.items():close(gate[key][sub],number,'resource '+key+'/'+sub)
        else:close(gate[key],value,'resource '+key)
    for key,value in history.items():
        if isinstance(value,dict):
            for sub,number in value.items():close(gate['historical_floors'][key][sub],number,'actual history floor')
        else:close(gate['historical_floors'][key],value,'actual history cost')
    prior=Path('exploratory/principled_ordering_20261105/finite_trials_v2/flow44/raw_sft/checkpoint-64')
    checkpoint=max(directory_bytes(files,prior),largest_checkpoint)
    adapter=sum(files.physical(path)['bytes'] for path in files.path(prior).glob('adapter_model*.safetensors'))
    storage=gate['storage'];required=disk_requirement(checkpoint,adapter,largest_continuity,True)
    require(storage['actual_checkpoint_bytes']==checkpoint and storage['actual_adapter_bytes']==adapter
            and storage['continuity_file_bound_bytes']==max(checkpoint+adapter,largest_continuity)
            and storage['remaining_checkpoint_count']==16 and storage['remaining_continuity_count']==4
            and storage['formal_checkpoints']==16 and storage['technical_checkpoints']==12
            and storage['formal_continuity_files']==storage['technical_continuity_files']==4
            and storage['technical_artifacts_already_present'] is True
            and storage['required_free_bytes']==required<=storage['actual_free_bytes'] and storage['passed'] is True,
            'Complete physical storage gate differs')
    remaining=(utc(parent['absolute_deadline_utc'])-utc(gate['completed_utc'])).total_seconds()
    require(0<=gate['remaining_seconds']-remaining<2 and forecast['estimated_seconds']<remaining
            and gate['passed'] is True and gate['all_eight_formal_arms_required'] is True
            and gate['formal_visits']==4096 and gate['formal_updates']==1024
            and gate['formal_reference_rows']==512 and gate['formal_final_train_bank_rows']==2048
            and gate['formal_predictions']==4672 and gate['new_q0_forwards']==0
            and gate['historical_shared_q0_forwards_charged_once']==25600
            and gate['safety_factor']==1.5 and gate['fixed_cpu_and_validation_margin_seconds']==600,
            'Complete conservative time gate or original clock differs')
    return {'passed':True,'estimated_seconds':forecast['estimated_seconds'],'remaining_at_gate_seconds':remaining,
            'required_free_bytes':required,'technical_branches':10,'formal_arms':8,
            'duration_upper_bound_claimed':False}


def audit_scored_bank(files,policy,run_dir,signed,q0,step,role,checkpoint_fingerprints):
    directory=Path(run_dir)/('refresh_at_step'+str(step) if role=='fresh_reference' else 'final_train_bank')
    cache=directory/('reference_policy.jsonl' if role=='fresh_reference' else 'scores.jsonl')
    config=files.json(str(cache)+'.config.json');done=files.json(str(cache)+'.completion.json')
    rows=files.rows(cache);records=q0['records'];payload=signed['payload'];checkpoint=Path(run_dir)/('checkpoint-'+str(step))
    checkpoint_audit=files.json(checkpoint/'finite_policy_checkpoint_audit.json')
    scored_identity={'base':policy['initial_identity']['base'],'adapter':checkpoint_audit['adapter_identity']}
    require(config['schema_version']==2 and config['role']==role
            and config['policy']=='canonical24_full_sequence_plusEOS_temperature1'
            and config['scored_identity']==scored_identity and config['initial_identity']==policy['initial_identity']
            and files.relative(config['checkpoint'])==files.relative(checkpoint)
            and config['checkpoint_audit_sha256']==files.sha(checkpoint/'finite_policy_checkpoint_audit.json')
            and config['global_step']==step and config['arm']==policy['arm'] and config['flow']==policy['flow']
            and config['input']==INPUT and config['revision']==REVISION
            and config['manifest_sha256']==policy['cli']['expected_manifest_sha256']
            and config['selected_records_sha256']==object_sha(records)
            and tuple(config['candidate_texts'])==ACTIONS and config['pixel_sha256']==payload['pixel_sha256']
            and config['q0_reference_policy_sha256']==q0['sha256']
            and config['q0_reference_lineage_sha256']==payload['reference_lineage_sha256']
            and config['scoring_source_sha256']==OLD_SCORING and config['versions']==payload['versions']
            and config['run_signature_sha256']==signed['sha256'] and config['numerical_reference_atol']==.01,
            'Fresh/final scored-bank configuration differs')
    expected_status='complete_current_checkpoint_reference' if role=='fresh_reference' else 'complete_fixed_final_train_bank'
    require(len(rows)==len(records)==done['rows'] and done['status']==expected_status and done['passed'] is True
            and done['role']==role and done['cache_sha256']==files.sha(cache)
            and done['reference_spec_sha256']==object_sha(config) and done['run_signature_sha256']==signed['sha256']
            and done['checkpoint_audit_sha256']==config['checkpoint_audit_sha256']
            and done['scored_identity']==scored_identity and done['global_step']==step
            and done['candidate_forwards']==25*len(records)
            and all(done[key]==0 for key in ('optimizer_steps','backward_calls','generation_calls')),
            'Incomplete/updated fresh-reference or endpoint bank')
    require(done['before_state']==done['after_state'] and done['before_state']['global_step']==step
            and done['before_state']['optimizer_steps']==[step]*392
            and done['before_state']['scheduler_last_epoch']==step,'Scoring changed actual runtime state summary')
    for key in ('optimizer','scheduler','rng'):
        require(done['before_state']['component_sha256'][key]==checkpoint_fingerprints[step][key],
                'Scoring state is not actual saved checkpoint state: '+key)
    require(done['adapter_fingerprint_before']==done['adapter_fingerprint_after']==checkpoint_fingerprints[step]['adapter']
            and done['visual_fingerprint_before']==done['visual_fingerprint_after']==payload['initial_visual_fingerprint'],
            'Scoring model/visual fingerprint differs from physical checkpoint')
    require(utc(policy['first_preflight_started_utc'])<=utc(done['started_utc'])<=utc(done['finished_utc'])
            <=utc(policy['absolute_deadline_utc']) and finite(done['wall_seconds']) and done['wall_seconds']>0,
            'Scoring outside original clock or missing wall cost')
    close(done['elapsed_seconds'],done['wall_seconds'],'scoring wall cost',1e-10)
    for i,(record,row,initial) in enumerate(zip(records,rows,q0['rows'])):
        require(all(row[key]==record[key] for key in ('qid','recipe_id','presentation_id','source_split',
                                                     'display_perm','target_order','target_labels'))
                and row['index']==i and row['role']==role and row['global_step']==step
                and row['run_signature_sha256']==signed['sha256']
                and row['reference_spec_sha256']==object_sha(config)
                and row['scored_identity_sha256']==object_sha(scored_identity)
                and row['q0_source_row_sha256']==object_sha(initial) and row['q0_reference_row_index']==i,
                'Scoring row input/model/source identity differs')
        require(row['input_audit']==initial['input_audit'] and row['candidates']==initial['candidates'],
                'Scoring changed actual q0 input or grammar')
        current=reconstruct(row['token_logps'],row['candidates']);check_policy_fields(row,current)
        reference=reconstruct(initial['token_logps'],initial['candidates']);gold,rewards=coordinate_gold(record)
        require(row['exact_rewards']==rewards and row['gold_candidate_index']==gold,'Scoring row gold differs')
        numeric_check(row['numerical_checks'],row['token_logps'],i==0)
        # Technical arithmetic checks only; never emit or select on method
        # performance before the full eight-arm scientific completion guard.
        endpoint_objective(current,reference,gold,BETA)
        endpoint_objective(current,reference,gold,BETA/2)
    return {'path':str(cache),'rows':rows,'config':config,'completion':done}


def audit_refresh_event(files,policy,run_dir,signed,q0,checkpoints):
    boundary=policy['cli']['refresh_boundary_step'];path=Path(run_dir)/('refresh_at_step'+str(boundary))/'refresh_event.json'
    event=files.json(path)
    bank=audit_scored_bank(files,policy,run_dir,signed,q0,boundary,'fresh_reference',checkpoints)
    done=bank['completion'];cache=bank['path']
    require(event['schema_version']==1 and event['passed'] is True and event['reference_generation']==1
            and event['boundary_step']==boundary and event['rows']==len(q0['records'])
            and event['run_signature_sha256']==signed['sha256']
            and event['child_policy_sha256']==signed['payload']['locked_policy_sha256']
            and files.relative(event['checkpoint'])==files.relative(Path(run_dir)/('checkpoint-'+str(boundary)))
            and event['checkpoint_audit_sha256']==done['checkpoint_audit_sha256']
            and event['scored_identity']==done['scored_identity'] and event['q0_reference_policy_sha256']==q0['sha256']
            and files.relative(event['reference_policy_path'])==files.relative(cache)
            and event['reference_policy_sha256']==files.sha(cache)
            and event['reference_spec_sha256']==object_sha(bank['config'])
            and event['config_sha256']==files.sha(str(cache)+'.config.json')
            and event['completion_sha256']==files.sha(str(cache)+'.completion.json')
            and event['candidate_forwards']==done['candidate_forwards']
            and all(event[key]==0 for key in ('optimizer_steps','backward_calls','generation_calls')),
            'Refresh event lacks actual current-model checkpoint/reference lineage')
    for field in ('started_utc','finished_utc','wall_seconds','elapsed_seconds','before_state','after_state'):
        require(event[field]==done[field],'Refresh event/completion differs: '+field)
    require(set(event['continuity_files'])=={'before','after'},'Missing actual before/after snapshots')
    summaries={}
    for side in ('before','after'):
        item=event['continuity_files'][side]
        require(files.relative(item['path'])==files.relative(path.parent/('continuity_'+side+'.pt')),
                'Snapshot escaped actual refresh boundary')
        require(files.physical(item['path'])=={'sha256':item['sha256'],'bytes':item['bytes']},'Continuity snapshot bytes differ')
        payload=load_cpu_state(files,item['path'])
        summary=continuity_summary(payload)
        require(summary==event[side+'_state'],'Actual snapshot contents differ from recorded logical components')
        require(payload['global_step']==boundary,'Wrong actual refresh snapshot boundary')
        summaries[side]=summary;del payload
    verify_unchanged_components(summaries['before']['component_sha256'],summaries['after']['component_sha256'],
                                summaries['before']['component_sha256'])
    require(summaries['before']==summaries['after'],'Logical snapshot state changed during refresh')
    return event,bank['rows']


def audit_training(files,policy,policy_path,run_dir,q0):
    cli=policy['cli'];unique=cli['unique_count'];limit=cli['limit'];steps_count=cli['max_steps']
    require(files.relative(run_dir)==files.relative(cli['output_dir']),'Run directory differs from frozen child')
    signed=files.json(Path(run_dir)/'run_signature.json');payload=signed['payload'];signature=signed['sha256']
    require(signature==object_sha(payload),'Run signature hash differs')
    expected_cli={**cli,'policy':str(REMOTE/files.relative(policy_path)),'arm':policy['arm'],'mode':policy['mode']}
    require(payload['cli']==expected_cli and payload['locked_policy_sha256']==files.sha(policy_path)
            and payload['method']==cli['method'] and payload['arm']==policy['arm']
            and payload['mode']==payload['refresh_mode']==policy['mode']
            and payload['unique_count']==unique and payload['passes']==cli['passes']
            and payload['actual_updates']==steps_count and payload['refresh_boundary_step']==cli['refresh_boundary_step']
            and payload['accumulation']==4 and payload['canonical_completion_token_count']==5
            and payload['resolved_beta']==cli['beta'],'Signed CLI/objective/exposure differs')
    visitspec=[{'visit_index':i,'pass_index':i//unique,'index_within_pass':i%unique,
                'record_sha256':object_sha(q0['records'][i%unique])} for i in range(limit)]
    require(payload['visit_sequence_sha256']==object_sha(visitspec)
            and payload['selected_records_sha256']==object_sha(q0['records'])
            and payload['manifest_sha256']==cli['expected_manifest_sha256']
            and payload['initial_identity']==policy['initial_identity'] and payload['input']==INPUT
            and payload['reference_policy_sha256']==q0['sha256'] and payload['reference_spec_sha256']==q0['spec_sha256']
            and payload['reference_lineage_sha256']==files.sha(policy['q0_reference_lineage'])
            and payload['generated_reference_receipt_sha256']==files.sha(policy['generated_reference_receipt'])
            and payload['sources_sha256']==policy['sources_sha256']
            and payload['versions']==q0['config']['versions'],'Run q0/input/source/version binding differs')
    require(payload['optimizer']=={'name':'AdamW','lr':5e-6,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,
                                   'max_grad_norm':1.,'schedule':'constant','fresh_state':True},'Optimizer configuration differs')
    require(payload['batch_clock_started_utc']==policy['batch_clock_started_utc']
            and payload['absolute_deadline_utc']==policy['absolute_deadline_utc'],'Run resets batch clock')
    state=files.json(Path(run_dir)/'state.json');outcome=files.json(Path(run_dir)/'train_outcome.json')
    require(state['status']==outcome['status']=='completed' and state['outcome']==outcome
            and state['global_step']==outcome['global_step']==steps_count
            and state['actual_prompt_visits']==outcome['actual_prompt_visits']==limit
            and state['run_signature_sha256']==outcome['run_signature_sha256']==signature
            and outcome['arm']==policy['arm'] and outcome['mode']==policy['mode']
            and outcome['unique_count']==unique and outcome['passes']==cli['passes'], 'Training completion/count differs')
    require(outcome['actual_visit_sequence_verified'] is True and outcome['resume_supported'] is False
            and outcome['test_access'] is False and state['test_access'] is False and state['no_resume'] is True,
            'Resume/heldout/truncation scope differs')
    started,finished=utc(state['started_utc']),utc(state['finished_utc'])
    require(utc(policy['first_preflight_started_utc'])<=started<=finished<=utc(state['updated_utc'])
            <=utc(policy['absolute_deadline_utc']),'Training escaped original clock')
    audit_authorization(files,policy,policy_path,started)
    require(payload['authorization_sha256']==files.sha(policy['authorization_path']),'Run authorization SHA differs')
    inventory_record=files.json(Path(run_dir)/'trainable_parameters.json');inventory=inventory_record['parameters']
    require(len(inventory)==len({r['name'] for r in inventory})==392
            and inventory_record['trainable_parameters']==sum(r['numel'] for r in inventory)==17432576
            and all('lora_' in r['name'] and '.language_model.layers.' in r['name']
                    and math.prod(r['shape'])==r['numel'] for r in inventory),'Unapproved/missing trainable LoRA tensors')
    initial=safetensor_fingerprint(files,cli['adapter_path'],inventory)
    require(initial==payload['initial_adapter_fingerprint']==outcome['initial_adapter_fingerprint'],
            'Initial actual safe tensors differ from signed runtime fingerprint')
    require(outcome['initial_visual_fingerprint']==outcome['final_visual_fingerprint']==payload['initial_visual_fingerprint']
            and len(payload['initial_visual_fingerprint'])==64 and outcome['visual_weights_unchanged'] is True,
            'Frozen visual fingerprint changed')
    expected_steps=list(range(cli['save_steps'],steps_count+1,cli['save_steps']))
    require([item['step'] for item in outcome['checkpoints']]==expected_steps
            and {path.name for path in files.path(run_dir).glob('checkpoint-*') if path.is_dir()}==
                {'checkpoint-'+str(i) for i in expected_steps},'Checkpoint schedule differs')
    checkpoint_fingerprints={};final_fingerprint=None
    for item in outcome['checkpoints']:
        step=item['step'];checkpoint=Path(run_dir)/('checkpoint-'+str(step))
        require(files.relative(item['path'])==files.relative(checkpoint),'Checkpoint path differs')
        audit=files.json(checkpoint/'finite_policy_checkpoint_audit.json');saved=files.json(checkpoint/'trainer_state.json')
        require(item['adapter_identity']==audit['adapter_identity'] and audit['global_step']==saved['global_step']==step
                and audit['run_signature_sha256']==saved['run_signature_sha256']==signature
                and saved['actual_prompt_visits']==step*4 and saved['resume_supported'] is False
                and audit['reference_policy_sha256']==q0['sha256'],'Checkpoint model/trainer/q0 identity differs')
        generation=int(cli['refresh_boundary_step'] is not None and step>cli['refresh_boundary_step'])
        expected_reference=(outcome['refresh_event']['reference_policy_sha256'] if generation else q0['sha256'])
        require(audit['reference_generation']==generation and audit['active_reference_policy_sha256']==expected_reference,
                'Saved checkpoint uses wrong reference generation')
        close(saved['epoch'],step/steps_count,'checkpoint epoch')
        require(set(audit['state_sha256'])=={'optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json'},
                'Checkpoint state archive coverage differs')
        states={}
        for name,digest in audit['state_sha256'].items():
            require(files.sha(checkpoint/name)==digest,'Saved checkpoint state hash differs')
            if name!='trainer_state.json':states[name]=load_cpu_state(files,checkpoint/name)
        optimizer_state=states['optimizer.pt'];optimizer_steps=[]
        for key in sorted(optimizer_state['state']):
            value=optimizer_state['state'][key]['step'];number=float(value.item()) if hasattr(value,'item') else float(value)
            require(number==int(number),'Noninteger actual saved Adam step');optimizer_steps.append(int(number))
        require(optimizer_steps==[step]*392 and states['scheduler.pt']['last_epoch']==step,
                'Saved Adam/scheduler state reset or incomplete')
        final_fingerprint=safetensor_fingerprint(files,checkpoint,inventory)
        checkpoint_fingerprints[step]={key:logical_fingerprint(states[name],cpu_tensor_encoder)
                                      for key,name in [('optimizer','optimizer.pt'),('scheduler','scheduler.pt'),('rng','rng_state.pth')]}
        checkpoint_fingerprints[step]['adapter']=final_fingerprint
        verify_identity(files,{'base':policy['initial_identity']['base'],'adapter':audit['adapter_identity']})
        del states,optimizer_state
    boundary=cli['refresh_boundary_step'];fresh=None;event=None
    if boundary is not None:
        event,fresh=audit_refresh_event(files,policy,run_dir,signed,q0,checkpoint_fingerprints)
        require(outcome['refresh_event']==event,'Final outcome refresh event differs')
    else:
        require(outcome['refresh_event'] is None and not list(files.path(run_dir).glob('refresh_at_step*')),
                'Unplanned refresh event')
    visits=files.rows(Path(run_dir)/'actual_prompt_visits.jsonl');steps=files.rows(Path(run_dir)/'step_times.jsonl')
    require(len(visits)==limit and len(steps)==steps_count,'Actual visit/update logs incomplete')
    require(outcome['actual_visits_sha256']==files.sha(Path(run_dir)/'actual_prompt_visits.jsonl')
            and outcome['step_times_sha256']==files.sha(Path(run_dir)/'step_times.jsonl'),'Trajectory completion SHA differs')
    current_count=back_count=0;active_count=0
    for i,visit in enumerate(visits):
        generation=int(boundary is not None and i//4>=boundary)
        ref=fresh[i%unique] if generation else q0['rows'][i%unique]
        ref_hash=event['reference_policy_sha256'] if generation else q0['sha256']
        require(visit['visit_id']==q0['records'][i%unique].get('visit_id') and visit['arm']==policy['arm']
                and visit['reference_policy_sha256']==ref_hash,'Visit identity or native active reference differs')
        verified=verify_visit(visit,q0['records'][i%unique],ref,i,unique,cli['method'],cli['beta'],
                              generation,ref_hash,q0['sha256'],signature,cli['seed'])
        current_count+=verified['current_forwards'];back_count+=verified['backwards'];active_count+=verified['active_group']
        if (i+1)%4==0:
            step=steps[i//4];number=i//4+1
            require(step['global_step_before_update']==number-1 and step['global_step']==number
                    and step['arm']==policy['arm'] and step['reference_generation']==generation
                    and step['active_reference_policy_sha256']==ref_hash
                    and step['actual_prompt_visits']==i+1 and step['groups']==4 and step['active_groups']==active_count
                    and step['optimizer_step_executed'] is True and step['optimizer_state_entries']==392
                    and step['optimizer_step_values']==[number]*392 and step['scheduler_last_epoch']==number
                    and step['run_signature_sha256']==signature,'Actual Adam/scheduler step/exposure is discontinuous')
            for position in ('before_clip','after_clip'):
                gradient=step[position]
                require(gradient['trainable_gradient_tensors']==392 and 0<=gradient['nonzero_gradient_tensors']<=392
                        and finite(gradient['l2_norm']) and gradient['l2_norm']>=0,'Invalid actual accumulated gradient')
            require(step['after_clip']['l2_norm']<=1.0001
                    and step['after_clip']['l2_norm']<=step['before_clip']['l2_norm']+1e-6
                    and step['zero_reward_gradient_batch']==(step['before_clip']['l2_norm']==0)
                    and finite(step['actual_parameter_update_l2']) and step['actual_parameter_update_l2']>=0
                    and finite(step['elapsed_seconds']) and step['elapsed_seconds']>0,'Gradient clip/update/time differs')
            close(step['learning_rate'],5e-6,'fixed Adam learning rate',1e-12)
            require(step['cumulative_current_policy_candidate_forwards_including_numerical_checks']==current_count
                    and step['cumulative_gradient_candidate_forwards']==step['cumulative_candidate_backwards']==back_count
                    and step['initial_reference_candidate_forwards_charged_this_run']==0
                    and step['cumulative_refreshed_reference_candidate_forwards']==(25*unique if generation else 0),
                    'Per-step forward/backward/reference cost differs')
            active_count=0
    bank=audit_scored_bank(files,policy,run_dir,signed,q0,steps_count,'final_train_bank',checkpoint_fingerprints)
    final=outcome['final_train_bank'];cache=bank['path'];done=bank['completion']
    require(final['passed'] is True and files.relative(final['path'])==files.relative(cache)
            and final['rows']==unique and final['raw_sha256']==files.sha(cache)
            and final['config_sha256']==files.sha(str(cache)+'.config.json')
            and final['completion_sha256']==files.sha(str(cache)+'.completion.json')
            and final['candidate_forwards']==25*unique and final['global_step']==steps_count
            and final['run_signature_sha256']==signature and final['checkpoint_identity']==outcome['final_adapter_identity']
            and all(final[key]==0 for key in ('optimizer_steps','backward_calls','generation_calls')),
            'Final fixed train bank not bound to committed final checkpoint')
    require(outcome['final_adapter_fingerprint']==final_fingerprint
            and outcome['final_adapter_identity']==outcome['checkpoints'][-1]['adapter_identity'],'Final saved adapter differs')
    effective=verify_effective_update(policy['mode'],initial,final_fingerprint,outcome['effective_adapter_update'],
                                      outcome['scientific_outcome'],[step['actual_parameter_update_l2'] for step in steps])
    require(outcome['unique_questions']==len({r['qid'] for r in q0['records']})
            and outcome['unique_recipes']==len({r['recipe_id'] for r in q0['records']}),'Question/recipe unit count differs')
    refresh_count=25*unique if boundary is not None else 0;bank_count=25*unique
    for key,value in {'reference_candidate_forwards_charged_this_run':0,
                      'current_policy_candidate_forwards_including_numerical_checks':current_count,
                      'gradient_candidate_forwards':back_count,'candidate_backwards':back_count,
                      'refreshed_reference_candidate_forwards':refresh_count,'final_bank_candidate_forwards':bank_count,
                      'candidate_forwards':current_count+back_count+refresh_count+bank_count}.items():
        require(outcome[key]==value,'Final compute sum differs: '+key)
    require(sum(step['elapsed_seconds'] for step in steps)<=outcome['elapsed_seconds_including_reference']
            <=(finished-started).total_seconds()+1,'Update wall costs contradict total runtime')
    return {'passed':True,'status':'complete_verified_reference_refresh_run','arm':policy['arm'],
            'mode':policy['mode'],'flow':policy['flow'],'steps':steps_count,'visits':limit,'unique':unique,
            'effective_adapter_update':effective,'run_signature_sha256':signature,
            'forward_backward_counts':{'q0_new_forwards':0,'current_policy_candidate_forwards':current_count,
                'gradient_candidate_forwards':back_count,'candidate_backwards':back_count,
                'refresh_reference_candidate_forwards':refresh_count,'final_train_bank_candidate_forwards':bank_count,
                'total_candidate_forwards':current_count+back_count+refresh_count+bank_count},
            'locked_policy_sha256':files.sha(policy_path),'initial_adapter_fingerprint':initial,
            'final_adapter_fingerprint':final_fingerprint,'reference_policy_sha256':q0['sha256'],
            'actual_snapshot_logical_contents_recomputed':boundary is not None,
            'scientific_performance_summary_emitted':False,'model_GPU_network_calls':0,
            'limits':['Persisted safe LoRA bytes and complete optimizer/scheduler/RNG/gradient/mode snapshots are independently checked on CPU; no VLM or Adam arithmetic replay.',
                      'Frozen visual runtime fingerprints are bound to unchanged physical base identity; the verifier does not rerun vision.',
                      'No method performance conclusion is emitted before the complete eight-arm stage guard.']}


def independently_derive_orbit(dev):
    require(len(dev)==200 and len({r['recipe_id'] for r in dev[:16]})==16,'Fixed first16 development recipes required')
    output=[]
    for original in dev[:16]:
        coordinate_gold(original)
        display=original['display_perm'];paths=dict(zip(display,original['image_paths_display']))
        statuses=dict(zip(display,original['image_status_display'])) if original.get('image_status_display') else None
        for layout in permutations(range(4)):
            row=deepcopy(original)
            row.update(presentation_id=original['qid']+'::p'+''.join(map(str,layout)),display_perm=list(layout),
                       target_labels=['ABCD'[layout.index(i)] for i in original['target_order']],
                       image_paths_display=[paths[i] for i in layout],image_paths_chronological=[paths[i] for i in range(4)],
                       orbit_origin_presentation_id=original['presentation_id'],orbit_origin_display_perm=display.copy())
            if statuses:row['image_status_display']=[statuses[i] for i in layout]
            output.append(row)
    return output


def independent_text_score(text,record):
    if not isinstance(text,str):return {'valid':0,'exact':0,'semantic_order':None}
    candidate=re.sub(r'^answer\s*:?\s*','',text.strip(),flags=re.IGNORECASE)
    letters=re.findall('[A-D]',candidate.upper())
    legal=bool(re.fullmatch(r'[A-D](?:[A-D,\s]*[A-D])?',candidate,flags=re.IGNORECASE))
    if not legal or len(letters)!=4 or sorted(letters)!=list('ABCD'):
        return {'valid':0,'exact':0,'semantic_order':None}
    semantic=[record['display_perm']['ABCD'.index(c)] for c in letters]
    return {'valid':1,'exact':int(semantic==record['target_order']),'semantic_order':semantic}


def validate_stage_command_window(state,schedule):
    require([c['stage'] for c in state['commands']]==schedule and len(state['commands'])==74,
            'Incomplete74-command stage: refuse scientific score/prediction reads')
    live=state['status']=='running_stage_global_verify'
    if live:
        last=state['commands'][-1]
        require(last['returncode'] is None and last['pid']==state['active_child_pid']==os.getpid()
                and last['pgid']==state['pgid']==os.getpgrp()
                and all(command['returncode']==0 for command in state['commands'][:-1]),
                'Only the actual self-live same-group final guard may inspect first73 completed commands')
    else:
        require(state['status']=='completed_eight_reference_refresh_endpoints_independently_verified'
                and state['active_child_pid'] is None and all(c['returncode']==0 for c in state['commands']),
                'Stage has not committed complete74 state')
    return live


def audit_stage_completion(files,policy_path):
    """The complete-only guard executes before opening scientific raw scores."""
    parent=files.json(policy_path);root=Path(parent['output_directory'])
    state=files.json(root/'state.json');arms,technical,schedule=fixed_stage_schedule()
    files.mutable_inputs.add(files.relative(root/'state.json'))
    expected_status='completed_eight_reference_refresh_endpoints_independently_verified'
    require(parent['status']=='locked_before_reference_refresh_stage' and parent['command_schedule']==schedule
            and parent['formal_order']==arms and parent['technical_order']==technical
            and set(parent['children'])==set(arms+technical),'Wrong complete parent protocol')
    live=validate_stage_command_window(state,schedule)
    require(state['policy_sha256']==files.sha(policy_path)
            and state['first_preflight_started_utc']==parent['first_preflight_started_utc']
            and state['absolute_deadline_utc']==parent['absolute_deadline_utc'],'Complete state resets original policy/clock')
    beginning,deadline=utc(parent['first_preflight_started_utc']),utc(parent['absolute_deadline_utc'])
    require(parent['wall_limit_seconds']==72000 and parent['budget_cny_cap']==32
            and 0<(deadline-beginning).total_seconds()<=72000
            and beginning<=utc(state['started_utc'])<=utc(state['updated_utc'])<deadline,
            'Complete stage outside original20h deadline')
    prior=utc(state['started_utc'])
    for command in state['commands'][:-1] if live else state['commands']:
        require(command['returncode']==0 and command['pgid']==state['pgid']
                and prior<=utc(command['started_utc'])<=utc(command['finished_utc'])<=utc(state['updated_utc']),
                'Failed/detached/out-of-order stage command')
        prior=utc(command['finished_utc'])
    if live:
        require(prior<=utc(state['commands'][-1]['started_utc'])<=utc(state['updated_utc']),
                'Final self-live CPU guard started before scientific command completion')
    require(all(state[key] is False for key in ('automatic_retry','resume','test_access','confirmation_access')),
            'Stage scope/retry differs')
    require(set(parent['sources_sha256'])==SOURCE_PATHS,'Whole-stage source closure differs')
    for group in ('sources_sha256','artifacts_sha256'):
        for name,digest in parent[group].items():require(files.sha(name)==digest,'Whole-stage frozen bytes differ')
    require(parent['sources_sha256']['research/reference_refresh_20261107/verify_refresh_run.py']==sha(Path(__file__).read_bytes()),
            'Actual whole-stage verifier source is not frozen')
    science=files.json(root/'science_completion.json')
    require(files.sha(root/'science_completion.json')==state['science_completion_sha256']
            and science['status']=='science_complete_pending_global_verification'
            and science['commands_completed']==73 and science['stage_policy_sha256']==files.sha(policy_path)
            and science['all_formal_arms']==arms and science['all_technical_arms']==technical
            and science['eval_predictions']==4672 and science['final_train_bank_rows']==2048
            and all(science[key] is False for key in ('test_access','confirmation_access','score_selection','scientific_metrics_read_by_coordinator')),
            'First73 scientific completion commitment differs')
    require(utc(state['commands'][-2]['finished_utc'])<=utc(science['completed_utc'])<=utc(state['commands'][-1]['started_utc']),
            'Global CPU guard preceded the persisted first73 completion receipt')
    if not live:
        done=files.json(root/'completion.json')
        require(files.sha(root/'completion.json')==state['completion_sha256'] and done['status']==expected_status
                and done['stage_policy_sha256']==files.sha(policy_path) and done['all_formal_arms']==arms
                and done['all_technical_arms']==technical and done['commands']==74
                and done['eval_predictions']==4672 and done['final_train_bank_rows']==2048
                and done['test_access'] is False and done['confirmation_access'] is False
                and done['score_selection'] is False and done['scientific_metrics_read_by_coordinator'] is False
                and files.sha(done['global_verification_path'])==done['global_verification_sha256'],
                'Complete74 independent stage commitment differs')
        prior_guard=files.json(done['global_verification_path'])
        require(prior_guard['passed'] is True and prior_guard['status']=='complete_reference_refresh_stage_analysis_gate_passed'
                and prior_guard['locked_policy_sha256']==files.sha(policy_path),'Final global proof is not the actual parent policy')
    gate=audit_resource_gate(files,parent,files.json(root/'whole_eight_arm_resource_gate.json'))
    total={key:0 for key in ('q0_new_forwards','current_policy_candidate_forwards','gradient_candidate_forwards',
                            'candidate_backwards','refresh_reference_candidate_forwards',
                            'final_train_bank_candidate_forwards','total_candidate_forwards')}
    formal={}
    for key in technical+arms:
        descriptor=parent['children'][key];child=files.json(descriptor['path'])
        require(files.sha(descriptor['path'])==descriptor['sha256'] and child['sources_sha256']==parent['sources_sha256']
                and child['first_preflight_started_utc']==parent['first_preflight_started_utc']
                and child['absolute_deadline_utc']==parent['absolute_deadline_utc'],'Whole-stage child lock/clock differs')
        report=files.json(Path(child['cli']['output_dir'])/'independent_verification.json')
        require(report['passed'] is True and report['status']=='complete_verified_reference_refresh_run'
                and report['locked_policy_sha256']==descriptor['sha256'] and report['mode']==child['mode']
                and report['arm']==child['arm'] and report['flow']==child['flow']
                and report['steps']==child['cli']['max_steps'] and report['visits']==child['cli']['limit']
                and report['unique']==child['cli']['unique_count']
                and report['verifier_sha256']==parent['sources_sha256']['research/reference_refresh_20261107/verify_refresh_run.py'],
                'Missing/unmatched independently verified stage child')
        for name,receipt in report['required_artifacts'].items():
            actual=files.inputs.get(files.relative(name)) or files.physical(name)
            require(actual==receipt,'Accepted child input SHA/bytes no longer match')
        for name in total:total[name]+=report['forward_backward_counts'][name]
        if key in arms:formal[key]=files.json(Path(child['cli']['output_dir'])/'train_outcome.json')
    dev=files.rows(parent['dev_manifest'])
    require(len(dev)==200 and len({r['recipe_id'] for r in dev})==200
            and files.sha(parent['dev_manifest'])==parent['dev_manifest_sha256'],'Whole-stage dev denominator changed')
    orbit=files.rows(root/'orbit16x24.jsonl')
    require(orbit==independently_derive_orbit(dev),'Orbit is not exact original first16 x24 photo layouts')
    prediction_count=0;common_generation=None
    for key in arms:
        outcome=formal[key];child=files.json(parent['children'][key]['path']);seed=child['cli']['seed']
        identity={'base':parent['initial_identity']['base'],'adapter':outcome['final_adapter_identity']}
        for suffix,manifest,records in (('dev200',parent['dev_manifest'],dev),('orbit384',root/'orbit16x24.jsonl',orbit)):
            tag=key+'_step128_'+suffix;raw=root/tag/'predictions.jsonl'
            config=files.json(str(raw)+'.config.json');predictions=files.rows(raw)
            require(config['weight_identity']==identity and config['model_weights_sha256']==object_sha(identity)
                    and config['input']==INPUT and config['manifest_sha256']==files.sha(manifest)
                    and config['generation']['do_sample'] is False and config['generation']['max_new_tokens']==32
                    and config['cli']['temperature']==0 and config['cli']['num_samples']==1
                    and config['cli']['limit'] is None and config['cli']['process'] is False
                    and config['cli']['continue_on_error'] is False,'Endpoint prediction decoder/model/input changed')
            if common_generation is None:common_generation=config['generation']
            require(config['generation']==common_generation,'Endpoint prediction presets differ')
            coverage=files.json(root/(tag+'_coverage.json'))
            require(coverage['passed'] is True,'Endpoint prediction coverage failed')
            indexed={(r['qid'],r['presentation_id']):r for r in records};observed=set()
            require(len(predictions)==len(indexed)==len(records),'Endpoint missing/duplicate denominator')
            for row in predictions:
                position=(row['qid'],row['presentation_id'])
                require(position in indexed and position not in observed,'Duplicate/unknown endpoint prediction')
                observed.add(position);record=indexed[position]
                require(all(row[field]==record[field] for field in ('recipe_id','source_split','display_perm','target_order','target_labels'))
                        and row['technical_error'] is None and row['sample_index']==0 and row['prompt_kind']=='answer'
                        and row['run_id']==tag and row['seed']==seed
                        and row['checkpoint']==identity['adapter']['directory']
                        and row['model_weights_sha256']==object_sha(identity)
                        and row['decoding']=={'do_sample':False,'temperature':0.,'max_new_tokens':32,'top_p':1.,'top_k':0},
                        'Endpoint raw prediction identity/failure/decoder differs')
                score=independent_text_score(row['raw_response'],record)
                require(all(row['score'][field]==score[field] for field in score),'Endpoint raw text score differs')
            prediction_count+=len(predictions)
    require(prediction_count==4672,'Complete-stage prediction count differs')
    return {'passed':True,'status':'complete_reference_refresh_stage_analysis_gate_passed',
            'commands':74,'scientific_commands':73,'self_live_final_guard':live,
            'formal_arms':8,'technical_smokes':10,'dev_predictions':1600,'orbit_predictions':3072,
            'formal_endpoint_train_cells':2048,'formal_train_questions_per_arm':256,'formal_train_recipe_clusters':254,
            'new_q0_candidate_forwards':0,'historical_q0_candidate_forwards_not_recharged':25600,
            'all_training_and_scoring_forward_backward_counts':total,'resource_gate':gate,
            'locked_policy_sha256':files.sha(policy_path),
            'scientific_performance_summary_emitted':False,'model_GPU_network_calls':0,
            'limits':['This gate binds all fixed endpoints, raw records and persisted proofs. It is not a method-performance success verdict.',
                      'No model/GPU or Adam arithmetic is independently replayed.']}


def coordinate_gold(record):
    display,target=record['display_perm'],record['target_order']
    require(len(display)==len(target)==4 and sorted(display)==sorted(target)==list(range(4)), 'Invalid photo permutation')
    labels=['ABCD'[display.index(i)] for i in target]
    require(record['target_labels']==labels,'Gold label/photo coordinates differ')
    gold=ACTIONS.index(' '.join(labels))
    return gold,[int(i==gold) for i in range(24)]


def reconstruct(token_logps,candidates):
    require(len(token_logps)==len(candidates)==24,'Need complete24 support')
    require(tuple(c['text'] for c in candidates)==ACTIONS,'Canonical support order differs')
    scores=[]
    for values,c in zip(token_logps,candidates):
        ids=c['token_ids']
        require(len(values)==len(ids)==c['token_count']==5
                and ids[-1]==c['eos_token_id']==151645,'Need complete C5 including EOS')
        require(all(type(i) is int and i>=0 for i in ids)
                and all(finite(v) and v<=1e-6 for v in values),'Invalid candidate tokens/logps')
        scores.append(math.fsum(values))
    high=max(scores)
    logz=high+math.log(math.fsum(math.exp(v-high) for v in scores))
    require(logz<=1e-4,'Complete canonical mass exceeds one')
    logq=[v-logz for v in scores]
    q=[math.exp(v) for v in logq]
    close(math.fsum(q),1.,'q24 normalization',1e-10)
    return {'q':q,'logq':logq,'logZ':logz,'scores':scores,'C':5}


def independent_terms(method,policy,gold,reference_logq,beta):
    require(method in ('raw_sft','exact_kl'),'Unplanned method')
    require(len(reference_logq)==24 and all(finite(v) for v in reference_logq), 'Invalid reference logq')
    require(finite(beta) and (beta==0 if method=='raw_sft' else beta>0),'Invalid active beta')
    require(type(gold) is int and 0<=gold<24,'Invalid gold')
    p,l=policy['q'],policy['logq']
    rewards=[int(i==gold) for i in range(24)]
    u=[r-beta*(a-b) for r,a,b in zip(rewards,l,reference_logq)]
    mean=math.fsum(a*b for a,b in zip(p,u))
    # Derivative of qgold-beta KL with respect to complete-sequence log scores.
    coefficients=([float(i==gold) for i in range(24)] if method=='raw_sft' else
                  [a*(b-mean) for a,b in zip(p,u)])
    if method=='exact_kl':close(math.fsum(coefficients),0.,'exact gradient coefficient sum',1e-10)
    kl=math.fsum(a*(b-c) for a,b,c in zip(p,l,reference_logq))
    require(kl>=-1e-9,'Negative KL')
    return {'coefficients':coefficients,'u':u,'mean_u':mean,'advantages':None,
            'exact_expected_reward':p[gold],'kl_q_reference':kl,
            'regularized_expected_reward':p[gold]-beta*kl,
            'closed_ce_nll':-l[gold],'gold_index':gold,'active_beta':beta}


def analytic_target(reference,gold,beta):
    require(beta>0 and finite(beta),'Positive diagnostic beta required')
    tilted=[p*(math.exp(1/beta) if i==gold else 1.) for i,p in enumerate(reference['q'])]
    normalizer=math.fsum(tilted)
    target=[p/normalizer for p in tilted]
    logtarget=[math.log(p) for p in target]
    kl=math.fsum(p*(l-r) for p,l,r in zip(target,logtarget,reference['logq']))
    jstar=target[gold]-beta*kl
    close(jstar,beta*math.log1p(reference['q'][gold]*math.expm1(1/beta)),'qstar objective identity',1e-10)
    return {'q':target,'logq':logtarget,'Jstar':jstar}


def endpoint_objective(current,reference,gold,beta):
    terms=independent_terms('exact_kl',current,gold,reference['logq'],beta)
    target=analytic_target(reference,gold,beta)
    klstar=math.fsum(p*(l-s) for p,l,s in zip(current['q'],current['logq'],target['logq']))
    gap=target['Jstar']-terms['regularized_expected_reward']
    close(gap,beta*klstar,'Jstar-J=beta KL(current||qstar)',1e-10)
    require(gap>=-1e-9 and klstar>=-1e-9,'Negative analytic gap')
    return {'qgold':current['q'][gold], 'KL_q_q0':terms['kl_q_reference'],
            'J':terms['regularized_expected_reward'],'Jstar':target['Jstar'],
            'Jref':reference['q'][gold],'gap':gap,'KL_q_qstar':klstar,
            'available_gain':target['Jstar']-reference['q'][gold],
            'achieved_gain':terms['regularized_expected_reward']-reference['q'][gold]}


def fixed_stratum(reference,gold):
    first=max(range(24),key=lambda i:reference['logq'][i])
    target=analytic_target(reference,gold,BETA)
    second=max(range(24),key=lambda i:target['logq'][i])
    return ('initial_MAP_correct' if first==gold else
            'initial_wrong_qstar_MAP_correct' if second==gold else 'initial_wrong_qstar_MAP_wrong')


def gain_realization(rows):
    available=math.fsum(r['available_gain'] for r in rows)
    achieved=math.fsum(r['achieved_gain'] for r in rows)
    return {'sum_available':available,'sum_achieved':achieved,
            'ratio_of_sums':achieved/available if available!=0 else None}


def verify_effective_update(mode,initial,final,disclosed,scientific_outcome,step_norms):
    effective=initial!=final
    require(disclosed is effective and scientific_outcome==('effective_training' if effective else 'no_effective_adapter_update'),
            'Effective update disclosure differs')
    require(all(finite(value) and value>=0 for value in step_norms),'Invalid actual parameter update norm')
    if mode in ('smoke','path_smoke'):
        require(effective and step_norms and all(value>0 for value in step_norms),
                'Technical smoke/path must execute a real nonzero adapter update at every planned step')
    return effective


def verify_two_pass_records(records,unique_count,passes):
    require(len(records)==unique_count,'Manifest must contain unique inputs, not repeated visits')
    require(len({(r['qid'],r['presentation_id']) for r in records})==unique_count,'Duplicate question/presentation')
    require(passes in (1,2),'Unplanned pass count')
    for r in records:
        require(r['source_split']=='train' and r['technical_status']=='ok'
                and r.get('manifest_set') not in ('test','val','orbit','process'),'Heldout/damaged input')
        coordinate_gold(r)
    return records*passes


def check_policy_fields(row,policy):
    vector_close(row['q'],policy['q'],'stored q',1e-10)
    vector_close(row['logq'],policy['logq'],'stored logq',1e-10)
    close(row['logZ'],policy['logZ'],'stored logZ',1e-10)


def numeric_check(checks,token_logps,literal):
    require(len(checks)==1 and checks[0]['candidate_index']==0
            and checks[0]['literal_all_position_logits'] is literal,'Wrong literal numerical-check origin')
    check=checks[0]
    require(len(check['reference_token_logps'])==5
            and all(finite(value) and value<=1e-6 for value in check['reference_token_logps']),
            'Missing/nonfinite numerical token references')
    delta=max(abs(a-b) for a,b in zip(check['reference_token_logps'],token_logps[0]))
    close(check['max_token_logp_abs_difference'],delta,'numerical reference delta',1e-10)
    require(delta<=.01,'Numerical check outside frozen tolerance')


def verify_visit(visit,record,reference,visit_index,unique_count,method,beta,
                 generation,reference_hash,q0_hash,signature,seed):
    for key in ('qid','recipe_id','presentation_id','source_split','display_perm','target_order','target_labels'):
        require(visit[key]==reference[key]==record[key],'Visit/reference input coordinate mismatch: '+key)
    require(visit['visit_index']==visit_index and visit['pass_index']==visit_index//unique_count
            and visit['index_within_pass']==visit_index%unique_count
            and visit['reference_row_index']==visit_index%unique_count
            and visit['global_step_before_update']==visit_index//4,'Wrong pass/visit/update boundary')
    require(visit['reference_generation']==generation
            and visit['q0_reference_policy_sha256']==q0_hash
            and visit['active_reference_policy_sha256']==reference_hash
            and visit['run_signature_sha256']==signature,'Wrong active reference/run signature')
    require(visit['input_audit']==reference['input_audit']
            and visit['input_hash']==reference['input_audit']['input_hash'],'Visit changed input')
    require(tuple(visit['candidate_texts'])==ACTIONS
            and visit['candidate_token_ids']==[c['token_ids'] for c in reference['candidates']],
            'Visit candidate encoding differs')
    current=reconstruct(visit['token_logps_including_eos'],reference['candidates'])
    ref=reconstruct(reference['token_logps'],reference['candidates'])
    check_policy_fields(visit,current)
    check_policy_fields(reference,ref)
    vector_close(visit['sequence_logps'],current['scores'],'token sum',1e-6)
    vector_close(visit['reference_logq'],ref['logq'],'actual active reference',1e-10)
    close(visit['canonical_mass'],math.exp(current['logZ']),'canonical mass',1e-10)
    require(visit['completion_token_count']==5,'Loss divisor C must include EOS')
    gold,rewards=coordinate_gold(record)
    require(visit['exact_rewards']==reference['exact_rewards']==rewards
            and reference['gold_candidate_index']==gold,'Reward is not complete exact ordering')
    numeric_check(visit['numerical_reference_checks'],visit['token_logps_including_eos'],visit_index==0)
    terms=independent_terms(method,current,gold,ref['logq'],beta)
    for key in ('coefficients','u'):vector_close(visit[key],terms[key],'independent '+key)
    for key in ('mean_u','exact_expected_reward','kl_q_reference','regularized_expected_reward','closed_ce_nll','active_beta'):
        close(visit[key],terms[key],'independent '+key)
    require(visit['gold_index']==gold and visit['advantages'] is None
            and visit['sampled_indices'] is None and visit['sampling_uniforms'] is None,
            'Enumerated/raw training unexpectedly sampled')
    require(visit['sampling_rng_state_sha256']==object_sha(random.Random(seed).getstate()),'Sampling RNG changed')
    selected=[i for i,c in enumerate(visit['coefficients']) if c!=0]
    require(visit['nonzero_coefficients']==len(selected) and visit['group_active']==bool(selected)
            and [c['candidate_index'] for c in visit['backward_checks']]==selected,
            'Backward schedule is not exact machine-nonzero coefficients')
    require(all(finite(c['max_token_logp_difference']) and 0<=c['max_token_logp_difference']<=.01
                for c in visit['backward_checks']),'Gradient forward mismatch')
    require(visit['consumption_point']=='all_selected_candidate_backward_calls_returned','Unconsumed visit')
    return {'current_forwards':25,'gradient_forwards':len(selected),'backwards':len(selected),
            'active_group':bool(selected),'current':current,'reference':ref,'gold':gold}


def canonical_structure(value,tensor_encoder=None):
    """Deterministic logical fingerprint; tensor handling is explicit and CPU-only.

    JSON dictionary keys are type-tagged to distinguish optimizer integer IDs
    from string keys. Ordered sequences preserve order and container type.
    """
    if tensor_encoder is not None:
        encoded=tensor_encoder(value)
        if encoded is not None:
            return encoded
    if value is None or type(value) in (bool,int,str):
        return {'kind':type(value).__name__,'value':value}
    if type(value) is float:
        require(math.isfinite(value),'Nonfinite state scalar')
        return {'kind':'float','value':value}
    if isinstance(value,dict):
        pairs=[[canonical_structure(k,tensor_encoder),canonical_structure(v,tensor_encoder)] for k,v in value.items()]
        pairs.sort(key=lambda pair:json.dumps(pair[0],sort_keys=True,ensure_ascii=False))
        return {'kind':'dict','items':pairs}
    if isinstance(value,(list,tuple)):
        return {'kind':type(value).__name__,'items':[canonical_structure(v,tensor_encoder) for v in value]}
    if isinstance(value,bytes):
        return {'kind':'bytes','length':len(value),'sha256':sha(value)}
    raise ValueError('Unsupported state object: '+str(type(value)))


def logical_fingerprint(value,tensor_encoder=None):
    return object_sha(canonical_structure(value,tensor_encoder))


def cpu_tensor_encoder(value):
    """Lazy CPU tensor/array adapter; no CUDA API, model or optimizer construction."""
    module=type(value).__module__
    if module.startswith('torch'):
        import torch
        require(isinstance(value,torch.Tensor),'Unsupported torch snapshot object')
        tensor=value.detach().to(device='cpu').contiguous()
        require(bool(torch.isfinite(tensor).all().item()),'Nonfinite CPU snapshot tensor')
        raw=tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {'kind':'torch_tensor','dtype':str(tensor.dtype),'shape':list(tensor.shape),'sha256':sha(raw)}
    if module.startswith('numpy'):
        import numpy as np
        if isinstance(value,np.ndarray):
            require(not value.dtype.hasobject,'Object-array snapshot forbidden')
            array_value=np.ascontiguousarray(value)
            return {'kind':'numpy_array','dtype':str(array_value.dtype),'shape':list(array_value.shape),
                    'sha256':sha(array_value.tobytes())}
        if isinstance(value,np.generic):
            return {'kind':'numpy_scalar','dtype':str(value.dtype),
                    'value':canonical_structure(value.item(),cpu_tensor_encoder)}
    return None


def continuity_summary(payload):
    keys={'schema_version','global_step','optimizer','scheduler','gradients','model_modes','rng'}
    require(set(payload)==keys and payload['schema_version']==1,'Incomplete actual continuity snapshot')
    step=payload['global_step']
    require(type(step) is int and step in (1,64),'Wrong refresh boundary snapshot step')
    state=payload['optimizer']['state']
    optimizer_steps=[]
    for key in sorted(state):
        value=state[key]['step'];number=float(value.item()) if hasattr(value,'item') else float(value)
        require(math.isfinite(number) and number==int(number),'Noninteger actual Adam state step')
        optimizer_steps.append(int(number))
        require({'step','exp_avg','exp_avg_sq'}<=set(state[key]),'Missing Adam first/second moment')
    require(len(optimizer_steps)==392 and set(optimizer_steps)=={step},'Actual Adam state reset or incomplete')
    require(payload['scheduler']['last_epoch']==step,'Actual scheduler state reset')
    require(isinstance(payload['gradients'],dict) and len(payload['gradients'])==392
            and all(value is not None for value in payload['gradients'].values()),'Missing actual boundary gradients')
    require(isinstance(payload['model_modes'],dict) and payload['model_modes']
            and all(type(value) is bool for value in payload['model_modes'].values()),'Missing per-module modes')
    require(set(payload['rng'])=={'python_global','sampling_rng','numpy','cpu','cuda'},'Missing actual RNG stream')
    components={key:logical_fingerprint(payload[key],cpu_tensor_encoder) for key in sorted(keys)}
    return {'logical_sha256':object_sha(components),'component_sha256':components,
            'optimizer_steps':optimizer_steps,'scheduler_last_epoch':payload['scheduler']['last_epoch'],'global_step':step}


def load_cpu_state(files,name):
    # Snapshot paths and exact bytes must already be bound to the frozen run's
    # event/checkpoint audit before this loader is called. Trusted producer
    # snapshots contain NumPy RNG arrays and therefore need full CPU loading.
    torch_archive(files,name)
    import torch
    return torch.load(files.path(name),map_location='cpu',weights_only=False)


def verify_unchanged_components(before,after,components):
    require(set(before)==set(after)==set(components),'Continuity components incomplete or unexpected')
    for component in components:
        require(before[component]==after[component],'Refresh changed component: '+component)


def primary_differences(columns):
    names={flow+'_'+arm for flow in ('flow44','flow45') for arm in ARMS}
    require(set(columns)==names,'Require all8 columns')
    require({len(v) for v in columns.values()}=={200},'Require common200 recipe rows')
    require(all(v in (0,1) for values in columns.values() for v in values),'Exact accuracy arrays must be binary')
    return {control:[statistics_mean([columns[f+'_refresh_beta'][i]-columns[f+'_'+control][i]
                                     for f in ('flow44','flow45')]) for i in range(200)]
            for control in ('fixed_half_beta','raw_sft')}


def statistics_mean(values):
    return math.fsum(values)/len(values)


def linear_percentile(values,probability):
    require(values and 0<=probability<=1,'Invalid percentile input')
    ordered=sorted(values);position=(len(ordered)-1)*probability
    lower=math.floor(position);upper=math.ceil(position)
    return ordered[lower]+(position-lower)*(ordered[upper]-ordered[lower])


def joint_bootstrap(columns,draws=5000,seed=20261107):
    require(draws==5000 and seed==20261107,'Frozen joint bootstrap settings differ')
    effects=primary_differences(columns)
    rng=random.Random(seed)
    samples={key:[] for key in effects}
    flow_effects={flow+'_refresh_minus_'+control:
                 [columns[flow+'_refresh_beta'][i]-columns[flow+'_'+control][i] for i in range(200)]
                 for flow in ('flow44','flow45') for control in ('fixed_half_beta','raw_sft')}
    flow_samples={key:[] for key in flow_effects}
    # The same recipe resample is used in both contrasts and both flows.
    for _ in range(draws):
        indices=[rng.randrange(200) for _ in range(200)]
        for key,values in effects.items():
            samples[key].append(math.fsum(values[i] for i in indices)/200)
        for key,values in flow_effects.items():
            flow_samples[key].append(math.fsum(values[i] for i in indices)/200)
    return {'point_estimates_pp':{key:100*statistics_mean(v) for key,v in effects.items()},
            'intervals95_pp':{key:[100*linear_percentile(v,.025),100*linear_percentile(v,.975)]
                              for key,v in samples.items()},
            'per_flow_point_estimates_pp':{key:100*statistics_mean(v) for key,v in flow_effects.items()},
            'per_flow_intervals95_pp':{key:[100*linear_percentile(v,.025),100*linear_percentile(v,.975)]
                                       for key,v in flow_samples.items()},
            'samples':samples,'per_flow_samples':flow_samples,
            'draws':draws,'seed':seed,'engine':'Python stdlib random.Random;200 randrange(200) per draw',
            'percentile':'linear interpolation at (B-1)*p','independent_recipe_units':200}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=PROJECT)
    parser.add_argument('--q0-only',action='store_true')
    parser.add_argument('--stage-only',action='store_true')
    args=parser.parse_args()
    require(not args.output.exists(),'Fresh verifier output required')
    files=Files(args.root)
    if args.stage_only:
        require(not args.q0_only and args.run_dir is None,'Stage audit takes only parent policy/output')
        report=audit_stage_completion(files,args.policy)
        files.stable()
        report['required_artifacts'],report['observed_mutable_stage_journal_receipts']=files.report_receipts()
        report['verifier_sha256']=sha(Path(__file__).read_bytes())
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps({key:report[key] for key in ('passed','status','commands','formal_arms','technical_smokes')}))
        return
    policy=audit_child_policy(files,args.policy)
    q0=audit_q0(files,policy,args.policy)
    if args.q0_only:
        require(args.run_dir is None,'q0-only audit does not accept training run directory')
        files.stable()
        report={'passed':True,'status':'complete_verified_reference_refresh_q0',
                'arm':policy['arm'],'mode':policy['mode'],'flow':policy['flow'],
                'unique':len(q0['records']),'new_q0_candidate_forwards':0,
                'historical_whole1024_reference_candidate_forwards':25600,
                'reference_policy_sha256':q0['sha256'],'reference_spec_sha256':q0['spec_sha256'],
                'locked_policy_sha256':files.sha(args.policy),'required_artifacts':files.inputs,
                'verifier_sha256':sha(Path(__file__).read_bytes()),'model_GPU_network_calls':0}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps({key:report[key] for key in ('passed','status','flow','unique')}))
        return
    require(args.run_dir is not None,'Training audit needs --run-dir')
    report=audit_training(files,policy,args.policy,args.run_dir,q0)
    files.stable()
    report['required_artifacts'],report['observed_mutable_stage_journal_receipts']=files.report_receipts()
    report['verifier_sha256']=sha(Path(__file__).read_bytes())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({key:report[key] for key in ('passed','status','arm','mode','flow','steps','visits','unique')}))


if __name__=='__main__':
    try:
        main()
    except (ValueError,KeyError,AssertionError,FileNotFoundError) as error:
        print(json.dumps({'passed':False,'error':type(error).__name__+': '+str(error)}),file=sys.stderr)
        raise SystemExit(1)
