#!/usr/bin/env python3
"""Prepare fixed manifests and unexecutable stage/child policies; no inference."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from recipeqa.modeling import InputConfig,read_jsonl,sha256_file,write_json,hash_json
from coverage_contract import (BASE,BETA,WALL_SECONDS,FULL_TRAIN,FULL_SHA,INITIAL_ADAPTER,MODEL_PATH,REVISION,
    TRAIN_SOURCE_PATHS,DEV_SHA,FLOWS,METHODS,fixed_pool,ordered_pools,validate_prior_exposure,require)

INITIAL_SIGNATURE='exploratory/principled_ordering_20261105/finite_trials_v2/flow44/raw_sft/run_signature.json'
PRIOR_VISITS='exploratory/principled_ordering_20261105/full_sft/train/actual_prompt_visits.jsonl'
PRIOR_SIGNATURE='exploratory/principled_ordering_20261105/full_sft/train/run_signature.json'
ORBIT_STATE='exploratory/grounding_equivariance_20261106/orbit_v1/state.json'


def jsonl(path,rows):
    path=Path(path);require(not path.exists(),'Preparation cannot overwrite an existing manifest')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        for row in rows:handle.write(json.dumps(row,allow_nan=False)+'\n')


def build(data_root,source_root,prepared_dir,output_directory,expected_cpu_tests=None):
    data_root=Path(data_root).resolve();source_root=Path(source_root).resolve();prepared_dir=Path(prepared_dir).resolve()
    relative=str(prepared_dir.relative_to(source_root))
    require(not prepared_dir.exists(),'Fresh preparation directory required')
    require(sha256_file(data_root/FULL_TRAIN)==FULL_SHA and sha256_file(data_root/'data/manifests/val.jsonl')==DEV_SHA,'Frozen original training/development bytes changed')
    full=read_jsonl(data_root/FULL_TRAIN);pool=fixed_pool(full);orders=ordered_pools(pool)
    prior_signed=json.loads((data_root/PRIOR_SIGNATURE).read_text())
    require(prior_signed['sha256']==hash_json(prior_signed['payload']),'Actual initialSFT training signature invalid')
    exposure=validate_prior_exposure(pool,read_jsonl(data_root/PRIOR_VISITS),prior_signed['sha256'])
    signed=json.loads((data_root/INITIAL_SIGNATURE).read_text());require(signed['sha256']==hash_json(signed['payload']),'Initial identity source signature invalid')
    initial=signed['payload']['initial_identity']
    require(initial['adapter']['directory']==str(BASE/INITIAL_ADAPTER) and initial['base']['directory']==str(BASE/MODEL_PATH),'Expected SFT768/base identity only')
    prepared_dir.mkdir(parents=True)
    manifests={}
    for flow,records in orders.items():
        for mode,values in [('formal',records),('smoke',records[:4])]:
            path=prepared_dir/f'{mode}_{flow}.jsonl';jsonl(path,values)
            manifests[(flow,mode)]={'path':str(path.relative_to(source_root)),'sha256':sha256_file(path),'rows':len(values)}
    receipt={'status':'cpu_fixed_pool_prepared_no_model_inference','full_source_sha256':FULL_SHA,
        'pool_selected_records_sha256':hash_json(pool),'pool_source_indices':list(range(1024)),
        'source_was_already_seen_by_initial_sft':exposure,
        'flow44_order':'original first1024 file order','flow45_order':'random.Random(45).shuffle copy once; same fullrecord bytes',
        'flow_orders':{flow:[{'qid':r['qid'],'presentation_id':r['presentation_id'],'record_sha256':hash_json(r)} for r in records] for flow,records in orders.items()}}
    write_json(prepared_dir/'pool_receipt.json',receipt)
    sources={p:sha256_file(source_root/p) for p in TRAIN_SOURCE_PATHS}
    artifacts={path:sha256_file(data_root/path) for path in (FULL_TRAIN,INITIAL_SIGNATURE,PRIOR_VISITS,PRIOR_SIGNATURE,ORBIT_STATE,'data/manifests/val.jsonl')}
    artifacts.update({spec['path']:spec['sha256'] for spec in manifests.values()})
    artifacts[f'{relative}/pool_receipt.json']=sha256_file(prepared_dir/'pool_receipt.json')
    for path in ('research/coverage_20261106/PLAN.md','research/coverage_20261106/ANALYSIS_PLAN.md',
                 'research/coverage_20261106/TRAINER_LINEAGE.json'):
        artifacts[path]=sha256_file(source_root/path)
    pool_spec={'full_train_path':FULL_TRAIN,'full_train_sha256':FULL_SHA,'full_train_count':7558,
        'pool_count':1024,'pool_source_indices':list(range(1024)),
        'prior_visits_path':PRIOR_VISITS,'prior_visits_sha256':artifacts[PRIOR_VISITS],
        'prior_signature_path':PRIOR_SIGNATURE,'prior_signature_sha256':artifacts[PRIOR_SIGNATURE],
        'prior_run_signature_sha256':prior_signed['sha256'],
        'initial_sft_step':768,'initial_consumed_visits':6144,'flow44_order':'original',
        'flow45_shuffle_seed':45,'pool_selected_records_sha256':hash_json(pool)}
    children={}
    def child(key,mode,flow,method):
        spec=manifests[(flow,'smoke' if mode=='smoke' else 'formal')]
        ref=None if mode=='reference' else f'{output_directory}/reference/reference_policy.jsonl' if flow=='flow44' and mode=='formal' else f'{output_directory}/references/{mode}_{flow}/reference_policy.jsonl'
        out=f'{output_directory}/reference' if mode=='reference' else f'{output_directory}/{key}/train'
        cli={'method':method,'model_path':str(BASE/MODEL_PATH),'adapter_path':str(BASE/INITIAL_ADAPTER),
            'revision':REVISION,'manifest':str(BASE/spec['path']),'expected_manifest_sha256':spec['sha256'],
            'image_root':str(BASE/'data/images'),'output_dir':str(BASE/out),
            'reference_cache':str(BASE/ref) if ref else None,'prepare_reference_only':mode=='reference',
            'limit':4 if mode=='smoke' else 1024,'max_steps':1 if mode=='smoke' else 256,
            'learning_rate':5e-6,'beta':BETA if method=='exact_kl' else 0.,'seed':int(flow[-2:]),
            'save_steps':1 if mode=='smoke' else 64,'wall_seconds':WALL_SECONDS,'logp_atol':.01,
            'min_pixels':65536,'max_pixels':262144,'recipe_token_budget':1536,'max_input_tokens':4096}
        result={'status':'draft_not_authorized_for_execution','mode':mode,'method':method,'flow':flow,'cli':cli,
            'initial_identity':initial,'sources_sha256':sources,'artifacts_sha256':artifacts,
            'reward':'unique_complete_order_exact_0_1','selection':'fixed256_primary_fixed64_diagnostic_no_dev_selection',
            'optimizer':{'accumulation':4,'clip_norm':1.,'canonical_C_including_eos':5,'fresh_adamw':True},
            'first_preflight_started_utc':None,'batch_clock_started_utc':None,'absolute_deadline_utc':None,
            'reference_lineage':None if mode=='reference' or (mode=='formal' and flow=='flow44') else ref+'.lineage.json',
            'reference_generation':'One actual new1024 SFT768 cache, source1024 or identity-only reordered/subset; no stale256 cache',
            'generated_reference_binding':'Fixed source/function/paths now; generated_reference_receipt seals actualSHA before firstsmoke; actualSHA independently bound in each run_signature',
            'generated_reference_receipt':f'{output_directory}/generated_reference_receipt.json',
            'pool_receipt':f'{relative}/pool_receipt.json','reference_source_manifest':manifests[('flow44','formal')]['path']}
        result['pool_spec']=pool_spec
        path=prepared_dir/f'child_{key}.json';write_json(path,result)
        children[key]={'path':str(path.relative_to(source_root)),'sha256':sha256_file(path)}
    child('reference','reference','flow44','raw_sft')
    for flow in FLOWS:
        for method in METHODS:
            arm=f'{flow}_{method}';child('smoke_'+arm,'smoke',flow,method);child(arm,'formal',flow,method)
    return {'schema_version':1,'status':'draft_not_authorized_for_execution','wall_limit_seconds':WALL_SECONDS,
        'budget_cny_cap':25.6,'reference_cny_per_hour':1.6,'budget_authorization':None,
        'first_preflight_started_utc':None,'absolute_deadline_utc':None,'output_directory':output_directory,
        'input':asdict(InputConfig()),'model_path':MODEL_PATH,'initial_adapter':INITIAL_ADAPTER,
        'initial_identity':initial,'revision':REVISION,'image_root':'data/images','primary_step':256,'diagnostic_step':64,
        'flow_order':list(FLOWS),'methods':list(METHODS),'expected_cpu_tests':expected_cpu_tests,
        'dev_manifest':'data/manifests/val.jsonl','dev_manifest_sha256':DEV_SHA,
        'historical_orbit_state':ORBIT_STATE,'orbit_first_recipes':16,'orbit_layouts':24,
        'manifests':{f'{flow}_{mode}':spec for (flow,mode),spec in manifests.items()},
        'pool_receipt':f'{relative}/pool_receipt.json','children':children,
        'pool_spec':pool_spec,
        'sources_sha256':sources,'artifacts_sha256':artifacts,
        'generated_reference_policy':'Frozen deterministic source/cache transform; actual SHA sealed before any smoke update, no selection',
        'test_access':False,'confirmation_access':False,'no_resume':True,'selection':'fixed256_primary_fixed64_diagnostic_no_dev_selection'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--source-root',default=ROOT,type=Path);parser.add_argument('--prepared-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--output-directory',default='exploratory/coverage_20261106/stage_v1')
    parser.add_argument('--expected-cpu-tests',type=int)
    args=parser.parse_args();require(not args.output.exists(),'Never overwrite an existing/frozen stage policy')
    policy=build(args.data_root,args.source_root,args.prepared_dir,args.output_directory,args.expected_cpu_tests)
    write_json(args.output,policy);print(json.dumps({'status':policy['status'],'model_processor_gpu_calls':0,'output':str(args.output)}))


if __name__=='__main__':main()
