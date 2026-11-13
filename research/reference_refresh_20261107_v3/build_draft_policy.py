#!/usr/bin/env python3
"""Prepare the complete reference-refresh design; no clock, inference or freeze."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from recipeqa.modeling import InputConfig,read_jsonl,sha256_file,write_json,hash_json
from refresh_contract import (BASE,BETA,WALL_SECONDS,FULL_TRAIN,FULL_SHA,INITIAL_ADAPTER,MODEL_PATH,
    REVISION,TRAIN_SOURCE_PATHS,DEV_SHA,FLOWS,ARMS,ARM_KINDS,PATH_SMOKES,budget_for,
    fixed_pool,ordered_pools,validate_prior_exposure,require)

INITIAL_SIGNATURE='exploratory/principled_ordering_20261105/finite_trials_v2/flow44/raw_sft/run_signature.json'
PRIOR_VISITS='exploratory/principled_ordering_20261105/full_sft/train/actual_prompt_visits.jsonl'
PRIOR_SIGNATURE='exploratory/principled_ordering_20261105/full_sft/train/run_signature.json'
ORBIT_STATE='exploratory/grounding_equivariance_20261106/orbit_v1/state.json'
COVERAGE='exploratory/coverage_20261106/stage_v1'
Q0_CACHE=COVERAGE+'/reference/reference_policy.jsonl'
Q0_MANIFEST='research/coverage_20261106/prepared_v1/formal_flow44.jsonl'
HISTORICAL_ARMS=tuple(f'{f}_{m}' for f in FLOWS for m in ('raw_sft','exact_kl'))


def jsonl(path,rows):
    path=Path(path);require(not path.exists(),'Never overwrite prepared manifests')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')


def command_schedule():
    """Literal complete order shared by producer/independent completeness checks."""
    result=['cpu_preflight']
    result += ['q0_'+mode+'_'+flow+'_verify' for flow in FLOWS for mode in ('smoke','formal')]
    for key in (*('smoke_'+a for a in ARMS),*PATH_SMOKES):result += [key+'_train',key+'_verify']
    for arm in ARMS:
        result += [arm+'_train',arm+'_verify',arm+'_step128_dev200',arm+'_step128_dev200_coverage',
                   arm+'_step128_orbit384',arm+'_step128_orbit384_coverage']
    result.append('stage_global_verify')
    return result


def build(data_root,source_root,prepared_dir,output_directory,expected_cpu_tests=None):
    data_root=Path(data_root).resolve();source_root=Path(source_root).resolve();prepared_dir=Path(prepared_dir).resolve()
    relative=str(prepared_dir.relative_to(source_root))
    require(not prepared_dir.exists(),'Fresh preparation directory only')
    require(Path(output_directory).parts[:2]==('exploratory','reference_refresh_20261107')
            and '..' not in Path(output_directory).parts,'New confined output namespace required')
    require(sha256_file(data_root/FULL_TRAIN)==FULL_SHA
            and sha256_file(data_root/'data/manifests/val.jsonl')==DEV_SHA,'Original fulltrain/dev bytes changed')
    pool=fixed_pool(read_jsonl(data_root/FULL_TRAIN));orders=ordered_pools(pool)
    require(len({r['recipe_id'] for r in pool})==254,'Original256/254 recipe pool required')
    prior=json.loads((data_root/PRIOR_SIGNATURE).read_text())
    require(prior['sha256']==hash_json(prior['payload']),'Actual initial SFT signature invalid')
    exposure=validate_prior_exposure(pool,read_jsonl(data_root/PRIOR_VISITS),prior['sha256'])
    initial_signed=json.loads((data_root/INITIAL_SIGNATURE).read_text())
    require(initial_signed['sha256']==hash_json(initial_signed['payload']),'Initial identity signature invalid')
    initial=initial_signed['payload']['initial_identity']
    require(initial['adapter']['directory']==str(BASE/INITIAL_ADAPTER)
            and initial['base']['directory']==str(BASE/MODEL_PATH),'Original SFT768/base identity required')
    prepared_dir.mkdir(parents=True)
    manifests={}
    for flow,records in orders.items():
        for mode,rows in (('formal',records),('smoke',records[:4])):
            path=prepared_dir/f'{mode}_{flow}.jsonl';jsonl(path,rows)
            manifests[(flow,mode)]={'path':str(path.relative_to(source_root)),'sha256':sha256_file(path),'rows':len(rows)}
    pool_receipt={'status':'cpu_fixed256_two_pass_pool_no_inference','full_source_sha256':FULL_SHA,
        'pool_selected_records_sha256':hash_json(pool),'pool_source_indices':list(range(256)),
        'unique_questions':256,'unique_recipes':254,'passes':2,
        'source_was_already_seen_by_initial_sft':exposure,
        'flow44_order':'original first256 file order','flow45_order':'random.Random(45).shuffle once of same256',
        'second_pass':'exact same ordered input records and display as first pass; no resampling',
        'flow_orders':{f:[{'qid':r['qid'],'presentation_id':r['presentation_id'],'record_sha256':hash_json(r)} for r in rs] for f,rs in orders.items()}}
    write_json(prepared_dir/'pool_receipt.json',pool_receipt)
    sources={p:sha256_file(source_root/p) for p in TRAIN_SOURCE_PATHS}
    old_paths=[FULL_TRAIN,INITIAL_SIGNATURE,PRIOR_VISITS,PRIOR_SIGNATURE,ORBIT_STATE,'data/manifests/val.jsonl',
        COVERAGE+'/state.json',Q0_CACHE,Q0_CACHE+'.config.json',Q0_CACHE+'.completion.json',Q0_MANIFEST,
        COVERAGE+'/reference/reference_only_outcome.json',COVERAGE+'/reference/run_signature.json',
        COVERAGE+'/reference/independent_verification.json']
    for arm in HISTORICAL_ARMS:
        old_paths += [COVERAGE+'/'+arm+'/train/'+n for n in ('train_outcome.json','step_times.jsonl','independent_verification.json')]
    artifacts={p:sha256_file(data_root/p) for p in old_paths}
    artifacts.update({s['path']:s['sha256'] for s in manifests.values()})
    artifacts[relative+'/pool_receipt.json']=sha256_file(prepared_dir/'pool_receipt.json')
    for name in ('PLAN.md','ANALYSIS_PLAN.md','TRAINER_LINEAGE.json','CONTRACT_DRAFT.md','IMPLEMENTATION_PLAN.md'):
        p='research/reference_refresh_20261107_v3/'+name;artifacts[p]=sha256_file(source_root/p)
    pool_spec={'full_train_path':FULL_TRAIN,'full_train_sha256':FULL_SHA,'full_train_count':7558,
        'pool_count':256,'pool_unique_recipes':254,'pool_source_indices':list(range(256)),
        'prior_visits_path':PRIOR_VISITS,'prior_visits_sha256':artifacts[PRIOR_VISITS],
        'prior_signature_path':PRIOR_SIGNATURE,'prior_signature_sha256':artifacts[PRIOR_SIGNATURE],
        'prior_run_signature_sha256':prior['sha256'],'initial_sft_step':768,'initial_consumed_visits':6144,
        'flow44_order':'original','flow45_shuffle_seed':45,'second_pass':'identical same flow order',
        'pool_selected_records_sha256':hash_json(pool)}
    children={}
    for mode,flow,arm in ([(mode,f,a) for f in FLOWS for a in ARM_KINDS for mode in ('smoke','formal')]
                          +[('path_smoke',f,'refresh_beta') for f in FLOWS]):
        key=('smoke_' if mode=='smoke' else 'path_smoke_' if mode=='path_smoke' else '')+flow+'_'+arm
        spec=manifests[(flow,'formal' if mode=='formal' else 'smoke')];budget=budget_for(mode)
        cache=f'{output_directory}/references/{"formal" if mode=="formal" else "smoke"}_{flow}/reference_policy.jsonl'
        cli={'method':'raw_sft' if arm=='raw_sft' else 'exact_kl','model_path':str(BASE/MODEL_PATH),
            'adapter_path':str(BASE/INITIAL_ADAPTER),'revision':REVISION,'manifest':str(BASE/spec['path']),
            'expected_manifest_sha256':spec['sha256'],'image_root':str(BASE/'data/images'),
            'output_dir':str(BASE/output_directory/key/'train'),'reference_cache':str(BASE/cache),
            'prepare_reference_only':False,**budget,'learning_rate':5e-6,
            'beta':0. if arm=='raw_sft' else BETA/2 if arm=='fixed_half_beta' else BETA,
            'seed':int(flow[-2:]),'refresh_boundary_step':budget['max_steps']//2 if arm=='refresh_beta' and mode!='smoke' else None,
            'final_bank_count':budget['unique_count'],'wall_seconds':WALL_SECONDS,'logp_atol':.01,
            'min_pixels':65536,'max_pixels':262144,'recipe_token_budget':1536,'max_input_tokens':4096}
        child={'status':'draft_not_authorized_for_execution','mode':mode,'arm':arm,'flow':flow,'cli':cli,
            'initial_identity':initial,'sources_sha256':sources,'artifacts_sha256':artifacts,'pool_spec':pool_spec,
            'reward':'unique_complete_order_exact_0_1','selection':'fixed128_primary_step64_boundary_no_dev_selection',
            'optimizer':{'accumulation':4,'clip_norm':1.,'canonical_C_including_eos':5,'fresh_adamw':True},
            'first_preflight_started_utc':None,'batch_clock_started_utc':None,'absolute_deadline_utc':None,
            'q0_reference_lineage':cache+'.lineage.json','reference_lineage':cache+'.lineage.json',
            'generated_reference_receipt':f'{output_directory}/generated_reference_receipt.json',
            'generated_reference_binding':'Identity-only existing true1024 q0 transform; zero new q0 forwards; seal before any smoke',
            'authorization_path':f'{output_directory}/authorizations/{"formal_" if mode=="formal" else ""}{key}.json',
            'pool_receipt':relative+'/pool_receipt.json','reference_source_manifest':Q0_MANIFEST,
            'q0_source_cache':Q0_CACHE}
        path=prepared_dir/f'child_{key}.json';write_json(path,child)
        children[key]={'path':str(path.relative_to(source_root)),'sha256':sha256_file(path)}
    return {'schema_version':1,'status':'draft_not_authorized_for_execution','wall_limit_seconds':WALL_SECONDS,
        'budget_cny_cap':32.,'reference_cny_per_hour':1.6,'budget_authorization':None,
        'first_preflight_started_utc':None,'absolute_deadline_utc':None,'output_directory':output_directory,
        'input':asdict(InputConfig()),'model_path':MODEL_PATH,'initial_adapter':INITIAL_ADAPTER,
        'initial_identity':initial,'revision':REVISION,'image_root':'data/images','primary_step':128,'boundary_step':64,
        'flow_order':list(FLOWS),'methods':list(ARM_KINDS),'formal_order':list(ARMS),
        'technical_order':[*('smoke_'+a for a in ARMS),*PATH_SMOKES],'command_schedule':command_schedule(),
        'expected_cpu_tests':expected_cpu_tests,'dev_manifest':'data/manifests/val.jsonl','dev_manifest_sha256':DEV_SHA,
        'historical_orbit_state':ORBIT_STATE,'historical_coverage_state':COVERAGE+'/state.json',
        'historical_training_runs':{a:COVERAGE+'/'+a+'/train' for a in HISTORICAL_ARMS},
        'historical_reference_outcome':COVERAGE+'/reference/reference_only_outcome.json',
        'q0_source_cache':Q0_CACHE,'q0_source_manifest':Q0_MANIFEST,
        'q0_source_signature':COVERAGE+'/reference/run_signature.json',
        'q0_source_verification':COVERAGE+'/reference/independent_verification.json',
        'orbit_first_recipes':16,'orbit_layouts':24,'manifests':{f'{f}_{m}':s for (f,m),s in manifests.items()},
        'pool_receipt':relative+'/pool_receipt.json','children':children,'pool_spec':pool_spec,
        'sources_sha256':sources,'artifacts_sha256':artifacts,
        'generated_reference_policy':'Four deterministic q0 subsets/reorderings sealed and independently checked before any model call',
        'test_access':False,'confirmation_access':False,'no_resume':True,
        'selection':'fixed128_primary_step64_boundary_no_dev_selection'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--source-root',default=ROOT,type=Path);p.add_argument('--prepared-dir',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--expected-cpu-tests',type=int)
    p.add_argument('--output-directory',default='exploratory/reference_refresh_20261107/stage_v1')
    a=p.parse_args();require(not a.output.exists(),'Never overwrite policies')
    policy=build(a.data_root,a.source_root,a.prepared_dir,a.output_directory,a.expected_cpu_tests)
    write_json(a.output,policy);print(json.dumps({'status':policy['status'],'model_processor_gpu_calls':0,'output':str(a.output)}))


if __name__=='__main__':main()
