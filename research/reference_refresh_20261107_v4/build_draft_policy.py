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
        p='research/reference_refresh_20261107_v4/'+name;artifacts[p]=sha256_file(source_root/p)
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
        'budget_cny_cap':38.4,'reference_cny_per_hour':1.6,'budget_authorization':None,
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



def build_continuation(data_root,source_root,prepared_dir,output_directory,expected_cpu_tests):
    """Derive new immutable metadata; preserve every original science setting.

    Model/cache/physical bytes are NOT inspected from this local incomplete
    mirror. The fresh remote CPU gate must rehash the complete saved union.
    """
    from copy import deepcopy
    from refresh_contract import (OLD_POLICY,OLD_POLICY_SHA,ORIGINAL_FIRST,APPROVED_DEADLINE,
        AUTHORIZATION,TECHNICAL_KEYS,historical_commands)
    data_root=Path(data_root).resolve();source_root=Path(source_root).resolve()
    prepared_dir=Path(prepared_dir).resolve();require(not prepared_dir.exists(),'Fresh preparation only')
    require(output_directory=='exploratory/reference_refresh_20261107/stage_v4','One new stage_v4 only')
    old=json.loads((source_root/OLD_POLICY).read_bytes())
    require(sha256_file(source_root/OLD_POLICY)==OLD_POLICY_SHA,'Frozen original v3 policy required')
    def existing(name):
        path=Path(name);path=path.relative_to(BASE) if path.is_absolute() else path
        require('..' not in path.parts,'Unsafe old path')
        candidate=data_root/path
        return candidate if candidate.exists() else source_root/path
    def read(name):return json.loads(existing(name).read_bytes())
    def receipt(name):
        path=Path(name);path=path.relative_to(BASE) if path.is_absolute() else path
        actual=existing(path);return {'path':str(path),'sha256':sha256_file(actual),'bytes':actual.stat().st_size}
    root=Path(old['output_directory']);state_name=str(root/'state.json');state=read(state_name)
    technical={};physical={};additional={}
    def add(name):
        x=receipt(name);additional[x['path']]={'sha256':x['sha256'],'bytes':x['bytes']};return x
    for name in [OLD_POLICY,state_name,str(root/'cpu_preflight.json'),
        str(root/'generated_reference_receipt.json'),str(root/'whole_eight_arm_resource_gate.json')]:add(name)
    for descriptor in old['children'].values():add(descriptor['path'])
    for flow in FLOWS:
        for mode in ('smoke','formal'):
            name=str(root/f'q0_{mode}_{flow}_verification.json');add(name);proof=read(name)
            require(proof['passed'] is True,'Old four q0 audits must pass')
            for file,item in proof['required_artifacts'].items():
                require(file not in physical or physical[file]==item,'Conflicting old q0 evidence')
                physical[file]=item
    for key in TECHNICAL_KEYS:
        child=read(old['children'][key]['path']);name=str(Path(child['cli']['output_dir']).relative_to(BASE)/'independent_verification.json')
        technical[key]=add(name);proof=read(name)
        for file,item in proof['required_artifacts'].items():
            require(file not in physical or physical[file]==item,'Conflicting old technical evidence')
            physical[file]=item
    for name in ('TECHNICAL_RESOURCE_V3_REVIEW.json','TECHNICAL_RESOURCE_V3_REVIEW.md',
                 'INITIAL_RUNTIME_V3_REVIEW.json','INITIAL_RUNTIME_V3_REVIEW.md'):
        add('research/reference_refresh_20261107_v3/'+name)
    reused={'old_policy':receipt(OLD_POLICY),'old_state':receipt(state_name),
        'old_cpu':receipt(str(root/'cpu_preflight.json')),'old_commands':deepcopy(state['commands'][5:]),
        'old_children':{k:deepcopy(old['children'][k]) for k in TECHNICAL_KEYS},
        'technical_receipts':technical,'physical_receipts':physical,'additional_receipts':additional,
        'old_q0_count':4,'historical_new_model_calls':0,'technical_reexecution_allowed':False,
        'old_source_sha256':deepcopy(old['sources_sha256']),
        'scope':'20 historical technical events retain original policy, argv, PID/PGID and times. Fresh CPU and four fresh q0 audits precede all eight new formal runs.'}
    historical_commands(reused,state)
    auth=json.loads((source_root/AUTHORIZATION).read_bytes())
    require(auth['clock_reset'] is False and auth['original_first_preflight_utc']==ORIGINAL_FIRST
        and auth['approved_cumulative_wall_limit_seconds']==86400
        and auth['approved_absolute_deadline_utc']==APPROVED_DEADLINE,'Explicit same-clock24h authorization required')
    sources={name:sha256_file(source_root/name) for name in TRAIN_SOURCE_PATHS}
    artifacts=deepcopy(old['artifacts_sha256'])
    artifacts.update(old['sources_sha256'])
    artifacts.update({name:v['sha256'] for name,v in additional.items()})
    for name in (AUTHORIZATION,'research/reference_refresh_20261107_v4/CONTINUATION_PLAN.md'):
        artifacts[name]=sha256_file(source_root/name)
    prepared_dir.mkdir(parents=True);children={}
    for key,descriptor in old['children'].items():
        child=read(descriptor['path']);new=deepcopy(child)
        new.update(status='draft_not_authorized_for_execution',sources_sha256=sources,artifacts_sha256=artifacts,
            first_preflight_started_utc=ORIGINAL_FIRST,batch_clock_started_utc=ORIGINAL_FIRST,
            absolute_deadline_utc=APPROVED_DEADLINE)
        mode='formal' if child['mode']=='formal' else 'smoke';flow=child['flow']
        cache=f'{output_directory}/references/{mode}_{flow}/reference_policy.jsonl'
        new['cli'].update(output_dir=str(BASE/output_directory/key/'train'),reference_cache=str(BASE/cache),wall_seconds=86400)
        new.update(q0_reference_lineage=cache+'.lineage.json',reference_lineage=cache+'.lineage.json',
            generated_reference_receipt=output_directory+'/generated_reference_receipt.json',
            authorization_path=f'{output_directory}/authorizations/{"formal_" if mode=="formal" else ""}{key}.json')
        out=prepared_dir/f'child_{key}.json';write_json(out,new)
        children[key]={'path':str(out.relative_to(source_root)),'sha256':sha256_file(out)}
    result=deepcopy(old)
    result.update(status='draft_not_authorized_for_execution',wall_limit_seconds=86400,budget_cny_cap=38.4,
        budget_authorization={'path':AUTHORIZATION,'sha256':sha256_file(source_root/AUTHORIZATION)},
        first_preflight_started_utc=ORIGINAL_FIRST,absolute_deadline_utc=APPROVED_DEADLINE,
        output_directory=output_directory,sources_sha256=sources,artifacts_sha256=artifacts,
        expected_cpu_tests=expected_cpu_tests,children=children,reused_technical=reused,
        continuation=True,logical_commands=74,newly_executed_commands=54,historical_reused_commands=20,
        technical_gpu_reexecution=False)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True,type=Path)
    p.add_argument('--source-root',default=ROOT,type=Path);p.add_argument('--prepared-dir',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--expected-cpu-tests',type=int)
    p.add_argument('--output-directory',default='exploratory/reference_refresh_20261107/stage_v4')
    a=p.parse_args();require(not a.output.exists(),'Never overwrite policies')
    policy=build_continuation(a.data_root,a.source_root,a.prepared_dir,a.output_directory,a.expected_cpu_tests)
    write_json(a.output,policy);print(json.dumps({'status':policy['status'],'model_processor_gpu_calls':0,'output':str(a.output)}))


if __name__=='__main__':main()
