#!/usr/bin/env python3
"""Prepare reviewable draft only from sealed artifacts; cannot freeze or launch."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
from endpoint_bank_contract import (BANKS,BETA,DEV,DEV_SHA,ENDPOINTS,FULL_TRAIN,FULL_SHA,
    HISTORICAL_OUTCOME,INPUT,REVISION,SOURCE_PATHS,WALL_SECONDS,fixed_tasks,hash_json,load,read_jsonl,
    require,sha256_file,validate_policy,validate_records)
from recipeqa.modeling import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-root',required=True,type=Path)
    parser.add_argument('--expected-cpu-tests',required=True,type=int)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--stage-output',default='exploratory/endpoint_bank_20261106/stage_v1')
    args=parser.parse_args();snapshot=args.snapshot_root.resolve();require(not args.output.exists(),'Preserve prior drafts; choose a new version')
    receipts=load(snapshot/'LOCAL_SNAPSHOT.json')['files'];artifacts={}
    def artifact(name):
        path=snapshot/name;digest=sha256_file(path)
        require(receipts[name]=={'sha256':digest,'bytes':path.stat().st_size},'Unverified/mutated source snapshot artifact:'+name)
        artifacts[name]=digest;return load(path) if path.suffix=='.json' else path
    previous=artifact('research/coverage_20261106/PROTOCOL_V1.json')
    require(previous['initial_identity']['adapter']['directory'].endswith('/full_sft/train/checkpoint-768'),'Actual initial SFT768 identity missing')
    for name,digest in previous['sources_sha256'].items():
        require(sha256_file(ROOT/name)==digest,'Old scientific source changed');artifacts[name]=digest
    full_path=artifact(FULL_TRAIN);require(sha256_file(full_path)==FULL_SHA,'Wrong full train source')
    full=read_jsonl(full_path);train=full[:256];validate_records(train,'train256');require(len(full)==7558,'Complete original train source required')
    prepared=HERE/'prepared_v1'/'train256.jsonl';prepared.parent.mkdir(parents=True,exist_ok=True)
    raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train)
    if prepared.exists():require(prepared.read_text()==raw,'Prepared bank exists with changed contents')
    else:prepared.write_text(raw)
    prepared_rel=str(prepared.relative_to(ROOT));artifacts[prepared_rel]=sha256_file(prepared)
    dev_path=artifact(DEV);require(sha256_file(dev_path)==DEV_SHA,'Wrong fixed development source')
    dev=read_jsonl(dev_path);validate_records(dev,'dev200')
    initial=previous['initial_identity'];endpoints={'initial_sft768':{'adapter_path':'exploratory/principled_ordering_20261105/full_sft/train/checkpoint-768','weight_identity':initial}}
    for flow in ('flow44','flow45'):
        for method in ('raw_sft','exact_kl'):
            directory='exploratory/coverage_20261106/stage_v1/'+flow+'_'+method+'/train'
            outcome=artifact(directory+'/train_outcome.json');audit=artifact(directory+'/independent_verification.json')
            checkpoint=artifact(directory+'/checkpoint-256/finite_policy_checkpoint_audit.json')
            artifact(directory+'/run_signature.json')
            require(outcome['status']=='completed' and outcome['global_step']==256 and outcome['actual_prompt_visits']==1024
                    and audit['passed'] is True and audit['optimizer_steps']==256 and audit['limit']==1024,'Incomplete fixed coverage endpoint')
            item=next(x for x in outcome['checkpoints'] if x['step']==256)
            require(item['adapter_identity']==checkpoint['adapter_identity'],'Fixed checkpoint physical audit identity differs')
            key=flow+'_'+method+'_256';endpoints[key]={'adapter_path':directory+'/checkpoint-256',
                'weight_identity':{'base':initial['base'],'adapter':item['adapter_identity']},
                'identity_source_audit':directory+'/independent_verification.json'}
    reference='exploratory/coverage_20261106/stage_v1/reference/'
    cache={'source_cache':reference+'reference_policy.jsonl','source_config':reference+'reference_policy.jsonl.config.json',
        'source_completion':reference+'reference_policy.jsonl.completion.json','source_manifest':'research/coverage_20261106/prepared_v1/formal_flow44.jsonl',
        'source_signature':reference+'run_signature.json','source_audit':reference+'independent_verification.json',
        'rows':256,'output_directory':'cached_initial_train'}
    for name in ('source_cache','source_config','source_completion','source_manifest','source_signature','source_audit'):artifact(cache[name])
    cache['source_receipt_sha256']={cache[k]:artifacts[cache[k]] for k in ('source_cache','source_config','source_completion','source_manifest','source_signature','source_audit')}
    required_versions=load(snapshot/cache['source_config'])['versions']
    sources={name:sha256_file(ROOT/name) for name in SOURCE_PATHS}
    history=artifact(HISTORICAL_OUTCOME)
    require(history['status']=='completed_reference_only' and history['rows']==1024
            and history['updates']==0 and history['elapsed_seconds']>0,'Historical reference timing incomplete')
    timing={'reference_outcome':HISTORICAL_OUTCOME,'sha256':artifacts[HISTORICAL_OUTCOME],
            'rows':1024,'elapsed_seconds':history['elapsed_seconds']}
    dev_greedy={}
    for endpoint in ENDPOINTS:
        prefix=('exploratory/principled_ordering_20261105/full_sft/dev768' if endpoint=='initial_sft768'
                else 'exploratory/coverage_20261106/stage_v1/'+endpoint.removesuffix('_256')+'_step256_dev200')
        names={'predictions':prefix+'/predictions.jsonl','config':prefix+'/predictions.jsonl.config.json'}
        for name in names.values():artifact(name)
        dev_greedy[endpoint]=names
    for name in ('exploratory/coverage_20261106/stage_v1/state.json','exploratory/coverage_20261106/stage_v1/summary.json'):artifact(name)
    coverage_review='research/coverage_20261106/final_results_audit_v1/COVERAGE_RESULTS_VERIFICATION.json'
    artifacts[coverage_review]=sha256_file(ROOT/coverage_review)
    artifacts['research/endpoint_bank_20261106/PLAN.md']=sha256_file(HERE/'PLAN.md')
    artifacts['research/endpoint_bank_20261106/IMPLEMENTATION_NOTES.md']=sha256_file(HERE/'IMPLEMENTATION_NOTES.md')
    policy={'schema_version':1,'status':'draft_endpoint_bank','wall_limit_seconds':WALL_SECONDS,
        'budget_cny_cap':4.8,'reference_cny_per_hour':1.6,'budget_authorization':'New independent3h/4.8CNY read-only stage announced in chat; original1000CNY project cap unchanged.',
        'first_preflight_started_utc':None,'absolute_deadline_utc':None,'output_directory':args.stage_output,
        'model_path':'models/Qwen3-VL-2B-Instruct','image_root':'data/images','revision':REVISION,
        'input':INPUT,'beta':BETA,'required_versions':required_versions,'candidate_policy':'canonical24_full_sequence_plusEOS_temperature1',
        'tie_policy':'exact_maximum_then_lowest_canonical_index','endpoints':endpoints,
        'banks':{'train256':{'manifest':prepared_rel,'manifest_sha256':sha256_file(prepared),'rows':256,
                            'selected_records_sha256':hash_json(train),'source_manifest':FULL_TRAIN,'source_manifest_sha256':FULL_SHA},
                 'dev200':{'manifest':DEV,'manifest_sha256':DEV_SHA,'rows':200,'selected_records_sha256':hash_json(dev),
                           'source_manifest':DEV,'source_manifest_sha256':DEV_SHA}},
        'tasks':fixed_tasks(),'formal_rows':2024,'pilot_rows':18,'pilot_rows_per_task':2,
        'cached_initial_train':cache,'dev_greedy':dev_greedy,'coverage_complete_review':coverage_review,
        'historical_timing':timing,'timing_gate_policy':'first_row_once_remaining_second_row_with_historical_floor_1.5_plus600',
        'expected_cpu_tests':args.expected_cpu_tests,'sources_sha256':sources,
        'artifacts_sha256':artifacts,'test_access':False,'confirmation_access':False,'resume':False,
        'automatic_retry':False,'optimizer_steps':0,'backward_calls':0,'generation_calls':0,
        'snapshot_captured_at':load(snapshot/'LOCAL_SNAPSHOT.json')['captured_at'],
        'model_binary_limit':'Model identities are read from sealed remote audits, not loaded locally; all five physical identities must match before any pilot.'}
    validate_policy(policy,locked=False);write_json(args.output,policy)
    print(json.dumps({'status':'draft_only_not_executed','path':str(args.output),'sha256':sha256_file(args.output),
        'formal_tasks':9,'formal_rows':2024,'pilot_rows_separate':18,'model_calls':0,'source_files':len(sources)}))


if __name__=='__main__':main()
