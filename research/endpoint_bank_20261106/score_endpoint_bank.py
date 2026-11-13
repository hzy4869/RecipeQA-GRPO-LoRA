#!/usr/bin/env python3
"""Fixed endpoint likelihoods only: no generation, optimizer or backward calls."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'exploratory/principled_ordering_20261105'))
from recipeqa.modeling import (InputConfig,encode_messages,frozen_vision_fingerprint,
    hash_json,load_images,load_model,load_processor,model_weight_identity,
    package_versions,prepare_record,read_jsonl,sha256_file,write_json)
# Only these immutable likelihood helpers are used; never its dev-only main,
# probability summary, get_greedy or generation path.
from score_permutations import candidate_suffixes,score_candidates
from endpoint_bank_contract import (BASE,INPUT,CANDIDATES,bank_records,confined,
    image_path,load,now,remaining_seconds,require,task_spec,validate_authorization,validate_environment,validate_policy)


def append(path,row):
    with Path(path).open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        handle.flush()


def lora_fingerprint(model):
    """Inference LoRA is frozen; never use the requires_grad-only old helper."""
    digest=hashlib.sha256();count=numel=0
    for name,p in model.named_parameters():
        if 'lora_' not in name:continue
        require('.language_model.layers.' in name and not p.requires_grad,'Unexpected trainable/nonlanguage adapter tensor')
        count+=1;numel+=p.numel();digest.update(name.encode())
        digest.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    require(count==392 and numel==17432576,'Inference LoRA inventory is missing or changed')
    return {'sha256':digest.hexdigest(),'tensors':count,'numel':numel,
        'selection':'all language-model lora_ parameters regardless of requires_grad','dtype':'float32 bytes'}


def check_frozen_sources(p,root=BASE):
    for group in ('sources_sha256','artifacts_sha256'):
        for name,digest in p[group].items():
            require(sha256_file(confined(name,root))==digest,'Frozen source/artifact changed:'+name)


def cached_initial_train(p,policy_path,output):
    """Identity-preserving reuse of the actual initial1024 reference prefix."""
    started=time.monotonic();started_utc=now()
    cache=p['cached_initial_train'];require(not output.exists(),'Fresh cache output only; never overwrite')
    for name,digest in cache['source_receipt_sha256'].items():
        require(sha256_file(confined(name))==digest,'Cached initial reference source changed')
    source=read_jsonl(confined(cache['source_cache']));config=load(confined(cache['source_config']))
    done=load(confined(cache['source_completion']));signed=load(confined(cache['source_signature']))
    audit=load(confined(cache['source_audit']));records=bank_records(p,'train256')
    source_records=read_jsonl(confined(cache['source_manifest']))
    require(len(source)==len(source_records)==1024 and records==source_records[:256],
            'Cached bank is not the exact original source prefix')
    require(done['status']=='complete_fixed_initial_reference' and done['rows']==1024
            and done['cache_sha256']==sha256_file(confined(cache['source_cache']))
            and done['reference_spec_sha256']==hash_json(config),'Incomplete actual cached source')
    require(audit['passed'] is True and audit['status']=='complete_verified_coverage_reference_only'
            and audit['updates']==0 and audit['reference_policy_sha256']==done['cache_sha256'],
            'Cached source lacks complete independent zero-update audit')
    require(signed['sha256']==hash_json(signed['payload']) and config['initial_identity']==p['endpoints']['initial_sft768']['weight_identity'],
            'Cached source came from a different actual initial model')
    prior=signed['payload']
    require(signed['sha256']==audit['run_signature_sha256'] and prior['initial_identity']==config['initial_identity']
            and prior['input']==config['input'] and prior['reference_spec_sha256']==hash_json(config)
            and prior['reference_policy_sha256']==done['cache_sha256']
            and prior['manifest_sha256']==sha256_file(confined(cache['source_manifest'])),
            'Reference signature/audit/spec/input/cache/manifest must bind the same actual source run')
    require(config['input']==p['input'] and config['revision']==p['revision']
            and config['versions']==p['required_versions'],'Cached source preprocessing/revision/environment differs')
    actual_pixels={x:sha256_file(image_path(p['image_root'],x)) for x in sorted({x for r in records for x in r['image_paths_display']})}
    require(all(config['pixel_sha256'][x]==h for x,h in actual_pixels.items()),'Cached source pixels changed')
    output.mkdir(parents=True)
    payload={'policy_sha256':sha256_file(policy_path),'phase':'cached','task_id':'initial_sft768__train256',
        'endpoint':'initial_sft768','bank':'train256','selected_records_sha256':hash_json(records),
        'manifest_sha256':p['banks']['train256']['manifest_sha256'],'input':p['input'],
        'weight_identity':config['initial_identity'],'model_weights_sha256':hash_json(config['initial_identity']),
        'sources_sha256':p['sources_sha256'],'pixel_sha256':actual_pixels,
        'first_preflight_started_utc':p['first_preflight_started_utc'],'absolute_deadline_utc':p['absolute_deadline_utc'],
        'zero_updates':True,'source_receipt_sha256':cache['source_receipt_sha256'],'new_model_calls':0,
        'versions':config['versions'],'versions_source':'sealed_source_cache'}
    signature=hash_json(payload);write_json(output/'signature.json',{'sha256':signature,'payload':payload})
    lineage=[]
    for i,(record,row) in enumerate(zip(records,source)):
        require(row['index']==i and all(row[k]==record[k] for k in ('qid','recipe_id','presentation_id','source_split','display_perm','target_order','target_labels')),
                'Cached source identity/order differs')
        require(tuple(c['text'] for c in row['candidates'])==CANDIDATES,'Cached source candidate grammar differs')
        result={k:record[k] for k in ('qid','recipe_id','presentation_id','source_split','manifest_set','display_perm','target_order','target_labels')}
        result.update(index=i,bank='train256',endpoint='initial_sft768',configuration_sha256=signature,
            model_weights_sha256=payload['model_weights_sha256'],input_audit=row['input_audit'],
            pixel_sha256={x:payload['pixel_sha256'][x] for x in record['image_paths_display']},
            candidates=row['candidates'],token_logps=row['token_logps'],numerical_checks=row['numerical_checks'],
            candidate_forwards=0,optimizer_steps=0,backward_calls=0,generation_calls=0,elapsed_seconds=0,
            completed_utc=now(),source_index=i,source_row_sha256=hash_json(row),source_payload=row)
        append(output/'scores.jsonl',result)
        lineage.append({'index':i,'qid':record['qid'],'presentation_id':record['presentation_id'],
                        'record_sha256':hash_json(record),'source_row_sha256':hash_json(row),
                        'output_row_sha256':hash_json(result)})
    write_json(output/'lineage.json',{'status':'identity_only_original_first256_cache_reuse',
        'rows':256,'source_receipt_sha256':cache['source_receipt_sha256'],'rows_lineage':lineage,
        'source_candidate_forwards_historical_whole1024':25600,'new_candidate_forwards':0})
    write_json(output/'completion.json',{'status':'complete_reused_initial_train_scores','signature_sha256':signature,
        'rows':256,'expected_rows':256,'output_sha256':sha256_file(output/'scores.jsonl'),
        'lineage_sha256':sha256_file(output/'lineage.json'),'optimizer_steps':0,'backward_calls':0,
        'generation_calls':0,'candidate_forwards':0,'new_model_calls':0,'runtime_model_reloaded':False,'model_was_loaded':False,
        'runtime_lora_before':None,'runtime_lora_after':None,'runtime_visual_before':None,'runtime_visual_after':None,
        'weight_identity_before':config['initial_identity'],'weight_identity_after':config['initial_identity'],
        'source_audit_sha256':sha256_file(confined(cache['source_audit'])),'started_utc':started_utc,
        'elapsed_seconds':time.monotonic()-started,'completed_utc':now()})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--task-id',required=True);parser.add_argument('--phase',choices=('pilot','formal'),required=True)
    args=parser.parse_args();policy_path=args.policy.resolve();p=load(policy_path)
    validate_policy(p);validate_environment();check_frozen_sources(p)
    require(os.environ.get('RECIPEQA_ENDPOINT_BANK_WATCHDOG')==sha256_file(policy_path)
            and os.environ.get('RECIPEQA_ENDPOINT_BANK_PGID')==str(os.getpgrp()),'Policy-bound native same-group watchdog required')
    task,count=task_spec(p,args.task_id,args.phase);records=bank_records(p,task['bank'])[:count]
    authorization=validate_authorization(p,policy_path,args.task_id,args.phase)
    actual_versions=package_versions();require(actual_versions==p['required_versions'],'Runtime libraries differ from the cached reference environment')
    out=confined(p['output_directory'])/args.phase/task['id'];require(not out.exists(),'Fresh scoring only; no resume or overwrite')
    out.mkdir(parents=True);started=time.monotonic();started_utc=now();remaining_seconds(p)
    endpoint=p['endpoints'][task['endpoint']]
    identity={'base':model_weight_identity(confined(p['model_path'])),
              'adapter':model_weight_identity(confined(endpoint['adapter_path']),adapter=True)}
    require(identity==endpoint['weight_identity'],'Actual endpoint model bytes differ from frozen endpoint')
    pixels={x:sha256_file(image_path(p['image_root'],x)) for x in sorted({x for r in records for x in r['image_paths_display']})}
    payload={'policy_sha256':sha256_file(policy_path),'phase':args.phase,'task_id':task['id'],
        'endpoint':task['endpoint'],'bank':task['bank'],'selected_records_sha256':hash_json(records),
        'manifest_sha256':p['banks'][task['bank']]['manifest_sha256'],'input':p['input'],'weight_identity':identity,
        'model_weights_sha256':hash_json(identity),'sources_sha256':p['sources_sha256'],'pixel_sha256':pixels,
        'first_preflight_started_utc':p['first_preflight_started_utc'],'absolute_deadline_utc':p['absolute_deadline_utc'],
        'zero_updates':True,'pid':os.getpid(),'pgid':os.getpgrp(),'versions':actual_versions,
        'versions_source':'actual_runtime_package_versions'}
    payload['authorization_sha256']=sha256_file(authorization)
    signature=hash_json(payload);write_json(out/'signature.json',{'sha256':signature,'payload':payload})
    config=InputConfig(**p['input']);processor=load_processor(str(confined(p['model_path'])),p['revision'],config)
    model=load_model(str(confined(p['model_path'])),p['revision'],adapter_path=str(confined(endpoint['adapter_path'])),trainable=False)
    import torch
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.eval();require(not any(m.training for m in model.modules()) and not any(x.requires_grad for x in model.parameters()),'Fully frozen eval model required')
    before_lora=lora_fingerprint(model);before_visual=frozen_vision_fingerprint(model)
    score_args=SimpleNamespace(microbatch=1,batch_atol=.01,tie_atol=1e-8)
    torch.cuda.reset_peak_memory_stats()
    for index,record in enumerate(records):
        remaining_seconds(p);row_start=time.monotonic()
        with torch.inference_mode():
            prepared,audit=prepare_record(record,processor,confined(p['image_root']),config)
            candidates=candidate_suffixes(processor,prepared['prompt'])
            require(tuple(c['text'] for c in candidates)==CANDIDATES and all(c['token_count']==5 for c in candidates),'Frozen full24 plusEOS C5 grammar required')
            encoded=encode_messages(processor,prepared['prompt'],load_images(record,confined(p['image_root']))).to(model.device)
            token_logps,checks=score_candidates(model,encoded,candidates,score_args,literal_check=index==0)
            torch.cuda.synchronize()
        require(all(len(x)==5 and all(math.isfinite(y) and y<=1e-6 for y in x) for x in token_logps),'Bad complete token likelihoods')
        row={k:record[k] for k in ('qid','recipe_id','presentation_id','source_split','manifest_set','display_perm','target_order','target_labels')}
        row.update(index=index,bank=task['bank'],endpoint=task['endpoint'],configuration_sha256=signature,
            model_weights_sha256=payload['model_weights_sha256'],input_audit=audit,
            pixel_sha256={x:pixels[x] for x in record['image_paths_display']},candidates=candidates,
            token_logps=token_logps,numerical_checks=checks,candidate_forwards=25,optimizer_steps=0,
            backward_calls=0,generation_calls=0,elapsed_seconds=time.monotonic()-row_start,completed_utc=now())
        append(out/'scores.jsonl',row)
        print(json.dumps({'qid':record['qid'],'index':index,'task_id':task['id'],'phase':args.phase,
                          'rows':index+1,'seconds':row['elapsed_seconds']}),flush=True)
        del encoded
    remaining_seconds(p);after_lora=lora_fingerprint(model);after_visual=frozen_vision_fingerprint(model)
    after_identity={'base':model_weight_identity(confined(p['model_path'])),
                    'adapter':model_weight_identity(confined(endpoint['adapter_path']),adapter=True)}
    require(before_lora==after_lora and before_visual==after_visual and identity==after_identity,'A read-only score changed model weights')
    require(not any(x.requires_grad for x in model.parameters()) and not any(m.training for m in model.modules()),'Eval/freeze mode changed')
    check_frozen_sources(p);require(sha256_file(policy_path)==payload['policy_sha256'],'Locked policy changed')
    write_json(out/'completion.json',{'status':'complete_zero_update_endpoint_scores','signature_sha256':signature,
        'rows':count,'expected_rows':count,'output_sha256':sha256_file(out/'scores.jsonl'),'optimizer_steps':0,
        'backward_calls':0,'generation_calls':0,'candidate_forwards':25*count,'weight_identity_before':identity,
        'weight_identity_after':after_identity,'runtime_lora_before':before_lora,'runtime_lora_after':after_lora,
        'runtime_visual_before':before_visual,'runtime_visual_after':after_visual,'all_modules_eval':True,
        'all_parameters_frozen':True,'trainable_parameters_numel':0,'model_was_loaded':True,
        'started_utc':started_utc,'elapsed_seconds':time.monotonic()-started,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'completed_utc':now()})


if __name__=='__main__':main()
