#!/usr/bin/env python3
"""Independent CPU/physical-artifact verifier for the new coverage experiment.

Explicitly derives the old complete-run audit; imports only its immutable CPU
math, safe-tensor fingerprint and ZIP-CRC helpers, never any trainer/contract.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'exploratory/principled_ordering_20261105'))
import verify_finite_run as legacy
from recipeqa.modeling import InputConfig,hash_json,model_weight_identity,package_versions,read_jsonl,sha256_file,write_json

ACTIONS=legacy.ACTIONS
STATE_NAMES=legacy.STATE_NAMES
require=legacy.require
close=legacy.close
vector_close=legacy.vector_close
load=legacy.load
data_path=legacy.data_path
recompute_policy=legacy.recompute_policy
independent_terms=legacy.independent_terms
check_policy_fields=legacy.check_policy_fields
exact_labels=legacy.exact_labels
audit_numerical_checks=legacy.audit_numerical_checks
tensor_fingerprint=legacy.tensor_fingerprint
audit_state_file=legacy.audit_state_file
audit_effective_update=legacy.audit_effective_update
LEGACY_SHA=legacy.SOURCE_SHA
OLD_VERIFIER_SHA='c2408abd93fb9b322262213a65b9d311abe88ef4bc7550aec1d164626f73a791'
BETA=1.6363096678310884
WALL_SECONDS=57600
FULL_TRAIN='exploratory/principled_ordering_20261105/full_train_data/train_all7558.jsonl'
FULL_SHA='b9979ee0612679d2e9460dd3bb8a6f68e7dea6794bc6cdcc3e1defdc87d8d949'
INITIAL_ADAPTER='exploratory/principled_ordering_20261105/full_sft/train/checkpoint-768'
REVISION='89644892e4d85e24eaac8bacfd4f463576704203'
SCORING_PATHS=tuple(k for k in LEGACY_SHA if not k.endswith('finite_policy_train.py'))
REQUIRED_SOURCE_PATHS=(
 'research/coverage_20261106/coverage_train.py','research/coverage_20261106/coverage_contract.py',
 'research/coverage_20261106/coverage_pipeline.py','research/coverage_20261106/build_draft_policy.py',
 'research/coverage_20261106/launch_stage.py','research/coverage_20261106/run_cpu_preflight.py',
 'research/coverage_20261106/test_coverage_contract.py','research/coverage_20261106/test_coverage_pipeline.py',
 'research/coverage_20261106/verify_coverage_run.py','research/coverage_20261106/test_verify_coverage_run.py',
 'research/grounding_equivariance_20261106/orbit_diagnostic.py','scripts/predict.py','scripts/validate_run.py',
 'exploratory/principled_ordering_20261105/finite_policy_train.py',
 'exploratory/principled_ordering_20261105/verify_finite_run.py',
 'exploratory/principled_ordering_20261105/test_finite_policy_tensor_reference.py',
 'exploratory/principled_ordering_20261105/score_permutations.py',
 'exploratory/principled_ordering_20261105/gradient_alignment.py',
 'src/recipeqa/modeling.py','src/recipeqa/metrics.py')


def stamp(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(result.tzinfo is not None,'UTC-aware scientific clock required')
    return result


def identity(row):return str(row['qid']),str(row['presentation_id'])


def audit_child(policy):
    """Independent fixed parameters; never imports the producer contract."""
    require(policy['status']=='locked_before_coverage_training','Draft child cannot be verified as an authorized run')
    mode=policy['mode'];cli=policy['cli'];method=policy['method'];flow=policy['flow']
    require(mode in ('reference','smoke','formal') and method in ('raw_sft','exact_kl') and flow in ('flow44','flow45'),'Unexpected child mode/method/flow')
    count,steps,save=(4,1,1) if mode=='smoke' else (1024,256,64)
    require(cli['method']==method and cli['seed']==int(flow[-2:]) and cli['limit']==count
        and cli['max_steps']==steps and cli['save_steps']==save,'Fixed flow/exposure/update/checkpoint budget differs')
    require(cli['learning_rate']==5e-6 and cli['beta']==(BETA if method=='exact_kl' else 0.),'Fixed learning-rate/beta differs')
    require(cli['prepare_reference_only']==(mode=='reference') and (mode!='reference' or (method=='raw_sft' and flow=='flow44')),'Reference-only method/flow/update mode differs')
    require(cli['reference_cache'] is None if mode=='reference' else bool(cli['reference_cache']),'Only reference source may build a cache')
    require(cli['revision']==REVISION and cli['logp_atol']==.01 and cli['min_pixels']==65536 and cli['max_pixels']==262144
        and cli['recipe_token_budget']==1536 and cli['max_input_tokens']==4096 and 0<cli['wall_seconds']<=WALL_SECONDS,'Frozen input/scoring/runtime contract differs')
    require(set(policy['sources_sha256'])==set(REQUIRED_SOURCE_PATHS),'Reviewed full source closure differs')
    require(all(policy['sources_sha256'].get(k)==v for k,v in LEGACY_SHA.items()),'An old frozen scientific source changed')
    require(policy['sources_sha256']['exploratory/principled_ordering_20261105/verify_finite_run.py']==OLD_VERIFIER_SHA,'Old independent verifier dependency changed')
    require(policy['reward']=='unique_complete_order_exact_0_1' and policy['selection']=='fixed256_primary_fixed64_diagnostic_no_dev_selection','Reward/endpoint-selection policy differs')
    require(policy['optimizer']=={'accumulation':4,'clip_norm':1.,'canonical_C_including_eos':5,'fresh_adamw':True},'Fresh-state/normalization/clip contract differs')
    require(policy['batch_clock_started_utc']==policy['first_preflight_started_utc'],'First scientific clock alias was reset')
    start=stamp(policy['batch_clock_started_utc']);end=stamp(policy['absolute_deadline_utc'])
    require(0<(end-start).total_seconds()<=WALL_SECONDS,'Shared original16h deadline changed')
    require(policy['initial_identity']['adapter']['directory']==str(ROOT/INITIAL_ADAPTER)
        and Path(cli['adapter_path'])==ROOT/INITIAL_ADAPTER,'Initial SFT768 adapter differs')
    require(policy['reference_lineage'] is None if mode=='reference' or (mode=='formal' and flow=='flow44') else bool(policy['reference_lineage']),'Required reference lineage missing/unexpected')
    return count,steps,save


def audit_pool(policy,records):
    spec=policy['pool_spec'];full_path=data_path(ROOT/spec['full_train_path']);prior_path=data_path(ROOT/spec['prior_visits_path'])
    require(spec['full_train_path']==FULL_TRAIN and spec['full_train_sha256']==FULL_SHA==sha256_file(full_path)
        and spec['full_train_count']==7558 and spec['pool_count']==1024 and spec['pool_source_indices']==list(range(1024)), 'Original first1024 pool specification differs')
    full=read_jsonl(full_path);require(len(full)==7558,'Original complete7558 source missing')
    pool=full[:1024];require(len({identity(r) for r in pool})==1024,'Pool identities repeated')
    require(hash_json(pool)==spec['pool_selected_records_sha256'] and spec['flow44_order']=='original' and spec['flow45_shuffle_seed']==45,'Pool content/order specification differs')
    ordered=list(pool)
    if policy['flow']=='flow45':random.Random(45).shuffle(ordered)
    expected=ordered[:4] if policy['mode']=='smoke' else ordered
    require(records==expected,'Actual records are not unchanged original first1024 / fixed45 order / smoke prefix')
    require(all(r['source_split']=='train' and r['technical_status']=='ok' and r['manifest_set'] not in ('val','test','confirmation','orbit','process') for r in pool),'Heldout/damaged pool input')
    prior_signature_path=data_path(ROOT/spec['prior_signature_path']);prior_signed=load(prior_signature_path)
    require(sha256_file(prior_signature_path)==spec['prior_signature_sha256']
        and prior_signed['sha256']==spec['prior_run_signature_sha256']==hash_json(prior_signed['payload'])
        and prior_signed['payload']['manifest_sha256']==FULL_SHA,'Initial SFT actual source signature differs')
    prior=read_jsonl(prior_path);require(sha256_file(prior_path)==spec['prior_visits_sha256'],'Prior actual-consumption evidence SHA differs')
    consumed=[r for r in prior if r['global_step_before_update']<768]
    require(spec['initial_sft_step']==768 and spec['initial_consumed_visits']==len(consumed)==len({identity(r) for r in consumed})==6144,'Initial SFT768 consumed6144 evidence differs')
    require(all(type(r['global_step_before_update']) is int and 0<=r['global_step_before_update']<768
        and r['run_signature_sha256']==prior_signed['sha256'] and r['consumption_point']=='training_step_forward_backward_returned' for r in consumed)
        and {r['global_step_before_update'] for r in consumed}==set(range(768))
        and all(sum(r['global_step_before_update']==i for r in consumed)==8 for i in range(768)),'Prior exposure is not eight actually completed visits per initial step')
    require({identity(r) for r in pool}<={identity(r) for r in consumed},'Some pool records were not actually consumed by selected initial SFT768')
    receipt=load(data_path(ROOT/policy['pool_receipt']))
    require(receipt['pool_source_indices']==list(range(1024)) and receipt['pool_selected_records_sha256']==hash_json(pool),'Pool preparation receipt differs')
    for flow in ('flow44','flow45'):
        order=list(pool)
        if flow=='flow45':random.Random(45).shuffle(order)
        require(receipt['flow_orders'][flow]==[{'qid':r['qid'],'presentation_id':r['presentation_id'],'record_sha256':hash_json(r)} for r in order],'Pool order receipt differs')
    return pool,[full_path,prior_path,prior_signature_path,data_path(ROOT/policy['pool_receipt'])]


def audit_lineage_rows(source,source_spec,source_records,target,target_spec,target_records,lineage):
    """Independent reconstruction; no producer reindex function is imported."""
    require(len(source)==len(source_records)==1024 and len(target)==len(target_records) in (4,1024),'Reference lineage source/target count differs')
    bank={identity(r):(i,r,source[i]) for i,r in enumerate(source_records)}
    require(len(bank)==1024 and len({identity(r) for r in target_records})==len(target_records),'Lineage duplicates an identity')
    spec=dict(source_spec);spec.update(manifest_sha256=lineage['target_manifest_sha256'],selected_records_sha256=hash_json(target_records),
        pixel_sha256={p:source_spec['pixel_sha256'][p] for p in sorted({p for r in target_records for p in r['image_paths_display']})})
    require(target_spec==spec,'Derived reference spec changed more than manifest/selected-record/pixel identities')
    source_spec_sha=hash_json(source_spec);target_spec_sha=hash_json(target_spec)
    require(lineage['source_spec_sha256']==source_spec_sha and lineage['target_spec_sha256']==target_spec_sha
        and lineage['source_selected_records_sha256']==hash_json(source_records) and lineage['target_selected_records_sha256']==hash_json(target_records), 'Lineage selected-record/spec identity differs')
    require(lineage['source_rows']==1024 and lineage['target_rows']==len(target)
        and lineage['allowed_row_changes']==['index','reference_spec_sha256']
        and lineage['allowed_spec_changes']==['manifest_sha256','pixel_sha256','selected_records_sha256'],'Lineage permitted edits/counts changed')
    require(len(lineage['rows'])==len(target),'Per-row lineage incomplete')
    origins=[]
    for index,(record,actual,entry) in enumerate(zip(target_records,target,lineage['rows'])):
        require(identity(record) in bank,'Target absent from original1024 reference pool')
        origin,original,raw=bank[identity(record)];require(record==original and raw['index']==origin,'Reference transform changed full source record/index')
        require(identity(raw)==identity(record) and raw['reference_spec_sha256']==source_spec_sha,'Original reference identity/spec differs')
        expected={**raw,'index':index,'reference_spec_sha256':target_spec_sha}
        require(actual==expected,'Derived cache changed scores/candidates/checks/input audit or other payload')
        unchanged={k:v for k,v in raw.items() if k not in ('index','reference_spec_sha256')}
        require(entry=={'index':index,'source_index':origin,'qid':record['qid'],'presentation_id':record['presentation_id'],
            'record_sha256':hash_json(record),'source_row_sha256':hash_json(raw),'target_row_sha256':hash_json(expected),
            'unchanged_payload_sha256':hash_json(unchanged),'reference_literal_check_was_source_index_zero':origin==0},'Per-row lineage evidence differs')
        origins.append(origin)
    return origins


def audit_reference_lineage(policy,ref_path,references,spec,records):
    if policy['reference_lineage'] is None:
        require(policy['flow']=='flow44' and policy['mode']!='smoke' and len(references)==1024,'Only original full flow44 may omit lineage')
        return list(range(1024)),[]
    line_path=data_path(ROOT/policy['reference_lineage']);line=load(line_path)
    source=data_path(line['source_cache']);source_manifest=data_path(line['source_manifest']);target_manifest=data_path(line['target_manifest'])
    source_cfg=Path(str(source)+'.config.json');source_complete=Path(str(source)+'.completion.json')
    require(source!=ref_path and source.is_relative_to(ROOT/'exploratory/coverage_20261106')
        and data_path(line['target_cache'])==ref_path and source_manifest==data_path(ROOT/policy['reference_source_manifest'])
        and target_manifest==data_path(policy['cli']['manifest']),'Reference lineage source/target paths differ')
    require(sha256_file(source)==line['source_cache_sha256'] and sha256_file(source_cfg)==line['source_config_sha256']
        and sha256_file(source_complete)==line['source_completion_sha256'] and sha256_file(ref_path)==line['target_cache_sha256']
        and sha256_file(source_manifest)==line['source_manifest_sha256'] and sha256_file(target_manifest)==line['target_manifest_sha256'],'Reference lineage physical-file binding differs')
    done=load(source_complete);source_spec=load(source_cfg);original=read_jsonl(source);source_records=read_jsonl(source_manifest)
    require(done['status']=='complete_fixed_initial_reference' and done['rows']==1024 and done['cache_sha256']==sha256_file(source)
        and done['reference_spec_sha256']==hash_json(source_spec),'Actual source reference completion failed')
    origins=audit_lineage_rows(original,source_spec,source_records,references,spec,records,line)
    return origins,[line_path,source,source_cfg,source_complete,source_manifest,target_manifest]


def resolve_args(parsed):
    policy_path=data_path(parsed.policy);policy=load(policy_path);count,steps,save=audit_child(policy);cli=policy['cli']
    require((policy['mode']=='reference')==parsed.reference_only,'Reference-only audit mode must be explicit and match child')
    return SimpleNamespace(run_dir=parsed.run_dir,output=parsed.output,policy_path=policy_path,policy=policy,
        manifest=cli['manifest'],expected_manifest_sha256=cli['expected_manifest_sha256'],initial_adapter=cli['adapter_path'],
        method=policy['method'],seed=cli['seed'],limit=count,steps=steps,beta=cli['beta'],save_steps=save)


def audit_generated_reference_receipt(policy,payload,source_ref):
    receipt_path=data_path(ROOT/policy['generated_reference_receipt']);receipt=load(receipt_path)
    require(payload['generated_reference_receipt_sha256']==sha256_file(receipt_path),'Signed sealed reference receipt changed')
    require(receipt['status']=='sealed_before_first_smoke_update' and receipt['shared_initial_reference_model_forwards_charged_once']==25600
        and data_path(receipt['source_creation_run'])==source_ref.parent,'Reference creation/seal accounting differs')
    require(isinstance(receipt['stage_policy_sha256'],str) and len(receipt['stage_policy_sha256'])==64,'Missing parent stage policy binding')
    required=[receipt_path]
    for filename,value in receipt['files'].items():
        p=data_path(filename);require(p.is_relative_to(source_ref.parent.parent) and sha256_file(p)==value,'Sealed generated reference file changed')
        required.append(p)
    expected={str(source_ref)+suffix for suffix in ('','.config.json','.completion.json')}
    if policy['cli']['reference_cache']!=str(source_ref):
        expected|={policy['cli']['reference_cache']+suffix for suffix in ('','.config.json','.completion.json','.lineage.json')}
    require(expected<=set(receipt['files']) and len(receipt['files'])==15,'Shared generated cache/sidecar/lineage seal inventory differs')
    return required,sha256_file(receipt_path),receipt['stage_policy_sha256']


def verify_reference(args):
    """Actual newly computed1024 reference, no optimizer/updates, physical audit."""
    policy=args.policy;cli=policy['cli'];root=data_path(args.run_dir);manifest=data_path(args.manifest)
    require(policy['mode']=='reference' and root==data_path(cli['output_dir']) and policy['reference_lineage'] is None,'Reference audit mode/output differs')
    require(ROOT.stat().st_dev!=Path('/').stat().st_dev,'Reference is not on separate data disk')
    require(sha256_file(manifest)==args.expected_manifest_sha256,'Reference manifest SHA differs')
    for group in ('sources_sha256','artifacts_sha256'):
        for name,value in policy[group].items():require(sha256_file(data_path(ROOT/name))==value,'Reference source/artifact changed: '+name)
    state=load(root/'state.json');outcome=load(root/'reference_only_outcome.json');signed=load(root/'run_signature.json');payload=signed['payload']
    require(state['status']==outcome['status']=='completed_reference_only' and state['outcome']==outcome
        and state['global_step']==state['actual_prompt_visits']==outcome['updates']==0
        and outcome['optimizer_constructed'] is False and outcome['initial_weights_unchanged'] is True,'Reference secretly incomplete or updated/constructed optimizer')
    require(state['test_access'] is False and state['no_resume'] is True,'Reference scope differs')
    require(stamp(policy['batch_clock_started_utc'])<=stamp(state['started_utc'])<=stamp(state['finished_utc'])<=stamp(state['updated_utc'])<=stamp(policy['absolute_deadline_utc']),'Reference timestamps violate original common deadline')
    require(signed['sha256']==hash_json(payload)==outcome['run_signature_sha256'],'Reference signed run differs')
    require(payload['cli']=={**cli,'policy':str(args.policy_path)} and payload['method']=='raw_sft' and payload['coverage_mode']=='reference'
        and payload['resolved_beta']==0 and payload['actual_updates']==256 and payload['accumulation']==4
        and payload['canonical_completion_token_count']==5 and payload['locked_policy_sha256']==sha256_file(args.policy_path)
        and payload['reference_lineage_sha256'] is None and payload['generated_reference_receipt_sha256'] is None,'Reference signed configuration differs')
    require(payload['batch_clock_started_utc']==policy['batch_clock_started_utc'] and payload['absolute_deadline_utc']==policy['absolute_deadline_utc']
        and payload['sources_sha256']==policy['sources_sha256'] and payload['versions']==package_versions(),'Reference clocks/sources/versions differ')
    records=read_jsonl(manifest);pool,pool_files=audit_pool(policy,records)
    require(len(records)==outcome['rows']==1024 and payload['selected_records_sha256']==hash_json(records)
        and payload['manifest_sha256']==args.expected_manifest_sha256,'Reference selected records differ')
    initial=data_path(args.initial_adapter)
    identities={'base':model_weight_identity(data_path(cli['model_path'])),'adapter':model_weight_identity(initial,adapter=True)}
    require(identities==policy['initial_identity']==payload['initial_identity'],'Reference actual initial weight bytes differ')
    inventory_record=load(root/'trainable_parameters.json');inventory=inventory_record['parameters']
    require(len(inventory)==len({r['name'] for r in inventory})==392
        and inventory_record['trainable_parameters']==sum(r['numel'] for r in inventory)==17432576,'Reference LoRA inventory differs')
    for item in inventory:require('lora_' in item['name'] and '.language_model.layers.' in item['name'] and math.prod(item['shape'])==item['numel'],'Reference contains unapproved parameter')
    fingerprint=tensor_fingerprint(initial,inventory);require(fingerprint==payload['initial_adapter_fingerprint'],'Actual reference initial tensor fingerprint differs')
    require(isinstance(payload['initial_visual_fingerprint'],str) and len(payload['initial_visual_fingerprint'])==64,'Reference runtime visual guard missing')
    image_root=data_path(cli['image_root']);pixels={}
    for r in records:
        for relative in r['image_paths_display']:
            p=data_path(image_root/relative);require(p.is_relative_to(image_root),'Reference image escapes image root');pixels[relative]=sha256_file(p)
    require(pixels==payload['pixel_sha256'] and payload['input']==asdict(InputConfig()),'Reference actual pixels/input differ')
    ref=root/'reference_policy.jsonl';cfg=Path(str(ref)+'.config.json');done_path=Path(str(ref)+'.completion.json')
    spec=load(cfg);done=load(done_path);refs=read_jsonl(ref);ref_sha=sha256_file(ref)
    expected={'schema_version':1,'policy':'canonical24_full_sequence_plusEOS_temperature1','initial_identity':identities,
        'input':payload['input'],'revision':cli['revision'],'manifest_sha256':args.expected_manifest_sha256,
        'selected_records_sha256':hash_json(records),'candidate_texts':list(ACTIONS),'pixel_sha256':pixels,
        'numerical_reference_atol':.01,'scoring_source_sha256':{p:policy['sources_sha256'][p] for p in SCORING_PATHS},'versions':payload['versions']}
    require(spec==expected and done['status']=='complete_fixed_initial_reference' and done['rows']==len(refs)==1024
        and done['cache_sha256']==payload['reference_policy_sha256']==outcome['reference_policy_sha256']==ref_sha
        and done['reference_spec_sha256']==payload['reference_spec_sha256']==outcome['reference_spec_sha256']==hash_json(spec),'Reference spec/completion differs')
    require(data_path(outcome['reference_cache'])==ref,'Reference outcome points outside actual source cache')
    for index,(record,row) in enumerate(zip(records,refs)):
        require(row['index']==index and identity(row)==identity(record) and row['recipe_id']==record['recipe_id']
            and row['source_split']=='train' and row['reference_spec_sha256']==hash_json(spec),'Reference source/index/order differs')
        for key in ('display_perm','target_order','target_labels'):require(row[key]==record[key],'Reference source labels differ')
        computed=recompute_policy(row['token_logps'],row['candidates']);check_policy_fields(row,computed)
        require(computed['token_count']==5 and row['exact_rewards']==exact_labels(record)
            and row['gold_candidate_index']==exact_labels(record).index(1),'Reference EOS/C5/Exact reward differs')
        audit_numerical_checks(row['numerical_checks'],index==0,.01,row['token_logps'])
        audit=row['input_audit'];require(identity(audit)==identity(record) and audit['recipe_id']==record['recipe_id']
            and audit['input_tokens']<=4096 and audit['recipe_tokens_used']<=1536
            and audit['titles_retained']==audit['steps']==len(record['steps']),'Reference actual input audit incomplete')
    require(outcome['reference_candidate_forwards_charged_this_run']==25*1024,'One actual1024 source reference forward count differs')
    require(math.isfinite(outcome['elapsed_seconds']) and 0<outcome['elapsed_seconds']
        <=(stamp(state['finished_utc'])-stamp(state['started_utc'])).total_seconds()+1.,'Reference elapsed time contradicts persisted UTC interval')
    require(not any((root/name).exists() for name in ('actual_prompt_visits.jsonl','step_times.jsonl','train_outcome.json','adapter_final'))
        and not list(root.glob('checkpoint-*')),'Reference-only directory contains undisclosed training/update artifacts')
    required=[args.policy_path,root/'state.json',root/'reference_only_outcome.json',root/'run_signature.json',root/'trainable_parameters.json',ref,cfg,done_path,*pool_files]
    required.extend(data_path(ROOT/p) for p in policy['sources_sha256'])
    return {'status':'complete_verified_coverage_reference_only','passed':True,'updates':0,'optimizer_constructed':False,
        'rows':1024,'reference_candidate_forwards_charged_this_run':25600,'reference_policy_sha256':ref_sha,
        'run_signature_sha256':signed['sha256'],'locked_policy_sha256':sha256_file(args.policy_path),
        'initial_adapter_fingerprint':fingerprint,'required_artifacts':{str(p.relative_to(ROOT)):{'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in required},
        'limits':['Reference-only signature records the planned max256 updates; actual committed outcome has zero updates and no optimizer.',
            'No VLM forwards are repeated. Persisted complete scores/inputs and actual initial adapter tensor bytes are audited.',
            'Unchanged in-memory weights during reference construction rely on reviewed runtime guards; no post-reference model tensors are exported.']}


def verify(args):
    policy=args.policy;bound_sources=policy['sources_sha256']
    root, manifest, initial, output = map(data_path, (args.run_dir, args.manifest, args.initial_adapter, args.output))
    require(ROOT.stat().st_dev != Path("/").stat().st_dev, "Project is not on a separately mounted data disk")
    require((args.limit,args.steps,args.save_steps)==((4,1,1) if policy['mode']=='smoke' else (1024,256,64)), "Only fixed4/1 smoke or1024/256/save64 formal is authorized")
    require(math.isfinite(args.beta) and (args.beta > 0 if args.method.endswith("_kl") else args.beta == 0), "Expected method/beta mismatch")
    require(sha256_file(manifest) == args.expected_manifest_sha256, "Expected manifest bytes differ")
    require(not output.exists(), "Verifier output exists; preserve prior audit")
    signature_file = root / "run_signature.json"
    signed = load(signature_file)
    signature, payload = signed["sha256"], signed["payload"]
    require(signature == hash_json(payload), "Run signature is internally inconsistent")
    cli = payload["cli"]
    require(cli=={**policy['cli'],'policy':str(args.policy_path)},'Actual CLI differs from frozen child policy')
    require(payload['locked_policy_sha256']==sha256_file(args.policy_path) and payload['coverage_mode']==policy['mode']
        and payload['batch_clock_started_utc']==policy['batch_clock_started_utc']
        and payload['absolute_deadline_utc']==policy['absolute_deadline_utc'],'Signed child/clock lineage differs')
    require(payload["method"] == cli["method"] == args.method and cli["seed"] == args.seed
            and cli["limit"] == args.limit and payload["actual_updates"] == args.steps
            and payload["accumulation"] == 4 and payload["resolved_beta"] == args.beta, "Expected method/seed/budget/beta differs")
    require(cli["max_steps"] in (None, args.steps) and not cli["prepare_reference_only"], "Reference-only or truncated run is not training completion")
    require(cli["save_steps"] == args.save_steps and cli["logp_atol"] == .01,
            "Save interval or numerical tolerance differs from reviewed smoke/trial")
    require((cli["beta"] is None or cli["beta"] == 0) if args.beta == 0 else cli["beta"] == args.beta, "CLI KL coefficient differs")
    require(data_path(cli["manifest"]) == manifest and data_path(cli["adapter_path"]) == initial
            and data_path(cli["output_dir"]) == root, "CLI source/run paths differ")
    require(payload["manifest_sha256"] == cli["expected_manifest_sha256"] == args.expected_manifest_sha256, "Signed manifest differs")
    require(payload["sources_sha256"] == bound_sources, "Run used unreviewed scientific sources")
    require(payload["input"] == asdict(InputConfig()), "Finite run changed frozen image/recipe/completion budgets")
    require(payload["optimizer"] == {"name": "AdamW", "lr": cli["learning_rate"], "betas": [.9, .999],
            "eps": 1e-8, "weight_decay": 0.0, "max_grad_norm": 1.0, "schedule": "constant", "fresh_state": True}
            and cli["learning_rate"]==5e-6, "Optimizer metadata differs from reviewed fresh-state objective")
    for name, digest in bound_sources.items():
        require(sha256_file(data_path(ROOT / name)) == digest, "Current scientific source differs: " + name)
    for name,digest in policy['artifacts_sha256'].items():require(sha256_file(data_path(ROOT/name))==digest,'Bound data artifact differs: '+name)
    require(payload["versions"] == package_versions(), "Audit environment differs from recorded runtime versions")
    state, outcome = load(root / "state.json"), load(root / "train_outcome.json")
    require(state["status"] == outcome["status"] == "completed" and state["outcome"] == outcome, "Run/state did not complete consistently")
    require(outcome["method"] == args.method and outcome["global_step"] == state["global_step"] == args.steps
            and outcome["actual_prompt_visits"] == state["actual_prompt_visits"] == args.limit
            and outcome["run_signature_sha256"] == state["run_signature_sha256"] == signature,
            "Completed run identity or coverage differs")
    require(outcome["actual_visit_sequence_verified"] is True and outcome["resume_supported"] is False
            and outcome["test_access"] is False and state["test_access"] is False and state["no_resume"] is True,
            "Completion scope changed")
    require(stamp(policy['batch_clock_started_utc'])<=stamp(state['started_utc'])<=stamp(state['updated_utc'])
        <=stamp(policy['absolute_deadline_utc']),'Training timestamps escaped original shared deadline')
    require(stamp(state['finished_utc'])<=stamp(state['updated_utc']),'Training finish/state timestamps differ')
    records = read_jsonl(manifest)[:args.limit]
    require(len(records) == args.limit and len({(r["qid"], r["presentation_id"]) for r in records}) == args.limit,
            "Selected training prefix is incomplete or duplicated")
    require(all(row["source_split"] == "train" and row["technical_status"] == "ok"
                and row["manifest_set"] not in ("val", "test", "orbit", "process") for row in records), "Held-out/damaged input")
    require(hash_json(records) == payload["selected_records_sha256"], "Signed selected records differ")
    pool,pool_artifacts=audit_pool(policy,records)
    identities = {"base": model_weight_identity(data_path(cli["model_path"])),
                  "adapter": model_weight_identity(initial, adapter=True)}
    require(identities == payload["initial_identity"]==policy['initial_identity'], "Actual initial base/adapter bytes differ")
    image_root = data_path(cli["image_root"])
    pixels = {}
    for row in records:
        for relative in row["image_paths_display"]:
            path = data_path(image_root / relative)
            require(path.is_relative_to(image_root), "Image escaped declared image root")
            pixels[relative] = sha256_file(path)
    require(pixels == payload["pixel_sha256"], "Actual training image bytes differ")
    inventory_record = load(root / "trainable_parameters.json")
    inventory = inventory_record["parameters"]
    require(len(inventory) == len({row["name"] for row in inventory}) == 392
            and inventory_record["trainable_parameters"] == sum(row["numel"] for row in inventory) == 17432576,
            "LoRA tensor inventory/element count differs")
    for row in inventory:
        require("lora_" in row["name"] and ".language_model.layers." in row["name"]
                and math.prod(row["shape"]) == row["numel"], "Unapproved LoRA parameter or shape")
    initial_fingerprint = tensor_fingerprint(initial, inventory)
    require(initial_fingerprint == payload["initial_adapter_fingerprint"] == outcome["initial_adapter_fingerprint"],
            "Initial in-memory adapter fingerprint does not match actual safe tensor bytes")
    for field in ("initial_visual_fingerprint", "final_visual_fingerprint"):
        require(outcome[field] == payload["initial_visual_fingerprint"] and isinstance(outcome[field], str)
                and len(outcome[field]) == 64, "Runtime frozen-vision fingerprint changed or is missing")
    require(outcome["visual_weights_unchanged"] is True, "Runtime visual-freeze guard failed")
    ref_path = data_path(cli["reference_cache"]) if cli["reference_cache"] else root / "reference_policy.jsonl"
    ref_config_path, ref_completion_path = Path(str(ref_path) + ".config.json"), Path(str(ref_path) + ".completion.json")
    spec, ref_completion = load(ref_config_path), load(ref_completion_path)
    reference_hash = sha256_file(ref_path)
    require(ref_completion["status"] == "complete_fixed_initial_reference" and ref_completion["rows"] == args.limit
            and ref_completion["cache_sha256"] == reference_hash
            and ref_completion["reference_spec_sha256"] == payload["reference_spec_sha256"] == hash_json(spec),
            "Reference cache lacks complete, matching signed sidecars")
    expected_spec = {"schema_version": 1, "policy": "canonical24_full_sequence_plusEOS_temperature1",
                     "initial_identity": identities, "input": payload["input"], "revision": cli["revision"],
                     "manifest_sha256": args.expected_manifest_sha256, "selected_records_sha256": hash_json(records),
                     "candidate_texts": list(ACTIONS), "pixel_sha256": pixels, "numerical_reference_atol": .01,
                     "scoring_source_sha256": {name: bound_sources[name] for name in SCORING_PATHS},
                     "versions": payload["versions"]}
    require(spec == expected_spec and payload["reference_policy_sha256"] == outcome["reference_policy_sha256"]
            == state["reference_policy_sha256"] == reference_hash, "Fixed reference identity differs")
    references, visits, steps = (read_jsonl(path) for path in (ref_path, root / "actual_prompt_visits.jsonl", root / "step_times.jsonl"))
    require(len(references) == len(visits) == args.limit and len(steps) == args.steps, "Reference/visit/update coverage is incomplete")
    require(ref_path.is_relative_to(ROOT/'exploratory/coverage_20261106'),'Reference came from an old/stale experiment namespace')
    origin_indices,lineage_artifacts=audit_reference_lineage(policy,ref_path,references,spec,records)
    expected_lineage=sha256_file(data_path(ROOT/policy['reference_lineage'])) if policy['reference_lineage'] else None
    require(payload['reference_lineage_sha256']==expected_lineage,'Signed physical reference-lineage SHA differs')
    require(cli['reference_cache'] is not None,'A training arm must reuse the actual new shared reference, not secretly rebuild it')
    source_ref=data_path(load(data_path(ROOT/policy['reference_lineage']))['source_cache']) if policy['reference_lineage'] else ref_path
    source_signed=load(source_ref.parent/'run_signature.json')
    source_args=resolve_args(SimpleNamespace(policy=source_signed['payload']['cli']['policy'],run_dir=source_ref.parent,
        output=output,reference_only=True))
    source_audit=verify_reference(source_args)
    require(source_audit['passed'] is True and source_audit['reference_policy_sha256']==sha256_file(source_ref),'Shared new source reference failed actual independent audit')
    require(source_args.policy['batch_clock_started_utc']==policy['batch_clock_started_utc']
        and source_args.policy['absolute_deadline_utc']==policy['absolute_deadline_utc']
        and source_args.policy['initial_identity']==policy['initial_identity'],'Reference came from a different stage clock/initial model')
    generated_artifacts,generated_receipt_sha,parent_stage_sha=audit_generated_reference_receipt(policy,payload,source_ref)
    reference_forwards = 0 if cli["reference_cache"] else 25 * args.limit
    current_forwards = backwards = 0
    rng = random.Random(args.seed)
    active_by_step = [0] * args.steps
    counts_by_step = []
    constant_count = None
    for index, (record, ref, visit) in enumerate(zip(records, references, visits)):
        expected_identity = tuple(record[field] for field in ("qid", "recipe_id", "presentation_id"))
        require(tuple(ref[field] for field in ("qid", "recipe_id", "presentation_id")) == expected_identity
                == tuple(visit[field] for field in ("qid", "recipe_id", "presentation_id")), "Reference/visit order or identity differs")
        require(ref["index"] == visit["visit_index"] == index and visit["visit_id"] == record.get("visit_id")
                and visit["global_step_before_update"] == index // 4, "Visit/sample/update boundary differs")
        require(ref["source_split"] == visit["source_split"] == "train", "Held-out row in actual trajectory")
        for field in ("display_perm", "target_order", "target_labels"):
            require(record[field] == ref[field] == visit[field], "Actual verifier source mapping differs")
        require(ref["reference_spec_sha256"] == hash_json(spec) and visit["reference_policy_sha256"] == reference_hash
                and visit["run_signature_sha256"] == signature, "Trajectory reference/run identity differs")
        require(ref["input_audit"] == visit["input_audit"] and visit["input_hash"] == ref["input_audit"]["input_hash"],
                "Actual input differs from fixed reference")
        audit = ref["input_audit"]
        require(tuple(audit[field] for field in ("qid", "recipe_id", "presentation_id")) == expected_identity
                and audit["input_tokens"] <= payload["input"]["max_input_tokens"]
                and audit["recipe_tokens_used"] <= payload["input"]["recipe_token_budget"]
                and audit["titles_retained"] == audit["steps"] == len(record["steps"]), "Input budget or title preservation failed")
        require(tuple(visit["candidate_texts"]) == ACTIONS
                and visit["candidate_token_ids"] == [candidate["token_ids"] for candidate in ref["candidates"]], "Actual action grammar differs")
        ref_policy = recompute_policy(ref["token_logps"], ref["candidates"])
        policy = recompute_policy(visit["token_logps_including_eos"], ref["candidates"])
        check_policy_fields(ref, ref_policy); check_policy_fields(visit, policy)
        vector_close(visit["sequence_logps"], policy["scores"], "Sequence score not token sum", 1e-6)
        vector_close(visit["reference_logq"], ref_policy["logq"], "Actual reference logq differs", 1e-9)
        close(visit["canonical_mass"], math.exp(policy["logZ"]), "Canonical mass differs")
        count = policy["token_count"]
        constant_count = count if constant_count is None else constant_count
        require(count == constant_count == payload["canonical_completion_token_count"] == visit["completion_token_count"]==5,
                "Sequence loss divisor C changed")
        labels = exact_labels(record); gold = labels.index(1)
        require(ref["exact_rewards"] == visit["exact_rewards"] == labels
                and ref["gold_candidate_index"] == gold, "Cached/actual verifier reward differs")
        audit_numerical_checks(ref["numerical_checks"], origin_indices[index] == 0, .01, ref["token_logps"])
        audit_numerical_checks(visit["numerical_reference_checks"], index == 0, .01, visit["token_logps_including_eos"])
        current_forwards += 25
        samples = None
        if args.method == "rloo_kl":
            samples, uniforms = legacy.replay_draws(policy["q"], rng)
            require(visit["sampled_indices"] == samples and visit["sampling_uniforms"] == uniforms,
                    "RLOO draws differ from exact seeded replay; no resampling is allowed")
        else:
            require(visit["sampled_indices"] is None and visit["sampling_uniforms"] is None, "Enumerated/raw method unexpectedly sampled")
        require(visit["sampling_rng_state_sha256"] == hash_json(rng.getstate()), "Sampling RNG state trajectory differs")
        terms = independent_terms(args.method, policy["q"], policy["logq"], gold, ref_policy["logq"], args.beta, samples)
        for field in ("coefficients", "u"):
            vector_close(visit[field], terms[field], "Independent " + field + " differs")
        require((visit["advantages"] is None) == (terms["advantages"] is None), "Wrong LOO baseline type")
        if terms["advantages"] is not None:
            vector_close(visit["advantages"], terms["advantages"], "Repeated-sample LOO advantages differ")
        for field in ("mean_u", "exact_expected_reward", "kl_q_reference", "regularized_expected_reward", "closed_ce_nll", "active_beta"):
            close(visit[field], terms[field], "Independent " + field + " differs")
        require(visit["gold_index"] == gold, "Wrong actual gold index")
        # Machine-zero selection is audited literally; no new approximation or
        # threshold changes which tiny stored coefficients trigger a backward.
        selected = [i for i, value in enumerate(visit["coefficients"]) if value != 0]
        require(visit["nonzero_coefficients"] == len(selected) and visit["group_active"] == bool(selected)
                and [check["candidate_index"] for check in visit["backward_checks"]] == selected,
                "Actual backward schedule does not match verified coefficients")
        require(all(isinstance(check["max_token_logp_difference"], (int, float))
                    and 0 <= check["max_token_logp_difference"] <= .01 for check in visit["backward_checks"]), "Gradient forward mismatch")
        require(visit["consumption_point"] == "all_selected_candidate_backward_calls_returned", "Visit is not a completed backward consumption")
        backwards += len(selected); active_by_step[index // 4] += bool(selected)
        if (index + 1) % 4 == 0:
            counts_by_step.append((current_forwards, backwards))
    for index, step in enumerate(steps):
        require(step["global_step_before_update"] == index and step["global_step"] == index + 1
                and step["actual_prompt_visits"] == (index + 1) * 4 and step["groups"] == 4
                and step["active_groups"] == active_by_step[index] and step["optimizer_step_executed"] is True
                and step["optimizer_state_entries"] == 392 and step["run_signature_sha256"] == signature, "Optimizer step coverage/identity differs")
        for position in ("before_clip", "after_clip"):
            gradient = step[position]
            require(gradient["trainable_gradient_tensors"] == 392 and 0 <= gradient["nonzero_gradient_tensors"] <= 392
                    and math.isfinite(gradient["l2_norm"]) and gradient["l2_norm"] >= 0, "Invalid recorded accumulated gradient")
        require(step["after_clip"]["l2_norm"] <= 1.0001, "Gradient clipping did not enforce the locked bound")
        require(step["zero_reward_gradient_batch"] == (step["before_clip"]["l2_norm"] == 0), "Wrong zero-gradient batch flag")
        require(math.isfinite(step["actual_parameter_update_l2"]) and step["actual_parameter_update_l2"] >= 0
                and math.isfinite(step["elapsed_seconds"]) and step["elapsed_seconds"] > 0, "Invalid update/time diagnostic")
        close(step["learning_rate"], cli["learning_rate"], "Learning rate changed")
        forwards, back = counts_by_step[index]
        require(step["cumulative_current_policy_candidate_forwards_including_numerical_checks"] == forwards
                and step["cumulative_gradient_candidate_forwards"] == step["cumulative_candidate_backwards"] == back
                and step["initial_reference_candidate_forwards_charged_this_run"] == reference_forwards, "Per-step compute counters differ")
    require(outcome["actual_visits_sha256"] == sha256_file(root / "actual_prompt_visits.jsonl")
            and outcome["step_times_sha256"] == sha256_file(root / "step_times.jsonl"), "Final trajectory hashes differ")
    counts = {"reference_candidate_forwards_charged_this_run": reference_forwards,
              "current_policy_candidate_forwards_including_numerical_checks": current_forwards,
              "gradient_candidate_forwards": backwards, "candidate_backwards": backwards,
              "candidate_forwards": reference_forwards + current_forwards + backwards}
    require(all(outcome[field] == value for field, value in counts.items()), "Final forward/backward counters differ")
    elapsed=outcome['elapsed_seconds_including_reference']
    require(math.isfinite(elapsed) and sum(s['elapsed_seconds'] for s in steps)<=elapsed
        <=(stamp(state['finished_utc'])-stamp(state['started_utc'])).total_seconds()+1.,'Actual update times contradict total persisted runtime')
    expected_checkpoint_steps = list(range(cli["save_steps"], args.steps + 1, cli["save_steps"]))
    if args.steps not in expected_checkpoint_steps:
        expected_checkpoint_steps.append(args.steps)
    require([item["step"] for item in outcome["checkpoints"]] == expected_checkpoint_steps
            and {path.name for path in root.glob("checkpoint-*") if path.is_dir()}
            == {f"checkpoint-{step}" for step in expected_checkpoint_steps}, "Checkpoint schedule differs")
    required = [signature_file, root / "state.json", root / "train_outcome.json", root / "trainable_parameters.json",
                root / "actual_prompt_visits.jsonl", root / "step_times.jsonl", ref_path, ref_config_path, ref_completion_path]
    required.extend([args.policy_path,*pool_artifacts,*lineage_artifacts])
    required.extend(data_path(ROOT/p) for p in source_audit['required_artifacts'])
    required.extend(generated_artifacts)
    required.extend(data_path(ROOT/p) for p in bound_sources)
    last_fingerprint = None
    for item in outcome["checkpoints"]:
        step, directory = item["step"], data_path(item["path"])
        require(directory == root / f"checkpoint-{step}", "Checkpoint escaped the audited run")
        audit_path = directory / "finite_policy_checkpoint_audit.json"
        audit, saved = load(audit_path), load(directory / "trainer_state.json")
        identity = model_weight_identity(directory, adapter=True)
        require(identity == item["adapter_identity"] == audit["adapter_identity"]
                and audit["global_step"] == saved["global_step"] == step
                and audit["run_signature_sha256"] == saved["run_signature_sha256"] == signature
                and audit["reference_policy_sha256"] == reference_hash
                and saved["actual_prompt_visits"] == step * 4 and saved["resume_supported"] is False,
                "Checkpoint identity/state differs")
        close(saved["epoch"], step / args.steps, "Checkpoint epoch differs")
        require(set(audit["state_sha256"]) == set(STATE_NAMES), "Checkpoint state hash coverage is incomplete")
        for name in STATE_NAMES:
            path = directory / name
            require(sha256_file(path) == audit["state_sha256"][name], "Checkpoint state bytes differ: " + name)
            if name != "trainer_state.json":
                audit_state_file(path)
            required.append(path)
        last_fingerprint = tensor_fingerprint(directory, inventory)
        required.extend([audit_path, directory / "adapter_model.safetensors", directory / "adapter_config.json"])
    final_dir = root / "adapter_final"
    final_identity = model_weight_identity(final_dir, adapter=True)
    final_fingerprint = tensor_fingerprint(final_dir, inventory)
    require(final_identity == outcome["final_adapter_identity"] and final_fingerprint == outcome["final_adapter_fingerprint"]
            == last_fingerprint, "Final safe tensor bytes differ from runtime fingerprint or last checkpoint")
    effective = audit_effective_update(initial_fingerprint, final_fingerprint, outcome["effective_adapter_update"], outcome["scientific_outcome"])
    require(outcome["unique_questions"] == len({row["qid"] for row in records})
            and outcome["unique_recipes"] == len({row["recipe_id"] for row in records}), "Unique question/recipe counts differ")
    required.extend([final_dir / "adapter_model.safetensors", final_dir / "adapter_config.json"])
    return {"status": "complete_verified_coverage_policy_run", "passed": True, "method": args.method, "seed": args.seed,
            "limit": args.limit, "optimizer_steps": args.steps, "beta": args.beta,
            "effective_adapter_update": effective, "ineffective_updates_are_scientifically_valid": True,
            "run_signature_sha256": signature, "forward_backward_counts": counts,
            "reference": {"path": str(ref_path), "sha256": reference_hash, "rows": len(references), "reused": bool(cli["reference_cache"])},
            "initial_adapter_fingerprint": initial_fingerprint, "final_adapter_fingerprint": final_fingerprint,
            "source_sha256": bound_sources, "manifest_sha256": args.expected_manifest_sha256,
            "locked_policy_sha256":sha256_file(args.policy_path),'coverage_mode':args.policy['mode'],
            "reference_lineage_sha256":expected_lineage,'prior_seen_pool_questions':len(pool),
            'generated_reference_receipt_sha256':generated_receipt_sha,'parent_stage_policy_sha256':parent_stage_sha,
            "shared_reference_source_audit":{'status':source_audit['status'],'reference_candidate_forwards_charged_once':25600,
                'updates':0,'policy_sha256':source_audit['locked_policy_sha256'],
                'accounting_scope':'Shared source evidence, not an extra25600 forwards charged to this training arm'},
            "required_artifacts": {str(path.relative_to(ROOT)): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in required},
            "provenance": {"verifier_sha256": sha256_file(__file__), "versions": package_versions(), "arguments": vars(args)},
            "limits": ["CPU verification recomputes mathematics and persisted artifacts, not actual VLM forwards or Adam tensor arithmetic.",
                       "Torch optimizer/scheduler/RNG archives are content-hashed and CRC-checked, never unpickled by this verifier.",
                       "Persisted LoRA tensors independently reproduce runtime adapter fingerprints; frozen vision relies on runtime guards plus unchanged initial base identity."]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',required=True,type=Path)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--reference-only',action='store_true',help='Audit actual reference-only source:1024 rows,zero optimizer/updates')
    parsed=parser.parse_args();output=data_path(parsed.output)
    require(not output.exists(),'Prior independent audit exists; do not overwrite its evidence')
    try:
        args=resolve_args(parsed)
        result=verify_reference(args) if parsed.reference_only else verify(args)
    except BaseException as error:
        if not output.exists():write_json(output,{'status':'failed_coverage_verification','passed':False,
            'error':type(error).__name__+': '+str(error),'verifier_sha256':sha256_file(__file__)})
        raise
    result['verified_utc']=datetime.now(timezone.utc).isoformat()
    result.setdefault('provenance',{})['arguments']={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    result['provenance']['verifier_sha256']=sha256_file(__file__)
    result['provenance']['old_independent_verifier_sha256']=OLD_VERIFIER_SHA
    write_json(output,result)
    print(json.dumps({k:result[k] for k in ('status','passed')},ensure_ascii=False))


if __name__=='__main__':main()

