"""CPU-only scientific/resource contracts for the new reference-refresh study."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from recipeqa.modeling import hash_json,sha256_file,read_jsonl,write_json

BASE=Path('/root/autodl-tmp/rl_recipeqa')
BETA=1.6363096678310884
WALL_SECONDS=86400
DEV_SHA='c998796fbc07a24c7a24062d6d0aba07500c75b34568fd7ba67f8c38daae03c9'
FULL_TRAIN='exploratory/principled_ordering_20261105/full_train_data/train_all7558.jsonl'
FULL_SHA='b9979ee0612679d2e9460dd3bb8a6f68e7dea6794bc6cdcc3e1defdc87d8d949'
INITIAL_ADAPTER='exploratory/principled_ordering_20261105/full_sft/train/checkpoint-768'
MODEL_PATH='models/Qwen3-VL-2B-Instruct'
REVISION='89644892e4d85e24eaac8bacfd4f463576704203'
FLOWS=('flow44','flow45');ARM_KINDS=('raw_sft','fixed_beta','fixed_half_beta','refresh_beta')
METHODS=ARM_KINDS
ARMS=tuple(f'{flow}_{arm}' for flow in FLOWS for arm in ARM_KINDS)
PATH_SMOKES=tuple(f'path_smoke_{flow}_refresh_beta' for flow in FLOWS)
SCORING_SOURCE_PATHS=('exploratory/principled_ordering_20261105/score_permutations.py',
 'exploratory/principled_ordering_20261105/gradient_alignment.py','src/recipeqa/modeling.py','src/recipeqa/metrics.py')
TRAIN_SOURCE_PATHS=('research/reference_refresh_20261107_v4/refresh_train.py','research/reference_refresh_20261107_v4/refresh_contract.py',
 'research/reference_refresh_20261107_v4/refresh_pipeline.py','research/reference_refresh_20261107_v4/build_draft_policy.py',
 'research/reference_refresh_20261107_v4/launch_stage.py','research/reference_refresh_20261107_v4/run_cpu_preflight.py',
 'research/reference_refresh_20261107_v4/test_refresh_contract.py','research/reference_refresh_20261107_v4/test_refresh_pipeline.py',
 'research/reference_refresh_20261107_v4/verify_refresh_run.py','research/reference_refresh_20261107_v4/test_verify_refresh_run.py',
 'research/grounding_equivariance_20261106/orbit_diagnostic.py','scripts/predict.py','scripts/validate_run.py',
 'exploratory/principled_ordering_20261105/finite_policy_train.py',
 'exploratory/principled_ordering_20261105/verify_finite_run.py',
 'exploratory/principled_ordering_20261105/test_finite_policy_tensor_reference.py',
 'exploratory/principled_ordering_20261105/score_permutations.py',
 'exploratory/principled_ordering_20261105/gradient_alignment.py',
 'src/recipeqa/modeling.py','src/recipeqa/metrics.py')
GENERATED_REFERENCE_ALLOWED_ROW_FIELDS=frozenset(('index','reference_spec_sha256'))
GENERATED_REFERENCE_ALLOWED_SPEC_FIELDS=frozenset(('manifest_sha256','selected_records_sha256','pixel_sha256'))


def require(value,message):
    if not value:raise ValueError(message)


def identity(row):return (str(row['qid']),str(row['presentation_id']))


def fixed_pool(full_rows):
    require(len(full_rows)==7558,'Original complete7558 training file required')
    pool=deepcopy(full_rows[:256])
    require(len({identity(r) for r in pool})==256 and len({str(r['recipe_id']) for r in pool})==254,'Fixed256 identities/254 recipes required')
    for r in pool:
        require(r.get('source_split')=='train' and r.get('technical_status')=='ok'
            and r.get('manifest_set') not in ('val','test','confirmation','orbit','process'),'Only undamaged train allowed')
        require(len(r['image_paths_display'])==4 and set(r['target_labels'])==set('ABCD'),'Four-image canonical training required')
    return pool


def ordered_pools(pool):
    require(len(pool)==256 and len({identity(r) for r in pool})==256,'Unfiltered256 pool required')
    second=deepcopy(pool);random.Random(45).shuffle(second)
    return {'flow44':deepcopy(pool),'flow45':second}


def validate_prior_exposure(pool,actual_visits,expected_signature):
    # Only actual forward/backward-completed visits before selected checkpoint768,
    # not preparation/input audits or the eventual later945 checkpoint.
    consumed=[r for r in actual_visits if r['global_step_before_update']<768]
    require(all(type(r['global_step_before_update']) is int and 0<=r['global_step_before_update']<768
        and r['consumption_point']=='training_step_forward_backward_returned'
        and r['run_signature_sha256']==expected_signature for r in consumed),'Actual consumed initial SFT signature/completion markers required')
    require({r['global_step_before_update'] for r in consumed}==set(range(768))
        and all(sum(r['global_step_before_update']==i for r in consumed)==8 for i in range(768)),
        'Initial768 updates must each contain eight actualcompleted visits')
    keys={identity(r) for r in consumed}
    require(len(consumed)==6144 and len(keys)==6144,'Initial SFT768 actual6144 exposure proof required')
    require({identity(r) for r in pool}.issubset(keys),'Do not mislabel unconsumed data as previously seen')
    return {'initial_checkpoint_step':768,'actual_consumed_visits':6144,'pool_size':256,
        'already_consumed_pool_questions':256,'source':'actual_prompt_visits beforeupdate768, not input preparation'}


def budget_for(mode):
    require(mode in ('smoke','path_smoke','formal'),'Unknown refresh child mode')
    unique,passes,updates,save=(4,1,1,1) if mode=='smoke' else ((4,2,2,1) if mode=='path_smoke' else (256,2,128,64))
    return {'unique_count':unique,'passes':passes,'limit':unique*passes,'max_steps':updates,'save_steps':save}


def validate_child(policy):
    require(policy['status']=='locked_before_reference_refresh_training','Draft refresh child cannot run')
    cli=policy['cli'];mode=policy['mode'];arm=policy['arm'];flow=policy['flow']
    require(flow in FLOWS and arm in ARM_KINDS and (mode!='path_smoke' or arm=='refresh_beta'),'Wrong child flow/arm/path role')
    budget=budget_for(mode)
    require(all(cli[k]==v for k,v in budget.items()),'Fixed unique/passes/exposure/update/save budget changed')
    require(cli['seed']==int(flow[-2:]) and Path(cli['manifest']).name==f'{"smoke" if mode!="formal" else "formal"}_{flow}.jsonl','Flow/order/seed manifest role differs')
    require(cli['model_path']==str(BASE/MODEL_PATH) and cli['adapter_path']==str(BASE/INITIAL_ADAPTER)
        and cli['revision']==REVISION,'Same original Qwen/SFT768 initialization required')
    method='raw_sft' if arm=='raw_sft' else 'exact_kl'
    beta=0. if arm=='raw_sft' else (BETA/2 if arm=='fixed_half_beta' else BETA)
    boundary=None if arm!='refresh_beta' or mode=='smoke' else budget['max_steps']//2
    require(cli['method']==method and cli['beta']==beta and cli['refresh_boundary_step']==boundary
        and cli['learning_rate']==5e-6 and cli['final_bank_count']==budget['unique_count'],'Fixed objective/refresh/finalbank differs')
    require(bool(cli['reference_cache']) and cli['prepare_reference_only'] is False,'Only sealed existing q0 cache; no reference-only training')
    require(cli['min_pixels']==65536 and cli['max_pixels']==262144 and cli['recipe_token_budget']==1536
        and cli['max_input_tokens']==4096 and cli['logp_atol']==.01,'Frozen original input/scoring config changed')
    require(0<cli['wall_seconds']<=WALL_SECONDS,'Child budget must fit common24h stage')
    require(set(policy['sources_sha256'])==set(TRAIN_SOURCE_PATHS),'Exact complete source closure required')
    require(policy['reward']=='unique_complete_order_exact_0_1' and policy['selection']=='fixed128_primary_step64_boundary_no_dev_selection','No new reward/selection')
    require(policy['optimizer']=={'accumulation':4,'clip_norm':1.,'canonical_C_including_eos':5,'fresh_adamw':True},'Normalization/optimizer contract changed')
    require(policy['batch_clock_started_utc']==policy['first_preflight_started_utc'],'One original first-preflight clock, no reset')
    return cli


def reference_reindex(source_rows,source_spec,source_records,target_records,manifest_sha256):
    """Deterministic full-order/subset transform; never recompute model scores.

    Source numerical checks are kept byte-for-byte. Their literal full-logit
    schedule refers to ORIGINAL source index0, not the target order's first row.
    """
    require(len(source_rows)==len(source_records)==1024,'One complete new1024 source cache required')
    require(len(target_records) in (4,256),'Only fixedsmoke4 or formal256 target order')
    bank={identity(r):(i,r,source_rows[i]) for i,r in enumerate(source_records)}
    require(len(bank)==1024 and len({identity(r) for r in target_records})==len(target_records),'Duplicate cache identities')
    new_spec=deepcopy(source_spec)
    new_spec['manifest_sha256']=manifest_sha256;new_spec['selected_records_sha256']=hash_json(target_records)
    pixels={p for r in target_records for p in r['image_paths_display']}
    require(pixels.issubset(source_spec['pixel_sha256']),'Target image absent from actual source cache')
    new_spec['pixel_sha256']={p:source_spec['pixel_sha256'][p] for p in sorted(pixels)}
    rows=[];lineage=[]
    for index,record in enumerate(target_records):
        source_index,original,raw=bank[identity(record)]
        require(record==original,'Reference reordering cannot change record, visitid, display, gold, images or recipe')
        require(raw['index']==source_index and identity(raw)==identity(record)
            and raw['reference_spec_sha256']==hash_json(source_spec),'Source cache index/identity/spec differs')
        transformed=deepcopy(raw);transformed.update(index=index,reference_spec_sha256=hash_json(new_spec))
        before={k:v for k,v in raw.items() if k not in GENERATED_REFERENCE_ALLOWED_ROW_FIELDS}
        after={k:v for k,v in transformed.items() if k not in GENERATED_REFERENCE_ALLOWED_ROW_FIELDS}
        require(before==after,'Forbidden cache payload modification')
        rows.append(transformed)
        lineage.append({'index':index,'source_index':source_index,'qid':record['qid'],
            'presentation_id':record['presentation_id'],'record_sha256':hash_json(record),
            'source_row_sha256':hash_json(raw),'target_row_sha256':hash_json(transformed),
            'unchanged_payload_sha256':hash_json(before),
            'reference_literal_check_was_source_index_zero':source_index==0})
    return rows,new_spec,{'schema_version':1,'source_spec_sha256':hash_json(source_spec),
        'target_spec_sha256':hash_json(new_spec),'source_selected_records_sha256':hash_json(source_records),
        'target_selected_records_sha256':hash_json(target_records),'source_rows':1024,'target_rows':len(rows),
        'allowed_row_changes':sorted(GENERATED_REFERENCE_ALLOWED_ROW_FIELDS),
        'allowed_spec_changes':sorted(GENERATED_REFERENCE_ALLOWED_SPEC_FIELDS),'rows':lineage,
        'scope':'Identity-only reorder/subset, all token scores/candidates/checks/inputaudit unchanged; no inference or new labels.'}


def materialize_reference(source_cache,source_manifest,target_manifest,target_cache):
    source_cache=Path(source_cache);target_cache=Path(target_cache)
    require(not target_cache.exists(),'Reference target must be fresh; never overwrite/retry')
    source_config=Path(str(source_cache)+'.config.json');source_complete=Path(str(source_cache)+'.completion.json')
    spec=json.loads(source_config.read_text());complete=json.loads(source_complete.read_text())
    require(complete['status']=='complete_fixed_initial_reference' and complete['rows']==1024
        and complete['cache_sha256']==sha256_file(source_cache) and complete['reference_spec_sha256']==hash_json(spec),'Source cache incomplete')
    rows,new_spec,lineage=reference_reindex(read_jsonl(source_cache),spec,read_jsonl(source_manifest),read_jsonl(target_manifest),sha256_file(target_manifest))
    target_cache.parent.mkdir(parents=True,exist_ok=True)
    with target_cache.open('x') as handle:
        for row in rows:handle.write(json.dumps(row,allow_nan=False)+'\n')
    write_json(str(target_cache)+'.config.json',new_spec)
    write_json(str(target_cache)+'.completion.json',{'status':'complete_fixed_initial_reference','rows':len(rows),
        'cache_sha256':sha256_file(target_cache),'reference_spec_sha256':hash_json(new_spec)})
    lineage.update(source_cache=str(source_cache.resolve()),source_cache_sha256=sha256_file(source_cache),
        source_config_sha256=sha256_file(source_config),source_completion_sha256=sha256_file(source_complete),
        source_manifest=str(Path(source_manifest).resolve()),source_manifest_sha256=sha256_file(source_manifest),
        target_cache=str(target_cache.resolve()),target_cache_sha256=sha256_file(target_cache),
        target_manifest=str(Path(target_manifest).resolve()),target_manifest_sha256=sha256_file(target_manifest))
    write_json(str(target_cache)+'.lineage.json',lineage)
    return lineage


def remaining_seconds(policy,utc_now=None):
    now=datetime.now(timezone.utc) if utc_now is None else utc_now
    start=datetime.fromisoformat(policy['first_preflight_started_utc'].replace('Z','+00:00'))
    end=datetime.fromisoformat(policy['absolute_deadline_utc'].replace('Z','+00:00'))
    require(start.tzinfo and end.tzinfo and now.tzinfo and start<=now
        and 0<(end-start).total_seconds()<=WALL_SECONDS,'Original common24h clock required')
    remaining=min(WALL_SECONDS-(now-start).total_seconds(),(end-now).total_seconds())
    require(remaining>0,'Original phase clock expired; no reset or resume')
    return remaining


# Resource estimation is implemented in the coordinator and independently
# checked from persisted technical and historical wall/counter receipts.


# Continuation has one approved cumulative clock. Historical evidence keeps its
# original policies and times; it is never relabelled as a v4 execution.
ORIGINAL_FIRST='2026-11-07T04:18:24.421606+00:00'
APPROVED_DEADLINE='2026-11-08T04:18:24.421606+00:00'
OLD_DEADLINE='2026-11-08T00:18:24.421606+00:00'
OLD_POLICY='research/reference_refresh_20261107_v3/PROTOCOL_V3.json'
OLD_POLICY_SHA='71106d4ea8a83d839449316957d2ebb94aefb9a3ff9c9d76d06587755db5d5e8'
AUTHORIZATION='research/reference_refresh_20261107_continuation/AUTHORIZATION_24H.json'
TECHNICAL_KEYS=tuple('smoke_'+a for a in ARMS)+PATH_SMOKES


def validate_continuation_clock(p):
    require(p['first_preflight_started_utc']==ORIGINAL_FIRST
        and p['absolute_deadline_utc']==APPROVED_DEADLINE,
        'Authorized24h continuation must preserve exact original first and approved absolute')
    require(p['wall_limit_seconds']==86400 and p['budget_cny_cap']==38.4,
        'Only explicitly approved cumulative24h/38.4CNY')


def historical_commands(reuse,old_state):
    require(old_state['status']=='failed_or_budget_stopped_no_retry'
        and old_state['policy_sha256']==OLD_POLICY_SHA
        and old_state['first_preflight_started_utc']==ORIGINAL_FIRST
        and old_state['absolute_deadline_utc']==OLD_DEADLINE
        and old_state['active_child_pid'] is None,'Historical stopped state/policy/clock differs')
    expected=['cpu_preflight']+['q0_'+m+'_'+f+'_verify' for f in FLOWS for m in ('smoke','formal')]
    expected += [k+'_'+suffix for k in TECHNICAL_KEYS for suffix in ('train','verify')]
    commands=old_state['commands']
    require([c['stage'] for c in commands]==expected and all(c['returncode']==0 for c in commands),
        'Original25 successful preformal commands required; formal must not have started')
    start=datetime.fromisoformat(ORIGINAL_FIRST);end=datetime.fromisoformat(OLD_DEADLINE)
    prior=start
    for c in commands:
        a=datetime.fromisoformat(c['started_utc']);b=datetime.fromisoformat(c['finished_utc'])
        require(c['pgid']==old_state['pgid'] and prior<=a<=b<end,
            'Historical actual command PGID/time/order invalid')
        prior=b
    require(reuse['old_commands']==commands[5:] and len(reuse['old_commands'])==20,
        'Historical technical command bytes/fields differ')
    return commands[5:]


def validate_reuse(p,root,physical=True):
    """Hash immutable old physical evidence without invoking an expired clock.

    Receipt validity is checked at its original actual timestamps. No old source
    is imported or executed, no GPU/model constructed, and old metadata unchanged.
    """
    root=Path(root).resolve();r=p['reused_technical']
    validate_continuation_clock(p)
    def path(name):
        f=Path(name);f=f.relative_to(BASE) if f.is_absolute() else f
        require('..' not in f.parts,'Historical evidence traversal')
        f=(root/f).resolve();require(f.is_relative_to(root),'Historical evidence outside data disk');return f
    def read(name):return json.loads(path(name).read_bytes())
    require(r['old_policy']['path']==OLD_POLICY and r['old_policy']['sha256']==OLD_POLICY_SHA
        and sha256_file(path(OLD_POLICY))==OLD_POLICY_SHA,'Original v3 policy identity differs')
    auth=read(AUTHORIZATION)
    require(auth['clock_reset'] is False and auth['original_first_preflight_utc']==ORIGINAL_FIRST
        and auth['approved_cumulative_wall_limit_seconds']==86400
        and auth['approved_absolute_deadline_utc']==APPROVED_DEADLINE
        and auth['approved_total_reference_CNY_cap']==38.4,'Explicit cumulative24h authorization differs')
    old=read(OLD_POLICY);state=read(r['old_state']['path'])
    require(sha256_file(path(r['old_state']['path']))==r['old_state']['sha256'],'Old25 command journal changed')
    historical_commands(r,state)
    require(old['wall_limit_seconds']==72000 and old['budget_cny_cap']==32
        and old['first_preflight_started_utc']==ORIGINAL_FIRST and old['absolute_deadline_utc']==OLD_DEADLINE,
        'Historical20h policy must remain historical')
    require(old['initial_identity']==p['initial_identity'] and old['pool_spec']==p['pool_spec'],
        'Historical technical initialization/pool differs')
    require(set(r['old_children'])==set(TECHNICAL_KEYS)
        and all(r['old_children'][k]==old['children'][k] for k in TECHNICAL_KEYS),
        'Historical10 child descriptors changed')
    for group in ('sources_sha256','artifacts_sha256'):
        for name,h in old[group].items():require(sha256_file(path(name))==h,'Old frozen dependency changed:'+name)
    import ast
    def functions(name):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(path(name).read_bytes()).body
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    before=functions('research/reference_refresh_20261107_v3/refresh_train.py')
    after=functions('research/reference_refresh_20261107_v4/refresh_train.py')
    require(set(before)==set(after) and all(before[k]==after[k] for k in before if k!='validate_authorization'),
        'New trainer changed function math beyond continuation authorization')
    cpu=read(r['old_cpu']['path'])
    require(sha256_file(path(r['old_cpu']['path']))==r['old_cpu']['sha256']
        and cpu['passed'] is True and cpu['tests_run']==cpu['expected_tests']==67
        and cpu['failures']==cpu['errors']==cpu['skipped']==0 and cpu['cuda_initialized'] is False
        and cpu['policy_sha256']==OLD_POLICY_SHA and cpu['sources_sha256']==old['sources_sha256']
        and cpu['native_watchdog_probe']['passed'] is True,'Historical67CPU/native proof differs')
    require(set(r['technical_receipts'])==set(TECHNICAL_KEYS),'All10 old technical proofs required')
    hashes={};count=0
    for flow in FLOWS:
        for mode in ('smoke','formal'):
            name=str(Path(old['output_directory'])/f'q0_{mode}_{flow}_verification.json')
            proof=read(name)
            oldkey=('smoke_' if mode=='smoke' else '')+flow+'_raw_sft'
            oldchild=read(old['children'][oldkey]['path'])
            require(proof['passed'] is True and proof['status']=='complete_verified_reference_refresh_q0'
                and proof['mode']==mode and proof['flow']==flow and proof['arm']=='raw_sft'
                and proof['unique']==(4 if mode=='smoke' else 256) and proof['new_q0_candidate_forwards']==0
                and proof['historical_whole1024_reference_candidate_forwards']==25600
                and proof['reference_policy_sha256']==sha256_file(path(oldchild['cli']['reference_cache']))
                and proof['locked_policy_sha256']==old['children'][oldkey]['sha256'],
                'Historical q0 role/policy changed')
            for name,v in proof['required_artifacts'].items():
                require(name not in hashes or hashes[name]==v,'Conflicting old q0 receipts')
                hashes[name]=v
    for key,item in r['technical_receipts'].items():
        proof=read(item['path']);child=read(r['old_children'][key]['path'])
        require(sha256_file(path(item['path']))==item['sha256']
            and sha256_file(path(r['old_children'][key]['path']))==r['old_children'][key]['sha256']
            and proof['passed'] is True and proof['status']=='complete_verified_reference_refresh_run'
            and proof['locked_policy_sha256']==r['old_children'][key]['sha256']
            and proof['verifier_sha256']==old['sources_sha256']['research/reference_refresh_20261107_v3/verify_refresh_run.py']
            and proof['effective_adapter_update'] is True
            and proof['initial_adapter_fingerprint']!=proof['final_adapter_fingerprint']
            and all(proof[f]==child[f] for f in ('mode','arm','flow')),'Unmatched historical technical proof')
        auth=read(child['authorization_path']);train=next(c for c in state['commands'] if c['stage']==key+'_train')
        require(auth['stage_policy_sha256']==OLD_POLICY_SHA and auth['child_policy_sha256']==r['old_children'][key]['sha256']
            and auth['child_key']==key and auth['phase']==child['mode'] and auth['pgid']==state['pgid']
            and auth['first_preflight_started_utc']==ORIGINAL_FIRST and auth['absolute_deadline_utc']==OLD_DEADLINE
            and datetime.fromisoformat(ORIGINAL_FIRST)<=datetime.fromisoformat(auth['issued_utc'])<=datetime.fromisoformat(train['started_utc']),
            'Historical authorization identity/time changed')
        steps=read_jsonl(path(Path(child['cli']['output_dir'])/'step_times.jsonl'))
        require(all(math.isfinite(row['actual_parameter_update_l2']) and row['actual_parameter_update_l2']>0 for row in steps),
            'Historical technical ineffective actual update')
        expected_steps=2 if key in PATH_SMOKES else 1
        require(proof['steps']==expected_steps and proof['visits']==4*expected_steps and len(steps)==expected_steps
            and child['initial_identity']==p['initial_identity']
            and child['first_preflight_started_utc']==ORIGINAL_FIRST
            and child['absolute_deadline_utc']==OLD_DEADLINE,'Old technical role/budget/clock differs')
        for name,v in proof['required_artifacts'].items():
            require(name not in hashes or hashes[name]==v,'Conflicting old physical receipts')
            hashes[name]=v
        count+=1
    require(r['physical_receipts']==hashes and len(hashes)>100,'Complete old physical receipt union required')
    if physical:
        for name,item in hashes.items():
            f=path(name)
            require(type(item['bytes']) is int and f.stat().st_size==item['bytes']
                and sha256_file(f)==item['sha256'],'Old raw physical content changed:'+name)
    for name,item in r['additional_receipts'].items():
        f=path(name);require(f.stat().st_size==item['bytes'] and sha256_file(f)==item['sha256'],
            'Old prerequisite/review receipt changed:'+name)
    require(r['old_q0_count']==4 and r['historical_new_model_calls']==0
        and r['technical_reexecution_allowed'] is False,'Historical reuse scope changed')
    return {'passed':True,'old_policy_sha256':OLD_POLICY_SHA,'old_commands':25,
        'historical_reused_commands':20,'technical_runs':count,'physical_files_rehashed':len(hashes) if physical else 0,
        'old_cpu_tests':67,'old_q0_audits':4,'current_model_GPU_calls':0,
        'clock_policy':'Original receipts checked at actual historical times; no current expiry check or relabelling.'}
