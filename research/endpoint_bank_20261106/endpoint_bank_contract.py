#!/usr/bin/env python3
"""Explicit CPU contract for the fixed, zero-update endpoint likelihood bank."""
from __future__ import annotations
from datetime import datetime, timezone
from itertools import permutations
import json
import math
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from recipeqa.modeling import hash_json,read_jsonl,sha256_file

BASE=Path('/root/autodl-tmp/rl_recipeqa')
WALL_SECONDS=10800
BETA=1.6363096678310884
REVISION='89644892e4d85e24eaac8bacfd4f463576704203'
FULL_TRAIN='exploratory/principled_ordering_20261105/full_train_data/train_all7558.jsonl'
FULL_SHA='b9979ee0612679d2e9460dd3bb8a6f68e7dea6794bc6cdcc3e1defdc87d8d949'
DEV='data/manifests/val.jsonl'
DEV_SHA='c998796fbc07a24c7a24062d6d0aba07500c75b34568fd7ba67f8c38daae03c9'
HISTORICAL_OUTCOME='exploratory/coverage_20261106/stage_v1/reference/reference_only_outcome.json'
INPUT={'min_pixels':65536,'max_pixels':262144,'recipe_token_budget':1536,
       'max_input_tokens':4096,'max_completion_tokens':32,'prompt_version':'recipeqa-order-v1'}
ENDPOINTS=('initial_sft768','flow44_raw_sft_256','flow44_exact_kl_256',
           'flow45_raw_sft_256','flow45_exact_kl_256')
BANKS={'train256':256,'dev200':200}
CANDIDATES=tuple(' '.join(p) for p in permutations('ABCD'))
THREADS=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
NEW_SOURCES=('endpoint_bank_contract.py','score_endpoint_bank.py','run_endpoint_bank.py',
 'launch_endpoint_bank.py','run_cpu_preflight.py','build_draft_policy.py',
 'test_endpoint_bank_contract.py','bank_math.py','verify_endpoint_scores.py',
 'analyze_endpoint_bank.py','test_bank_math.py','test_verify_endpoint_scores.py',
 'test_analyze_endpoint_bank.py')
SOURCE_PATHS=tuple('research/endpoint_bank_20261106/'+n for n in NEW_SOURCES)+(
 'exploratory/principled_ordering_20261105/score_permutations.py',
 'src/recipeqa/modeling.py','src/recipeqa/metrics.py')
OLD_SOURCES={
 'exploratory/principled_ordering_20261105/score_permutations.py':'affeda80416145d2b6a134172e812d15d48a86d77c67128bd2e5f84e0c341819',
 'src/recipeqa/modeling.py':'5fedb7a60877e965861aa9174179880754cb0555c4167265fdd8c3781af39771',
 'src/recipeqa/metrics.py':'5287d44a3bb4ede3a44ce0a6625aee399260afecd42c984441a8e07492a2423a'}


def require(value,message):
    if not value:raise ValueError(message)


def now():return datetime.now(timezone.utc).isoformat()
def load(path):return json.loads(Path(path).read_text())


def confined(path,root=BASE):
    p=Path(path);p=(root/p).resolve() if not p.is_absolute() else p.resolve()
    require(p.is_relative_to(root.resolve()),'Path escaped the project data disk: '+str(path))
    return p


def image_path(image_root,relative):
    root=confined(image_root);p=(root/relative).resolve()
    require(not Path(relative).is_absolute() and p.is_relative_to(root),'Image escaped its frozen image root')
    return p


def stamp(value):
    t=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(t.tzinfo is not None and t.utcoffset().total_seconds()==0,'Explicit UTC clock required')
    return t


def remaining_seconds(policy,at=None):
    start=stamp(policy['first_preflight_started_utc']);end=stamp(policy['absolute_deadline_utc'])
    require((end-start).total_seconds()==WALL_SECONDS,'Original3h absolute clock changed')
    at=datetime.now(timezone.utc) if at is None else at
    require(at>=start,'First preflight cannot be in the future')
    left=min(WALL_SECONDS-(at-start).total_seconds(),(end-at).total_seconds())
    require(left>0,'Original endpoint-bank deadline expired')
    return left


def fixed_tasks():
    return [{'id':e+'__'+b,'endpoint':e,'bank':b,'rows':n}
            for e in ENDPOINTS for b,n in BANKS.items() if (e,b)!=('initial_sft768','train256')]


def validate_policy(p,*,locked=True):
    require(p['status']==('locked_before_endpoint_bank' if locked else 'draft_endpoint_bank'),
            'Only independently reviewed and frozen policies may execute')
    require(p['wall_limit_seconds']==WALL_SECONDS and p['budget_cny_cap']==4.8
            and p['reference_cny_per_hour']==1.6,'Announced3h/4.8CNY cap differs')
    require(set(p['endpoints'])==set(ENDPOINTS) and set(p['banks'])==set(BANKS),'All five fixed endpoints and both banks required')
    require(p['tasks']==fixed_tasks() and p['formal_rows']==2024 and p['pilot_rows']==18
            and p['pilot_rows_per_task']==2,'All nine tasks and separate18pilot cost required')
    require(p['input']==INPUT and p['revision']==REVISION and p['beta']==BETA,'Frozen input/model/beta differs')
    require(isinstance(p['required_versions'],dict) and bool(p['required_versions'])
            and all(isinstance(k,str) and isinstance(v,str) for k,v in p['required_versions'].items()),
            'Historical source environment version pin required')
    require(p['candidate_policy']=='canonical24_full_sequence_plusEOS_temperature1'
            and p['tie_policy']=='exact_maximum_then_lowest_canonical_index','Candidate/tie policy differs')
    require(all(p[k] is False for k in ('test_access','confirmation_access','resume','automatic_retry'))
            and p['optimizer_steps']==p['backward_calls']==p['generation_calls']==0,'Read-only zero-update stage only')
    require(set(p['sources_sha256'])==set(SOURCE_PATHS),'Complete reviewed source closure required')
    require(all(p['sources_sha256'][k]==v for k,v in OLD_SOURCES.items()),'A frozen historical source changed')
    require(type(p['expected_cpu_tests']) is int and p['expected_cpu_tests']>0,'Nonvacuous CPU suite count required')
    timing=p['historical_timing']
    require(timing['reference_outcome']==HISTORICAL_OUTCOME and timing['rows']==1024
            and math.isfinite(timing['elapsed_seconds']) and timing['elapsed_seconds']>0
            and p['artifacts_sha256'][HISTORICAL_OUTCOME]==timing['sha256']
            and p['timing_gate_policy']=='first_row_once_remaining_second_row_with_historical_floor_1.5_plus600',
            'Historical timing floor/formula must be frozen with its actual outcome SHA')
    for name,n in BANKS.items():
        b=p['banks'][name]
        require(b['rows']==n and len(b['manifest_sha256'])==len(b['selected_records_sha256'])==64,'Bank identity incomplete')
        require((b['source_manifest'],b['source_manifest_sha256'])==((FULL_TRAIN,FULL_SHA) if name=='train256' else (DEV,DEV_SHA)),
                'Only original first256 train and existing dev200 permitted')
        confined(b['manifest'])
    base=p['endpoints']['initial_sft768']['weight_identity']['base']
    for name in ENDPOINTS:
        e=p['endpoints'][name];confined(e['adapter_path'])
        require(e['weight_identity']['base']==base and e['weight_identity']['adapter']['weight_files'],
                'Missing endpoint model or common base identity')
        require(e['weight_identity']['adapter']['directory']==str(confined(e['adapter_path'])),
                'Adapter path and locked physical identity disagree')
    require(p['endpoints']['initial_sft768']['adapter_path']=='exploratory/principled_ordering_20261105/full_sft/train/checkpoint-768',
            'Initial endpoint must be SFT768')
    for flow in ('flow44','flow45'):
        for method in ('raw_sft','exact_kl'):
            key=flow+'_'+method+'_256'
            require(p['endpoints'][key]['adapter_path']=='exploratory/coverage_20261106/stage_v1/'+flow+'_'+method+'/train/checkpoint-256',
                    'Only fixed coverage checkpoint256 is authorized')
    confined(p['model_path']);confined(p['image_root'])
    out=confined(p['output_directory']);require(out.is_relative_to(BASE/'exploratory/endpoint_bank_20261106'),'New output namespace required')
    cache=p['cached_initial_train'];require(cache['rows']==256 and cache['output_directory']=='cached_initial_train','Only fixed initial train cache reuse')
    require(set(cache['source_receipt_sha256'])=={cache[k] for k in ('source_cache','source_config','source_completion','source_manifest','source_signature','source_audit')},'Complete immutable source reference receipts required')
    if locked:remaining_seconds(p)
    else:require(p['first_preflight_started_utc'] is None and p['absolute_deadline_utc'] is None,'Draft may not pretend to have run')


def validate_records(records,bank):
    require(bank in BANKS and len(records)==BANKS[bank],'Complete fixed bank required')
    require(len({(r['qid'],r['presentation_id']) for r in records})==len(records),'Duplicate bank identities')
    require(len({r['recipe_id'] for r in records})==(254 if bank=='train256' else 200),'Fixed recipe-cluster count differs')
    for r in records:
        require((r['source_split'],r['manifest_set'])==(('train','train_all7558') if bank=='train256' else ('val','val')),
                'test/confirmation/process/orbit records are forbidden')
        require(r['technical_status']=='ok' and len(r['image_paths_display'])==4,'Missing/damaged four-image input')
        require(sorted(r['display_perm'])==sorted(r['target_order'])==list(range(4))
                and set(r['target_labels'])==set('ABCD') and len(r['target_labels'])==4,'Malformed ordering task')


def bank_records(p,bank,root=BASE):
    b=p['banks'][bank];path=confined(b['manifest'],root);require(sha256_file(path)==b['manifest_sha256'],'Bank bytes changed')
    records=read_jsonl(path);validate_records(records,bank)
    require(hash_json(records)==b['selected_records_sha256'],'Selected bank records changed')
    source=confined(b['source_manifest'],root);require(sha256_file(source)==b['source_manifest_sha256'],'Original bank source changed')
    allrows=read_jsonl(source);require(records==(allrows[:256] if bank=='train256' else allrows),'Bank is not original fixed source/order')
    return records


def task_spec(p,task_id,phase):
    require(phase in ('pilot','formal'),'Only independent pilot/formal scoring modes')
    matches=[t for t in p['tasks'] if t['id']==task_id];require(len(matches)==1,'Unknown or cached-only task')
    t=matches[0];return t,2 if phase=='pilot' else t['rows']


def authorization_path(p,task_id,phase,root=BASE):
    task_spec(p,task_id,phase)
    return confined(p['output_directory'],root)/'authorizations'/(phase+'_'+task_id+'.json')


def validate_authorization(p,policy_path,task_id,phase,root=BASE):
    """A naked scorer command cannot load a model before persisted real gates."""
    task,count=task_spec(p,task_id,phase);output=confined(p['output_directory'],root)
    path=authorization_path(p,task_id,phase,root);a=load(path)
    require(a['policy_sha256']==sha256_file(policy_path) and a['task_id']==task_id and a['phase']==phase
            and a['rows']==count and a['pgid']==os.getpgrp(),'Scoring authorization scope/group differs')
    require(stamp(p['first_preflight_started_utc'])<=stamp(a['issued_utc'])<=datetime.now(timezone.utc)
            and stamp(a['issued_utc'])<stamp(p['absolute_deadline_utc']),'Authorization is outside original scientific clock')
    cpu_path=output/'cpu_preflight.json';cpu=load(cpu_path)
    require(a['cpu_preflight_sha256']==sha256_file(cpu_path) and cpu['passed'] is True
            and cpu['tests']==cpu['expected_tests']==p['expected_cpu_tests']
            and all(cpu[k]==0 for k in ('failures','errors','skipped','expected_failures','unexpected_successes'))
            and cpu['model_gpu_calls']==0 and cpu['torch_imported'] is False
            and cpu['native_watchdog_probe']['passed'] is True and cpu['policy_sha256']==sha256_file(policy_path)
            and cpu['sources_sha256']==p['sources_sha256']
            and cpu['first_preflight_started_utc']==p['first_preflight_started_utc']
            and cpu['absolute_deadline_utc']==p['absolute_deadline_utc'],'Actual complete CPU/native gate required before GPU')
    cached_path=output/'cached_initial_train'/'independent_verification.json';cached=load(cached_path)
    require(a['cached_initial_train_audit_sha256']==sha256_file(cached_path) and cached['passed'] is True
            and cached['rows']==256 and cached['phase']=='cached' and cached['task_id']=='initial_sft768__train256'
            and cached['policy_sha256']==sha256_file(policy_path)
            and cached['verifier_sha256']==p['sources_sha256']['research/endpoint_bank_20261106/verify_endpoint_scores.py'],
            'Independent cached source audit must precede GPU')
    require(stamp(a['issued_utc'])>=stamp(cpu['completed_utc'])
            and stamp(a['issued_utc'])>=stamp(cached['completed_utc']),'GPU authorization preceded CPU/cache audit completion')
    if phase=='formal':
        gate_path=output/'whole_nine_task_gate.json';gate=load(gate_path)
        require(a['whole_gate_sha256']==sha256_file(gate_path) and gate['passed'] is True
                and gate['formal_rows']==2024 and gate['pilot_rows_separate']==18
                and set(gate['pilot_receipts_sha256'])=={t['id'] for t in p['tasks']},'Complete all-nine formal gate required')
        for t in p['tasks']:
            receipt=output/'pilot'/t['id']/'independent_verification.json';r=load(receipt)
            require(sha256_file(receipt)==gate['pilot_receipts_sha256'][t['id']] and r['passed'] is True
                    and r['rows']==2 and r['phase']=='pilot' and r['task_id']==t['id']
                    and r['policy_sha256']==sha256_file(policy_path)
                    and r['verifier_sha256']==p['sources_sha256']['research/endpoint_bank_20261106/verify_endpoint_scores.py'],
                    'One pilot independent receipt changed/missing')
        require(stamp(a['issued_utc'])>=stamp(gate['completed_utc']),'Formal authorization preceded whole gate')
    else:require(a['whole_gate_sha256'] is None,'Pilot cannot pretend formal gate exists')
    return path


def validate_environment():
    require(ROOT==BASE and ROOT.stat().st_dev!=Path('/').stat().st_dev,'Existing separately mounted data disk required')
    require(Path(sys.prefix).resolve().is_relative_to(BASE) and Path(sys.executable).absolute().is_relative_to(BASE),'Python environment escaped data disk')
    require(all(os.environ.get(n)=='4' for n in THREADS),'All four thread environment variables must be4 before Python starts')
    required=('HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TMPDIR','TORCH_HOME')
    optional=('XDG_CACHE_HOME','PIP_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','HF_MODULES_CACHE','TRANSFORMERS_CACHE')
    for n in required:require(os.environ.get(n) and Path(os.environ[n]).resolve().is_relative_to(BASE),'Cache escaped/missing:'+n)
    for n in optional:require(not os.environ.get(n) or Path(os.environ[n]).resolve().is_relative_to(BASE),'Active cache escaped:'+n)


def whole_gate(pilots,tasks,remaining,free_bytes,historical):
    require(set(pilots)=={t['id'] for t in tasks} and len(pilots)==9,'All nine independently verified pilots required')
    require(historical['rows']==1024 and math.isfinite(historical['elapsed_seconds'])
            and historical['elapsed_seconds']>0,'Actual complete historical reference timing required')
    floor=historical['elapsed_seconds']/1024;loads=score_seconds=payload=0.;components={}
    for t in tasks:
        r=pilots[t['id']]
        require(r['passed'] is True and r['rows']==2 and len(r['row_seconds'])==2
                and all(math.isfinite(x) and x>0 for x in r['row_seconds']) and r['command_wall_seconds']>0,'Invalid actual pilot timing/coverage')
        task_score=r['row_seconds'][0]+(t['rows']-1)*max(r['row_seconds'][1],floor)
        score_seconds+=task_score
        loads+=max(0.,r['command_wall_seconds']-sum(r['row_seconds']))
        components[t['id']]={'first_row_once_seconds':r['row_seconds'][0],
            'second_row_observed_seconds':r['row_seconds'][1],'later_row_forecast_seconds':max(r['row_seconds'][1],floor),
            'formal_score_forecast_seconds':task_score,
            'model_load_hash_fingerprint_io_allowance_seconds':max(0.,r['command_wall_seconds']-sum(r['row_seconds']))}
        require(r['raw_bytes']>0,'Actual persisted pilot storage evidence required')
        payload+=t['rows']*r['raw_bytes']/2
    estimate=1.5*(loads+score_seconds)+600
    storage=math.ceil(2*payload+256*1024**2+1024**3)
    return {'passed':estimate<remaining and storage<=free_bytes,'estimated_seconds':estimate,
            'remaining_seconds':remaining,'formal_scoring_seconds':score_seconds,'nine_model_load_allowance_seconds':loads,
            'safety_factor':1.5,'validation_allowance_seconds':600,'required_free_bytes':storage,'actual_free_bytes':free_bytes,
            'historical_reference_timing':historical,'historical_per_row_floor_seconds':floor,'task_components':components,
            'pilot_timing_inputs':{key:{k:row[k] for k in ('passed','rows','row_seconds','raw_bytes','command_wall_seconds')} for key,row in pilots.items()},
            'all9_formal_tasks_required':True,'formal_rows':2024,'pilot_rows_separate':18,
            'forecast_limit':'Two-question resource estimate, not a duration upper bound or a causal decomposition of literal cost.',
            'policy':'No task dropping, score-based stopping, source change or clock reset.'}
