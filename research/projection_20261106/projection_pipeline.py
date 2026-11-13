#!/usr/bin/env python3
"""DRAFT new bounded two-flow projection comparison; no existing run is resumed.

A new locked stage protocol and three pre-frozen child policies are mandatory.
This coordinator never selects a checkpoint, samples a reward-conditioned action,
changes data or invokes test/confirmation. Existing SFT/EK64 controls are historical.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'research/grounding_equivariance_20261106'))
from recipeqa.modeling import hash_json, model_weight_identity, read_jsonl, sha256_file, write_json
from orbit_diagnostic import (make_orbit, rescore, summarize, jsonl, remaining_budget,
    install_budget_alarm, cancel_budget_alarm, install_termination_handlers, restore_termination_handlers)
from run_cpu_preflight import validate_report

BASE=Path('/root/autodl-tmp/rl_recipeqa')
DEV_SHA='c998796fbc07a24c7a24062d6d0aba07500c75b34568fd7ba67f8c38daae03c9'
BETA=1.6363096678310884
FLOWS=('flow44','flow45')
HISTORICAL_ARMS=('flow44_raw_sft','flow44_exact_kl','flow45_raw_sft','flow45_exact_kl')
LOCK_NAMES=('formal_training_pipeline','evaluation_pipeline','evaluation_handoff','seed43_pipeline',
            'reward_search_20261105','finite_trials_20261105','orbit_diagnostic_20261106','projection_20261106')
THREADS=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def validate_stage_policy(policy, *, utc_now=None):
    require(policy['status']=='locked_before_projection_stage','Draft/unreviewed/unfunded stage cannot execute')
    require(policy['wall_limit_seconds']==14400 and policy['method']=='forward_projection'
            and policy['flow_order']==list(FLOWS) and policy['primary_checkpoint_step']==64,
            'Keep new4h/two-original-flow/fixed64 policy')
    require(policy['dev_manifest']=='data/manifests/val.jsonl' and policy['dev_manifest_sha256']==DEV_SHA
            and policy['orbit_first_recipes']==16 and policy['orbit_layouts']==24,
            'Only existing dev200 and its fixed first16 full24 orbit allowed')
    require(policy['selection']=='fixed_endpoint64_no_development_selection'
            and policy['test_access'] is False and policy['confirmation_access'] is False,
            'No checkpoint selection, held-out evaluation or decoder change')
    start=datetime.fromisoformat(policy['first_preflight_started_utc'].replace('Z','+00:00'))
    end=datetime.fromisoformat(policy['absolute_deadline_utc'].replace('Z','+00:00'))
    require(start.tzinfo is not None and end.tzinfo is not None and 0<(end-start).total_seconds()<=14400,
            'New absolute deadline must originate at first preflight and cannot be reset')
    current=datetime.now(timezone.utc) if utc_now is None else utc_now
    require(current.tzinfo is not None and start<=current and end>current,'Preflight start cannot be future-dated and new absolute budget must remain')
    require(set(policy['child_policies'])=={'smoke44','formal44','formal45'},'All three child policies must exist before first update')
    require(set(policy['historical'])==set(HISTORICAL_ARMS),'All four frozen historical controls required')
    return end


def validate_child_policy(child,key,stage_destination):
    require(child['status']=='locked_before_projection_training' and child['method']=='forward_projection'
            and child['reward']=='unique_complete_order_exact_0_1','Child is not independently locked')
    mode='smoke' if key=='smoke44' else 'formal'
    seed=44 if key in ('smoke44','formal44') else 45
    cli=child['cli']
    require(child['mode']==cli['mode']==mode and cli['seed']==seed and cli['beta']==BETA
            and cli['learning_rate']==5e-6,'Child mode/seed/fixedbeta/LR differs')
    require((cli['limit'],cli['max_steps'],cli['save_steps'])==((4,1,1) if mode=='smoke' else (256,64,16)),
            'Child exposure, update or checkpoint schedule differs')
    expected=Path(stage_destination)/('smoke44/train' if mode=='smoke' else 'flow'+str(seed)+'/train')
    require(Path(cli['output_dir'])==expected and not expected.exists(),'New exact child destination required; no warm start or resume')
    return cli


def validate_historical_training(signed,outcome,child,method,initial_identity):
    payload=signed['payload']; cli=payload['cli']; expected=child['cli']
    require(signed['sha256']==hash_json(payload),'Historical run signature invalid')
    require(payload['method']==cli['method']==method and payload['initial_identity']==initial_identity
            and payload['actual_updates']==64 and payload['accumulation']==4,
            'Historical control start/method/update budget differs')
    for field in ('manifest','expected_manifest_sha256','seed','limit','max_steps','learning_rate'):
        require(cli[field]==expected[field],'Historical training flow differs: '+field)
    require(cli['limit']==256 and payload['resolved_beta']==(BETA if method=='exact_kl' else 0),
            'Historical control exposure or objective differs')
    require(outcome['status']=='completed' and outcome['actual_prompt_visits']==256 and outcome['global_step']==64
            and outcome['run_signature_sha256']==signed['sha256'],'Historical control did not complete its signed budget')
    require(payload['reference_policy_sha256']==child['artifacts_sha256'][expected['reference_cache']],
            'Historical control and new forward arm use different fixed reference bytes')


def estimate_whole_stage(smoke_wall_seconds,smoke_step_seconds,historical_commands):
    require(math.isfinite(smoke_wall_seconds) and math.isfinite(smoke_step_seconds)
            and smoke_wall_seconds>=smoke_step_seconds>0,'Actual complete smoke timing required')
    formal=[command for command in historical_commands if command['stage'] in {'formal_'+arm for arm in HISTORICAL_ARMS}]
    require(len(formal)==4 and {command['stage'] for command in formal}=={'formal_'+arm for arm in HISTORICAL_ARMS}
            and all(command.get('returncode')==0 for command in formal),'Four distinct completed historical orbit timing commands required')
    first=[]; intervals=[]
    for command in formal:
        events=command['prediction_emissions']
        require(len(events)==384,'Historical timing must cover all384 actual predictions')
        arrival=[event['seconds_since_start'] for event in events]
        require(arrival[0]>0 and all(b>a for a,b in zip(arrival,arrival[1:])), 'Historical emission times must be monotonic')
        first.append(arrival[0]); intervals.extend(b-a for a,b in zip(arrival,arrival[1:]))
    per_record=max(intervals)
    overhead=smoke_wall_seconds-smoke_step_seconds
    components={'two_64step_trainings':128*smoke_step_seconds,
                'two_loads_plus_eight_checkpoint_allowances':10*overhead,
                'two_dev200_plus_two_orbit384':1168*per_record+4*max(first)}
    return {'estimated_seconds':1.5*sum(components.values())+600,'components_seconds':components,
            'safety_factor':1.5,'cpu_verification_allowance_seconds':600,
            'max_historical_preprocess_generation_write_interval_seconds':per_record,
            'smoke_step_seconds':smoke_step_seconds,'smoke_nonstep_seconds':overhead,
            'limits':'Runtime estimate only, never guarantees future tails. Fixed4h/absolute SIGALRM remains independent; fail gate before either formal arm rather than truncate or drop a low-score arm.'}


def greedy_summary(scored):
    require(len(scored)==200 and len({str(r['recipe_id']) for r in scored})==200,'Full dev200 required')
    return {metric:sum(row[metric] for row in scored)/200 for metric in ('exact','pair','valid')}


def paired_orbit(left,right):
    l={row['recipe_id']:row for row in left['per_recipe']}; r={row['recipe_id']:row for row in right['per_recipe']}
    require(l.keys()==r.keys() and len(l)==16,'Exact paired16 recipe denominator required')
    metrics=('K_over24','all24_exact','semantic_modal_consistency','pair','valid')
    differences=[{'recipe_id':key,**{metric:l[key][metric]-r[key][metric] for metric in metrics}} for key in sorted(l)]
    return {'direction':'forward_minus_historical_control','per_recipe':differences,
            'recipe_macro':{metric:sum(row[metric] for row in differences)/16 for metric in metrics}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True)
    args=parser.parse_args()
    policy_path=args.policy.resolve(); policy=load(policy_path)
    absolute=validate_stage_policy(policy)
    require(ROOT==BASE and ROOT.stat().st_dev!=Path('/').stat().st_dev,'Separate data disk required')
    require(Path(sys.prefix).resolve().is_relative_to(ROOT) and Path(sys.executable).absolute().is_relative_to(ROOT),'Python environment escaped data disk')
    for name in ('HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TMPDIR','TORCH_HOME'):
        require(os.environ.get(name) and Path(os.environ[name]).resolve().is_relative_to(ROOT),'Required cache/temp escaped data disk: '+name)
    for name in ('XDG_CACHE_HOME','PIP_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','HF_MODULES_CACHE','TRANSFORMERS_CACHE'):
        require(not os.environ.get(name) or Path(os.environ[name]).resolve().is_relative_to(ROOT),'Active cache escaped data disk: '+name)
    require(all(os.environ.get(name)=='4' for name in THREADS),'Reviewed four-thread environment required')
    destination=(ROOT/policy['output_directory']).resolve()
    require(destination.is_relative_to(ROOT/'exploratory/projection_20261106') and not destination.exists(),'Fresh stage only: no restart/resume/continuation')
    destination.mkdir(parents=True); (destination/'logs').mkdir()
    started=time.monotonic(); phase_deadline=started+14400
    policy_sha=sha256_file(policy_path)
    state={'status':'preflight','pid':os.getpid(),'started_utc':now(),'first_preflight_started_utc':policy['first_preflight_started_utc'],
           'policy_sha256':policy_sha,'wall_limit_seconds':14400,'absolute_deadline_utc':policy['absolute_deadline_utc'],
           'commands':[],'active_child_pid':None,'automatic_retry':False,'resume':False,'test_access':False,
           'confirmation_access':False,'primary_step':64,'historic_controls_disclosed':True}
    active=None; locks=[]; alarm_previous=None

    def confined(value):
        path=(ROOT/value).resolve()
        require(path.is_relative_to(ROOT),'Path escaped data disk: '+str(path))
        return path

    def remaining():
        return remaining_budget(phase_deadline,absolute)

    def update(**values):
        state.update(values,updated_utc=now()); write_json(destination/'state.json',state)

    def immutable():
        require(sha256_file(policy_path)==policy_sha,'Stage policy changed')
        for field in ('sources_sha256','artifacts_sha256'):
            for relative,digest in policy[field].items():
                require(sha256_file(confined(relative))==digest,'Frozen source/artifact changed: '+relative)
        for spec in policy['child_policies'].values():
            require(sha256_file(confined(spec['path']))==spec['sha256'],'Child policy changed after first preflight')

    def clear_alarm():
        nonlocal alarm_previous
        if alarm_previous is not None:
            cancel_budget_alarm(alarm_previous); alarm_previous=None

    def stop_child():
        nonlocal active
        if active and active.poll() is None:
            try: os.killpg(active.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: active.wait(timeout=4)
            except subprocess.TimeoutExpired:
                try: os.killpg(active.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                active.wait(timeout=4)

    def run(tag,argv,cpu=False):
        nonlocal active
        remaining(); immutable()
        entry={'stage':tag,'argv':list(map(str,argv)),'started_utc':now(),'returncode':None,'prediction_emissions':[]}
        state['commands'].append(entry); child_started=time.monotonic()
        env=dict(os.environ)
        if cpu: env['CUDA_VISIBLE_DEVICES']=''
        with (destination/'logs'/(tag+'.log')).open('x') as log:
            active=subprocess.Popen([sys.executable,'-u',*map(str,argv)],cwd=ROOT,env=env,stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
            stream=active.stdout; entry['pid']=active.pid
            def consume():
                for line in stream:
                    log.write(line); log.flush()
                    try:
                        value=json.loads(line)
                        if all(k in value for k in ('qid','presentation_id','raw_response','seconds')):
                            entry['prediction_emissions'].append({'qid':value['qid'],'presentation_id':value['presentation_id'],
                                                                  'seconds_since_start':time.monotonic()-child_started})
                    except (ValueError,TypeError): pass
            reader=threading.Thread(target=consume,daemon=True); reader.start(); update(active_child_pid=active.pid)
            try: entry['returncode']=active.wait(timeout=remaining())
            except BaseException:
                clear_alarm(); stop_child(); entry['returncode']=active.returncode; raise
            finally:
                reader.join(timeout=2); require(not reader.is_alive(),'Owned child stdout did not finish')
                stream.close(); entry.update(wall_seconds=time.monotonic()-child_started,finished_utc=now())
                active=None; update(active_child_pid=None)
        require(entry['returncode']==0,'Technical failure; no automatic retry: '+tag)
        return entry

    def predict(checkpoint,manifest,seed,tag):
        output=destination/tag/'predictions.jsonl'
        entry=run(tag,[confined('scripts/predict.py'),'--model-path',confined(policy['model_path']),'--revision',policy['revision'],
            '--adapter-path',checkpoint,'--manifest',manifest,'--image-root',confined(policy['image_root']),
            '--output',output,'--run-id',tag,'--seed',str(seed),'--temperature','0','--max-new-tokens','32',
            '--min-pixels','65536','--max-pixels','262144','--recipe-token-budget','1536','--max-input-tokens','4096'])
        config=load(str(output)+'.config.json')
        require(config['weight_identity']=={'base':base_identity,'adapter':model_weight_identity(checkpoint,adapter=True)}
                and config['input']==policy['input'] and config['manifest_sha256']==sha256_file(manifest), 'Prediction identity/input differs')
        coverage=destination/(tag+'_coverage.json'); orbit='orbit' in tag
        command=[confined('scripts/validate_run.py'),'predictions','--manifest',manifest,'--predictions',output,'--run-id',tag,
            '--checkpoint',checkpoint,'--seed',str(seed),'--expected-questions',str(16 if orbit else 200),
            '--expected-recipes',str(16 if orbit else 200),'--output',coverage]
        if orbit: command.append('--require-orbit')
        run(tag+'_coverage',command,cpu=True); require(load(coverage)['passed'] is True,'Frozen prediction coverage failed')
        scores=rescore(read_jsonl(manifest),read_jsonl(output),config['model_weights_sha256'])
        jsonl(destination/(tag+'_independent_scores.jsonl'),scores)
        return summarize(scores) if orbit else greedy_summary(scores)

    def trainer(key):
        spec=policy['child_policies'][key]; child=load(confined(spec['path'])); cli=child['cli']
        argv=[confined('research/projection_20261106/projection_train.py'),'--policy',confined(spec['path'])]
        for name,value in cli.items():
            if value is not None: argv += ['--'+name.replace('_','-'),str(value)]
        entry=run(key+'_train',argv)
        out=Path(cli['output_dir']); report=out/'independent_verification.json'
        run(key+'_verify',[confined('research/projection_20261106/verify_projection_run.py'),'--run-dir',out,
            '--manifest',cli['manifest'],'--expected-manifest-sha256',cli['expected_manifest_sha256'],
            '--initial-adapter',cli['adapter_path'],'--seed',str(cli['seed']),'--limit',str(cli['limit']),
            '--steps',str(cli['max_steps']),'--beta',str(cli['beta']),'--policy',confined(spec['path']),'--output',report],cpu=True)
        verified=load(report)
        require(verified['passed'] is True and verified['status']=='complete_verified_projection_run','Independent forward-specific verification failed')
        return out,entry

    previous_handlers=install_termination_handlers()
    try:
        alarm_previous=install_budget_alarm(remaining()); update(); immutable()
        required={'research/projection_20261106/projection_pipeline.py','research/projection_20261106/projection_train.py',
            'research/projection_20261106/projection_math.py','research/projection_20261106/verify_projection_run.py',
            'research/projection_20261106/run_cpu_preflight.py','research/projection_20261106/test_projection_torch_reference.py',
            'research/grounding_equivariance_20261106/orbit_diagnostic.py','scripts/predict.py','scripts/validate_run.py'}
        required.update(str(path.relative_to(ROOT)) for path in HERE.glob('test*.py'))
        require(required.issubset(policy['sources_sha256']),'Source closure omits coordinator/verifier/Torch reference/predictor')
        children={}
        for key,spec in policy['child_policies'].items():
            children[key]=load(confined(spec['path'])); validate_child_policy(children[key],key,destination)
        required_artifacts={str(confined(policy['dev_manifest'])),str(confined(policy['historical_orbit_state']))}
        for arm_spec in policy['historical'].values():
            for kind in ('dev','orbit'):
                path=str(confined(arm_spec[kind+'_predictions'])); required_artifacts.update((path,path+'.config.json'))
            for field in ('train_signature','train_outcome'):
                required_artifacts.add(str(confined(arm_spec[field])))
        for child in children.values():
            required_artifacts.update(str(confined(value)) for value in child['artifacts_sha256'])
        bound_artifacts={str(confined(value)) for value in policy['artifacts_sha256']}
        require(required_artifacts.issubset(bound_artifacts),'Stage must bind all historical raw/config/state, dev and child reference artifacts')
        require(all(child['initial_identity']==policy['initial_identity'] for child in children.values()),'All starts must be the same original SFT768')
        for name in LOCK_NAMES:
            handle=(ROOT/'metadata'/(name+'.lock')).open('a+')
            try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException: handle.close(); raise
            locks.append(handle)
        health=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=min(15,remaining()))
        gpu=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=min(15,remaining()))
        write_json(destination/'gpu_health.json',{'compute_returncode':health.returncode,'compute_stdout':health.stdout,'compute_stderr':health.stderr,
                     'gpu_returncode':gpu.returncode,'gpu_stdout':gpu.stdout,'gpu_stderr':gpu.stderr})
        require(health.returncode==gpu.returncode==0 and not health.stdout.strip(),'GPU idle/health query failed')
        require(len(gpu.stdout.strip().splitlines())==1,'Existing single GPU only')
        temperature,utilization,used,total=[float(v) for v in gpu.stdout.split(',')]
        require(temperature<85 and utilization<10 and used<1024 and total>20000,'GPU temperature/utilization/memory gate failed')
        # First actual operation is CPU-only test discovery, zero skips allowed.
        cpu_receipt=destination/'cpu_preflight.json'
        run('cpu_preflight',[HERE/'run_cpu_preflight.py','--expected-tests',str(policy['expected_cpu_tests']),'--output',cpu_receipt],cpu=True)
        require(validate_report(load(cpu_receipt),policy['expected_cpu_tests']),'CPU preflight did not pass fully without CUDA')
        base_identity=model_weight_identity(confined(policy['model_path']))
        require(base_identity==policy['initial_identity']['base'],'Base identity changed')
        initial_adapter=confined(children['formal44']['cli']['adapter_path'])
        require(model_weight_identity(initial_adapter,adapter=True)==policy['initial_identity']['adapter'],'Initial SFT identity changed')
        dev=confined(policy['dev_manifest']); orbit=make_orbit(read_jsonl(dev)); orbit_manifest=destination/'orbit16x24.jsonl'
        jsonl(orbit_manifest,orbit)
        history_state=load(confined(policy['historical_orbit_state']))
        require(history_state['status']=='complete_four_endpoint_orbits_development_only' and history_state['active_child_pid'] is None,
                'Prior orbit batch must be committed complete, never borrowed or continued')
        historical={}
        for arm in HISTORICAL_ARMS:
            spec=policy['historical'][arm]; checkpoint=confined(spec['checkpoint'])
            flow,method=arm.split('_',1)
            child=children['formal'+flow[-2:]]
            validate_historical_training(load(confined(spec['train_signature'])),load(confined(spec['train_outcome'])),child,method,policy['initial_identity'])
            adapter=model_weight_identity(checkpoint,adapter=True)
            require(spec['step']==64 and checkpoint.name=='checkpoint-64' and adapter==spec['identity'],'Historical fixed64 adapter changed')
            summaries={}
            for kind,manifest in [('dev',dev),('orbit',orbit_manifest)]:
                path=confined(spec[kind+'_predictions']); config=load(str(path)+'.config.json')
                require(config['weight_identity']=={'base':base_identity,'adapter':adapter} and config['input']==policy['input']
                        and config['manifest_sha256']==sha256_file(manifest) and config['cli']['seed']==int(flow[-2:]),'Historical prediction identity/input/seed differs')
                scores=rescore(read_jsonl(manifest),read_jsonl(path),config['model_weights_sha256'])
                summaries[kind]=summarize(scores) if kind=='orbit' else greedy_summary(scores)
            historical[arm]=summaries
        write_json(destination/'historical_controls_verified.json',{'controls':historical,'disclosure':'Previously trained and development-known fixed64 controls, not newly randomized simultaneous arms.'})
        update(status='smoke4q1step')
        smoke_out,smoke_entry=trainer('smoke44')
        smoke_steps=read_jsonl(smoke_out/'step_times.jsonl')
        require(len(smoke_steps)==1 and smoke_steps[0]['actual_prompt_visits']==4,'Smoke coverage incomplete')
        smoke_outcome=load(smoke_out/'train_outcome.json')
        require(smoke_outcome['candidate_backwards']==96 and smoke_outcome['current_policy_candidate_forwards_including_numerical_checks']==100,
                'Timing proxy must observe full24 backwards plus25 scoring forwards per smoke input; no sparse-smoke extrapolation')
        gate=estimate_whole_stage(smoke_entry['wall_seconds'],smoke_steps[0]['elapsed_seconds'],history_state['commands'])
        gate['remaining_seconds']=remaining(); gate['passed']=gate['estimated_seconds']<gate['remaining_seconds']
        write_json(destination/'whole_two_arm_timing_gate.json',gate)
        require(gate['passed'],'Whole two-formal-arm plus endpoint evaluations do not fit; no formal arm launched or dropped')
        results={}
        for flow,key in [('flow44','formal44'),('flow45','formal45')]:
            update(status='running_'+key)
            out,_=trainer(key); checkpoint=out/'checkpoint-64'; seed=children[key]['cli']['seed']
            results[flow]={'checkpoint':str(checkpoint),'step':64,
                'dev':predict(checkpoint,dev,seed,flow+'_forward_dev200'),
                'orbit':predict(checkpoint,orbit_manifest,seed,flow+'_forward_orbit384')}
            write_json(destination/(flow+'_result.json'),results[flow])
        remaining(); immutable()
        require(model_weight_identity(initial_adapter,adapter=True)==policy['initial_identity']['adapter']
                and model_weight_identity(confined(policy['model_path']))==base_identity,'Initial weights changed by new runs')
        comparisons={}
        for flow in FLOWS:
            comparisons[flow]={}
            for control in ('raw_sft','exact_kl'):
                old=historical[flow+'_'+control]
                comparisons[flow][control]={'dev_delta':{m:results[flow]['dev'][m]-old['dev'][m] for m in ('exact','pair','valid')},
                                          'orbit_delta':paired_orbit(results[flow]['orbit'],old['orbit'])}
        summary={'status':'completed_two_forward_endpoints_development_only','results':results,'historical_controls':historical,
                 'comparisons':comparisons,'selection':'fixed64 endpoint only, no score-based selection',
                 'training_compute':{flow:load(Path(results[flow]['checkpoint']).parent/'train_outcome.json') for flow in FLOWS},
                 'limits':['Historical SFT/EK controls and known development data; no simultaneous randomization or final confirmation.',
                           'Shared SFT initialization, two old flows; not three independent full-training seeds.',
                           '16 recipes with nested24 layouts are exploratory, not384 independent observations.',
                           'Fixed analytic RL soft-target supervised projection; do not claim novel RL, general superiority or SOTA.',
                           'Same visits/updates do not mean equal candidate compute or wall time.']}
        remaining(); write_json(destination/'summary.json',summary)
        receipts=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256_file(p)}
                  for p in sorted(destination.rglob('*')) if p.is_file() and p not in (destination/'state.json',destination/'artifacts_receipt.json')
                  and p.suffix not in ('.safetensors','.pt','.pth')]
        write_json(destination/'artifacts_receipt.json',{'files':receipts,'excluded_large_weights_optimizer_rng':'Covered by independent verifier and child outcome/checkpoint hashes, not duplicated in metadata export.'})
        remaining(); update(status=summary['status'],summary_sha256=sha256_file(destination/'summary.json'),wall_seconds=time.monotonic()-started)
    except BaseException as error:
        clear_alarm(); stop_child(); update(status='failed_or_budget_stopped_no_retry',error=type(error).__name__+': '+str(error),wall_seconds=time.monotonic()-started); raise
    finally:
        clear_alarm()
        for handle in reversed(locks): fcntl.flock(handle,fcntl.LOCK_UN); handle.close()
        restore_termination_handlers(previous_handlers)


if __name__=='__main__':
    main()
