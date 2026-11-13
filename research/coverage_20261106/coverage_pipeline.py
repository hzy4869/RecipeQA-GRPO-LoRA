#!/usr/bin/env python3
"""Fixed four-arm coverage coordinator; no checkpoint/score selection or resume."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'research/grounding_equivariance_20261106'))
from recipeqa.modeling import hash_json,model_weight_identity,read_jsonl,sha256_file,write_json
from orbit_diagnostic import (make_orbit,jsonl,rescore,summarize as summarize_orbit,
    install_budget_alarm,cancel_budget_alarm,install_termination_handlers,restore_termination_handlers)
from coverage_contract import (BASE,ARMS,FLOWS,METHODS,WALL_SECONDS,TRAIN_SOURCE_PATHS,DEV_SHA,
    remaining_seconds,validate_child,materialize_reference,estimate_whole_stage,require)

LOCK_NAMES=('formal_training_pipeline','evaluation_pipeline','evaluation_handoff','seed43_pipeline',
 'reward_search_20261105','finite_trials_20261105','orbit_diagnostic_20261106','projection_20261106',
 'grounding_decomposition_20261106','coverage_20261106')
THREADS=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def now():return datetime.now(timezone.utc).isoformat()
def load(path):return json.loads(Path(path).read_text())


def validate_stage(policy):
    require(policy['status']=='locked_before_coverage_stage','Unreviewed/unfunded draft cannot execute')
    require(policy['wall_limit_seconds']==WALL_SECONDS and policy['budget_cny_cap']==25.6,'Announced16h/25.6CNY resourcecap required')
    require(policy['flow_order']==list(FLOWS) and policy['methods']==list(METHODS)
        and policy['primary_step']==256 and policy['diagnostic_step']==64,'Four fixed scientific arms/endpoints only')
    require(policy['dev_manifest']=='data/manifests/val.jsonl' and policy['dev_manifest_sha256']==DEV_SHA
        and policy['orbit_first_recipes']==16 and policy['orbit_layouts']==24,'Fixed known dev/orbit only')
    require(policy['test_access'] is False and policy['confirmation_access'] is False and policy['no_resume'] is True
        and policy['selection']=='fixed256_primary_fixed64_diagnostic_no_dev_selection','No heldout/checkpoint selection/retry')
    require(set(policy['children'])=={'reference',*ARMS,*('smoke_'+arm for arm in ARMS)},'Reference plus all four smokes/formal children must be frozen')
    require(set(policy['sources_sha256'])==set(TRAIN_SOURCE_PATHS),'Reviewed complete source closure required')
    require(type(policy['expected_cpu_tests']) is int and policy['expected_cpu_tests']>0,'Nonvacuous CPU count must be frozen')
    remaining_seconds(policy)


def disk_estimate(checkpoint_bytes,adapter_bytes,free_bytes):
    require(checkpoint_bytes>adapter_bytes>0 and free_bytes>0,'Actual prior checkpoint/storage measurements required')
    # 16formal checkpoints +4smoke, eight final adapter exports. Reserve512MiB
    # references/metadata,1GiB free margin,20% size variation; never delete olddata.
    payload=20*checkpoint_bytes+8*adapter_bytes+512*1024**2
    required=1.2*payload+1024**3
    return {'actual_prior_checkpoint_bytes':checkpoint_bytes,'actual_adapter_bytes':adapter_bytes,
        'formal_checkpoints':16,'smoke_checkpoints':4,'final_adapter_exports':8,
        'required_free_bytes':int(required),'actual_free_bytes':free_bytes,'passed':free_bytes>=required,
        'policy':'Preserve existing artifacts; no automatic deletion or save-cadence reduction.'}


def historical_prediction_timing(state):
    require(state['status']=='complete_four_endpoint_orbits_development_only','Complete prior orbit timing only')
    commands=[r for r in state['commands'] if r['stage'].startswith('formal_') and not r['stage'].endswith('_coverage')]
    require(len(commands)==4 and all(r['returncode']==0 and len(r['prediction_emissions'])==384 for r in commands),'All four full384 timing traces required')
    require({r['stage'] for r in commands}=={'formal_'+arm for arm in ARMS},'Four distinct fixed historical controls required for timing')
    first=[];intervals=[]
    for row in commands:
        arrival=[e['seconds_since_start'] for e in row['prediction_emissions']]
        require(arrival[0]>0 and all(b>a for a,b in zip(arrival,arrival[1:])),'Monotonic full timing events required')
        first.append(arrival[0]);intervals.extend(b-a for a,b in zip(arrival,arrival[1:]))
    return max(intervals),max(first)


def greedy_summary(scores):
    require(len(scores)==200 and len({r['recipe_id'] for r in scores})==200,'All200 independent recipe records required')
    return {k:sum(r[k] for r in scores)/200 for k in ('exact','pair','valid')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',required=True,type=Path)
    args=parser.parse_args();policy_path=args.policy.resolve();policy=load(policy_path);validate_stage(policy)
    policy_sha=sha256_file(policy_path)
    require(os.environ.get('RECIPEQA_COVERAGE_OUTER_WATCHDOG')==policy_sha,'Entire stage must use policy-bound native process-group watchdog')
    require(ROOT==BASE and ROOT.stat().st_dev!=Path('/').stat().st_dev,'Separate data disk required')
    require(Path(sys.prefix).resolve().is_relative_to(ROOT) and Path(sys.executable).absolute().is_relative_to(ROOT),'Environment escaped data disk')
    for name in ('HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TMPDIR','TORCH_HOME'):
        require(os.environ.get(name) and Path(os.environ[name]).resolve().is_relative_to(ROOT),'Required cache escaped data disk:'+name)
    for name in ('XDG_CACHE_HOME','PIP_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','HF_MODULES_CACHE','TRANSFORMERS_CACHE'):
        require(not os.environ.get(name) or Path(os.environ[name]).resolve().is_relative_to(ROOT),'Active cache escaped data disk:'+name)
    require(all(os.environ.get(n)=='4' for n in THREADS),'Four-thread environment must already be applied to this Python process')
    def confined(value):
        path=(ROOT/value).resolve();require(path.is_relative_to(ROOT),'Path escaped data disk:'+str(value));return path
    output=confined(policy['output_directory'])
    require(output.is_relative_to(ROOT/'exploratory/coverage_20261106') and not output.exists(),'Fresh stage only; no resume/retry')
    output.mkdir(parents=True);(output/'logs').mkdir()
    state={'status':'preflight','pid':os.getpid(),'pgid':os.getpgrp(),'started_utc':now(),'policy_sha256':policy_sha,
        'first_preflight_started_utc':policy['first_preflight_started_utc'],'absolute_deadline_utc':policy['absolute_deadline_utc'],
        'active_child_pid':None,'commands':[],'automatic_retry':False,'resume':False,'test_access':False,'confirmation_access':False}
    locks=[];active=None;alarm_previous=None;generated={};started=time.monotonic()
    def update(**values):state.update(values,updated_utc=now());write_json(output/'state.json',state)
    def immutable():
        require(sha256_file(policy_path)==policy_sha,'Stage policy changed')
        for group in ('sources_sha256','artifacts_sha256'):
            for path,digest in policy[group].items():require(sha256_file(confined(path))==digest,'Frozen source/artifact changed:'+path)
        for spec in policy['children'].values():require(sha256_file(confined(spec['path']))==spec['sha256'],'Frozen child policy changed')
        for path,digest in generated.items():require(sha256_file(path)==digest,'Sealed generated reference changed:'+path)
    def clear_alarm():
        nonlocal alarm_previous
        if alarm_previous is not None:cancel_budget_alarm(alarm_previous);alarm_previous=None
    def stop_child():
        if active is not None and active.poll() is None:
            active.terminate()
            try:active.wait(timeout=4)
            except subprocess.TimeoutExpired:active.kill();active.wait(timeout=4)
    def run(tag,argv,cpu=False):
        nonlocal active
        remaining_seconds(policy);immutable();env=dict(os.environ)
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        entry={'stage':tag,'argv':list(map(str,argv)),'started_utc':now(),'returncode':None,'prediction_emissions':[]}
        state['commands'].append(entry);tick=time.monotonic()
        with (output/'logs'/(tag+'.log')).open('x') as log:
            # Never detach: inherited GNUtimeout group includes every descendant.
            active=subprocess.Popen([sys.executable,'-u',*map(str,argv)],cwd=ROOT,env=env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            require(os.getpgid(active.pid)==os.getpgrp(),'Child escaped outerwatchdog process group')
            entry.update(pid=active.pid,pgid=os.getpgid(active.pid));stream=active.stdout
            def consume():
                for line in stream:
                    log.write(line);log.flush()
                    try:
                        value=json.loads(line)
                        if all(k in value for k in ('qid','presentation_id','raw_response','seconds')):
                            entry['prediction_emissions'].append({'qid':value['qid'],'presentation_id':value['presentation_id'],
                                'seconds_since_start':time.monotonic()-tick})
                    except (ValueError,TypeError):pass
            reader=threading.Thread(target=consume,daemon=True);reader.start();update(active_child_pid=active.pid)
            try:entry['returncode']=active.wait(timeout=remaining_seconds(policy))
            except BaseException:clear_alarm();stop_child();entry['returncode']=active.returncode;raise
            finally:
                reader.join(timeout=2);require(not reader.is_alive(),'Child output stream failed to close')
                stream.close();entry.update(wall_seconds=time.monotonic()-tick,finished_utc=now());active=None;update(active_child_pid=None)
        require(entry['returncode']==0,'Technical failure; no automatic retry:'+tag)
        return entry
    children={}
    def train(key,reference_only=False):
        child=children[key];path=confined(policy['children'][key]['path']);entry=run(key+'_train',[HERE/'coverage_train.py','--policy',path])
        out=confined(child['cli']['output_dir']);report=out/'independent_verification.json'
        argv=[HERE/'verify_coverage_run.py','--policy',path,'--run-dir',out,'--output',report]
        if reference_only:argv.append('--reference-only')
        run(key+'_verify',argv,cpu=True);verified=load(report)
        require(verified['passed'] is True,'Independent child/reference audit failed')
        return out,entry
    def predict(checkpoint,manifest,seed,tag,orbit=False):
        raw=output/tag/'predictions.jsonl'
        run(tag,[ROOT/'scripts/predict.py','--model-path',confined(policy['model_path']),'--revision',policy['revision'],
            '--adapter-path',checkpoint,'--manifest',manifest,'--image-root',confined(policy['image_root']),'--output',raw,
            '--run-id',tag,'--seed',str(seed),'--temperature','0','--max-new-tokens','32','--min-pixels','65536',
            '--max-pixels','262144','--recipe-token-budget','1536','--max-input-tokens','4096'])
        config=load(str(raw)+'.config.json');adapter=model_weight_identity(checkpoint,adapter=True)
        require(config['weight_identity']=={'base':policy['initial_identity']['base'],'adapter':adapter}
            and config['input']==policy['input'] and config['manifest_sha256']==sha256_file(manifest),'Prediction identity/input mismatch')
        coverage=output/(tag+'_coverage.json')
        argv=[ROOT/'scripts/validate_run.py','predictions','--manifest',manifest,'--predictions',raw,'--run-id',tag,
            '--checkpoint',checkpoint,'--seed',str(seed),'--expected-questions',str(16 if orbit else 200),
            '--expected-recipes',str(16 if orbit else 200),'--output',coverage]
        if orbit:argv.append('--require-orbit')
        run(tag+'_coverage',argv,cpu=True);require(load(coverage)['passed'] is True,'Complete prediction coverage failed')
        scores=rescore(read_jsonl(manifest),read_jsonl(raw),config['model_weights_sha256'])
        jsonl(output/(tag+'_independent_scores.jsonl'),scores)
        return summarize_orbit(scores) if orbit else greedy_summary(scores)
    previous=install_termination_handlers()
    try:
        alarm_previous=install_budget_alarm(remaining_seconds(policy));update();immutable()
        for key,spec in policy['children'].items():
            child=load(confined(spec['path']));validate_child(child);children[key]=child
            require(child['sources_sha256']==policy['sources_sha256'] and child['initial_identity']==policy['initial_identity']
                and child['first_preflight_started_utc']==policy['first_preflight_started_utc']
                and child['absolute_deadline_utc']==policy['absolute_deadline_utc'],'Children share full sources, init and firstclock')
            expected_key='reference' if child['mode']=='reference' else ('smoke_' if child['mode']=='smoke' else '')+child['flow']+'_'+child['method']
            require(key==expected_key and confined(child['cli']['output_dir']).is_relative_to(output),'Child role/output mismatch')
        for name in LOCK_NAMES:
            handle=(ROOT/'metadata'/(name+'.lock')).open('a+')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:handle.close();raise
            locks.append(handle)
        health=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=min(15,remaining_seconds(policy)))
        require(health.returncode==0 and not health.stdout.strip(),'GPU idle state not established')
        write_json(output/'gpu_idle_preflight.json',{'returncode':health.returncode,'stdout':health.stdout,'stderr':health.stderr,'utc':now()})
        old_checkpoint=ROOT/'exploratory/principled_ordering_20261105/finite_trials_v2/flow44/raw_sft/checkpoint-64'
        measured=sum(p.stat().st_size for p in old_checkpoint.rglob('*') if p.is_file())
        adapter=sum(p.stat().st_size for p in old_checkpoint.glob('adapter_model*.safetensors'))
        storage=disk_estimate(measured,adapter,shutil.disk_usage(ROOT).free);write_json(output/'storage_gate.json',storage)
        require(storage['passed'],'Full16formal+4smoke checkpoint budget does not fit; no deletion or cadence change')
        cpu=output/'cpu_preflight.json'
        run('cpu_preflight',[HERE/'run_cpu_preflight.py','--expected-tests',str(policy['expected_cpu_tests']),'--output',cpu],cpu=True)
        report=load(cpu);require(report['passed'] and report['tests_run']==policy['expected_cpu_tests'] and report['skipped']==report['failures']==report['errors']==0 and report['cuda_initialized'] is False,'Exact full zero-skip CPU preflight required')
        dev=confined(policy['dev_manifest']);orbit_manifest=output/'orbit16x24.jsonl';jsonl(orbit_manifest,make_orbit(read_jsonl(dev)))
        timing=historical_prediction_timing(load(confined(policy['historical_orbit_state'])))
        update(status='building_new_shared_reference1024')
        reference_out,_=train('reference',reference_only=True)
        source_cache=reference_out/'reference_policy.jsonl';source_manifest=confined(children['reference']['cli']['manifest'])
        for flow in FLOWS:
            for mode in ('smoke','formal'):
                if (flow,mode)==('flow44','formal'):continue
                child=children[('smoke_' if mode=='smoke' else '')+flow+'_raw_sft']
                target=confined(child['cli']['reference_cache'])
                materialize_reference(source_cache,source_manifest,confined(child['cli']['manifest']),target)
        for cache in {str(source_cache),*(child['cli']['reference_cache'] for child in children.values() if child['cli']['reference_cache'])}:
            for suffix in ('','.config.json','.completion.json','.lineage.json'):
                path=Path(cache+suffix)
                if path.exists():generated[str(path)]=sha256_file(path)
        write_json(output/'generated_reference_receipt.json',{'status':'sealed_before_first_smoke_update','files':generated,
            'stage_policy_sha256':policy_sha,
            'shared_initial_reference_model_forwards_charged_once':load(reference_out/'reference_only_outcome.json')['reference_candidate_forwards_charged_this_run'],
            'source_creation_run':str(reference_out),'selection':'deterministic identity-only transforms, no score inspection'})
        generated[str(output/'generated_reference_receipt.json')]=sha256_file(output/'generated_reference_receipt.json')
        smokes={}
        for arm in ARMS:
            update(status='running_smoke_'+arm)
            out,entry=train('smoke_'+arm);steps=read_jsonl(out/'step_times.jsonl')
            require(len(steps)==1 and steps[0]['actual_prompt_visits']==4,'Smoke must consume exactly4 before1update')
            smokes[arm]={'verified':True,'steps':1,'visits':4,'step_seconds':steps[0]['elapsed_seconds'],'wall_seconds':entry['wall_seconds']}
        gate=estimate_whole_stage(smokes,*timing);gate['remaining_seconds']=remaining_seconds(policy);gate['passed']=gate['estimated_seconds']<gate['remaining_seconds']
        write_json(output/'whole_four_arm_timing_gate.json',gate)
        require(gate['passed'],'All four formal arms/evaluations fail common timing gate; launch none, no shrinking or score-based arm dropping')
        results={}
        for arm in ARMS:
            update(status='running_'+arm);out,_=train(arm);seed=children[arm]['cli']['seed']
            result={'diagnostic64_dev':predict(out/'checkpoint-64',dev,seed,arm+'_step64_dev200'),
                'primary256_dev':predict(out/'checkpoint-256',dev,seed,arm+'_step256_dev200'),
                'primary256_orbit':predict(out/'checkpoint-256',orbit_manifest,seed,arm+'_step256_orbit384',orbit=True),
                'training_outcome':load(out/'train_outcome.json')}
            results[arm]=result;write_json(output/(arm+'_result.json'),result)
        remaining_seconds(policy);immutable()
        require(model_weight_identity(confined(policy['model_path']))==policy['initial_identity']['base']
            and model_weight_identity(confined(policy['initial_adapter']),adapter=True)==policy['initial_identity']['adapter'],'Initial weights changed')
        contrasts={flow:{m:results[flow+'_exact_kl']['primary256_dev'][m]-results[flow+'_raw_sft']['primary256_dev'][m] for m in ('exact','pair','valid')} for flow in FLOWS}
        summary={'status':'completed_four_coverage_endpoints_development_only','results':results,'within_flow_primary256_RL_minus_SFT':contrasts,
            'mean_two_flow_exact_difference':sum(c['exact'] for c in contrasts.values())/2,
            'selection':'fixed256_primary_fixed64_diagnostic_no_dev_selection','eval_predictions':3136,
            'limits':['Known dev, sharedSFT768 initialization and previouslyseen training pool; not independent full-training seeds.',
                '24candidate conditionalpolicy RL, samevisitedquestions/updates but unequalcandidatecompute; no free-token rollout or newreward.',
                '16recipes with24layouts nested, not384independentobservations; no confirmation/test evaluated.',
                'Follow frozen ANALYSIS_PLAN joint200recipe bootstrap, no score-based checkpoint selection or SOTA claim.']}
        write_json(output/'summary.json',summary)
        write_json(output/'artifacts_receipt.json',{'files':[{'path':str(p.relative_to(ROOT)),'sha256':sha256_file(p),'bytes':p.stat().st_size}
            for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('state.json','artifacts_receipt.json') and p.suffix not in ('.pt','.pth','.safetensors')],
            'large_binary_scope':'Actual state/weight bytes independently verified perchild; excluded from metadata archive only.'})
        remaining_seconds(policy);update(status=summary['status'],summary_sha256=sha256_file(output/'summary.json'),wall_seconds=time.monotonic()-started)
    except BaseException as error:
        clear_alarm();stop_child();update(status='failed_or_budget_stopped_no_retry',error=type(error).__name__+': '+str(error));raise
    finally:
        clear_alarm()
        for handle in reversed(locks):fcntl.flock(handle,fcntl.LOCK_UN);handle.close()
        restore_termination_handlers(previous)


if __name__=='__main__':main()
