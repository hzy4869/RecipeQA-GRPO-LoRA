#!/usr/bin/env python3
"""Complete, fail-closed eight-arm queue. No score selection, retry or resume."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'research/grounding_equivariance_20261106'))
from recipeqa.modeling import hash_json,model_weight_identity,read_jsonl,sha256_file,write_json
from orbit_diagnostic import (make_orbit,jsonl,install_budget_alarm,cancel_budget_alarm,
    install_termination_handlers,restore_termination_handlers)
from refresh_contract import (BASE,ARMS,ARM_KINDS,FLOWS,PATH_SMOKES,WALL_SECONDS,TRAIN_SOURCE_PATHS,
    DEV_SHA,remaining_seconds,validate_child,materialize_reference,require)
from build_draft_policy import command_schedule,HISTORICAL_ARMS

LOCK_NAMES=('formal_training_pipeline','evaluation_pipeline','evaluation_handoff','seed43_pipeline',
 'reward_search_20261105','finite_trials_20261105','orbit_diagnostic_20261106','projection_20261106',
 'grounding_decomposition_20261106','coverage_20261106','endpoint_bank_20261106','reference_refresh_20261107')
THREADS=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
TECHNICAL_KEYS=tuple('smoke_'+a for a in ARMS)+PATH_SMOKES
FINAL_GUARD_BOOTSTRAP=("import os,sys; token=sys.stdin.buffer.read(1); "
    "token==b'1' or sys.exit('Final-guard parent release missing'); "
    "os.execv(sys.executable,[sys.executable,'-u',*sys.argv[1:]])")


def now():return datetime.now(timezone.utc).isoformat()
def load(path):return json.loads(Path(path).read_text())
def file_receipt(path):return {'path':str(path),'sha256':sha256_file(path)}


def require_science_commands_complete(state,schedule):
    """First73 may commit science artifacts, never claim the final74 audit."""
    require(len(schedule)==74 and schedule[-1]=='stage_global_verify','Final independent audit must be scheduled')
    require([r['stage'] for r in state['commands']]==schedule[:-1]
            and all(r['returncode']==0 for r in state['commands'])
            and state['active_child_pid'] is None,'All first73 commands must finish before global audit')


def validate_stage(p):
    require(p['status']=='locked_before_reference_refresh_stage','Draft stage cannot execute')
    require(p['wall_limit_seconds']==WALL_SECONDS and p['budget_cny_cap']==32.,'Announced20h/32CNY only')
    require(p['flow_order']==list(FLOWS) and p['methods']==list(ARM_KINDS)
            and p['formal_order']==list(ARMS) and p['technical_order']==list(TECHNICAL_KEYS),'All fixed eight and ten technical arms required')
    require(p['primary_step']==128 and p['boundary_step']==64,'Fixed128 endpoint and64 boundary only')
    require(p['command_schedule']==command_schedule(),'Exact complete command schedule changed')
    require(p['dev_manifest']=='data/manifests/val.jsonl' and p['dev_manifest_sha256']==DEV_SHA
            and p['orbit_first_recipes']==16 and p['orbit_layouts']==24,'Original known dev/orbit only')
    require(p['test_access'] is False and p['confirmation_access'] is False and p['no_resume'] is True
            and p['selection']=='fixed128_primary_step64_boundary_no_dev_selection','No heldout/checkpoint selection/resume')
    require(set(p['children'])==set(ARMS)|set(TECHNICAL_KEYS),'All18 frozen children required')
    require(set(p['sources_sha256'])==set(TRAIN_SOURCE_PATHS),'Complete reviewed source closure required')
    require(type(p['expected_cpu_tests']) is int and p['expected_cpu_tests']>0,'Nonvacuous frozen CPU count required')
    require(set(p['historical_training_runs'])==set(HISTORICAL_ARMS),'Four complete historical timing controls required')
    remaining_seconds(p)


def disk_estimate(checkpoint_bytes,adapter_bytes,free_bytes,technical_complete=False,continuity_bytes=None):
    require(checkpoint_bytes>adapter_bytes>0 and free_bytes>0,'Actual completed checkpoint sizes required')
    # A continuity file holds Adam moments + parameter gradients + RNG/mode state.
    # Prior full checkpoint contains Adam + adapter + RNG; one extra adapter is
    # a conservative initial bound, replaced upward by actual technical sizes.
    continuity=max(checkpoint_bytes+adapter_bytes,continuity_bytes or 0)
    checkpoints=16 if technical_complete else 28
    snapshots=4 if technical_complete else 8
    payload=checkpoints*checkpoint_bytes+snapshots*continuity+512*1024**2
    required=math.ceil(1.2*payload+1024**3)
    return {'actual_checkpoint_bytes':checkpoint_bytes,'actual_adapter_bytes':adapter_bytes,
        'continuity_file_bound_bytes':continuity,'formal_checkpoints':16,'technical_checkpoints':12,
        'formal_continuity_files':4,'technical_continuity_files':4,'final_adapter_exports':0,
        'remaining_checkpoint_count':checkpoints,'remaining_continuity_count':snapshots,
        'technical_artifacts_already_present':technical_complete,'metadata_reference_prediction_reserve_bytes':512*1024**2,
        'size_variation_factor':1.2,'free_margin_bytes':1024**3,
        'required_free_bytes':required,'actual_free_bytes':free_bytes,'passed':free_bytes>=required,
        'policy':'Preserve all existing/failed artifacts. No deletion, checkpoint reduction or arm dropping.'}


def historical_prediction_timing(state):
    require(state['status']=='complete_four_endpoint_orbits_development_only','Complete prior orbit timing required')
    rows=[r for r in state['commands'] if r['stage'].startswith('formal_') and not r['stage'].endswith('_coverage')]
    require(len(rows)==4 and {r['stage'] for r in rows}=={'formal_'+a for a in HISTORICAL_ARMS},'All four original orbit controls required')
    first=[];intervals=[]
    for r in rows:
        require(r['returncode']==0 and len(r['prediction_emissions'])==384,'Complete384 emission trace required')
        arrival=[x['seconds_since_start'] for x in r['prediction_emissions']]
        require(arrival[0]>0 and all(b>a for a,b in zip(arrival,arrival[1:])),'Monotonic prediction wall trace required')
        first.append(arrival[0]);intervals.extend(b-a for a,b in zip(arrival,arrival[1:]))
    return max(intervals),max(first)


def historical_training_timing(state,outcomes,steps,reference):
    require(state['status']=='completed_four_coverage_endpoints_development_only'
            and set(outcomes)==set(steps)==set(HISTORICAL_ARMS),'All four complete training wall controls required')
    commands={r['stage']:r for r in state['commands']}
    floors={'raw_sft':0.,'exact_kl':0.};verifier=0.;loads=0.
    for arm in HISTORICAL_ARMS:
        o=outcomes[arm];s=steps[arm];cmd=commands[arm+'_train'];v=commands[arm+'_verify']
        require(o['status']=='completed' and o['global_step']==256 and o['actual_prompt_visits']==1024
                and len(s)==256 and [r['global_step'] for r in s]==list(range(1,257)),'Full256update history only')
        require(cmd['returncode']==v['returncode']==0 and all(r['elapsed_seconds']>0 for r in s),'Successful actual history required')
        method=o['method'];require(method in floors,'Unknown historical method')
        floors[method]=max(floors[method],cmd['wall_seconds']/256)
        loads=max(loads,max(0.,cmd['wall_seconds']-sum(r['elapsed_seconds'] for r in s)))
        verifier=max(verifier,v['wall_seconds'])
    require(reference['status']=='completed_reference_only' and reference['rows']==1024
            and reference['updates']==0 and reference['reference_candidate_forwards_charged_this_run']==25600
            and reference['elapsed_seconds']>0,'Real completed1024 reference cost required')
    return {'full_run_step_floor_seconds':floors,'training_load_and_checkpoint_floor_seconds':loads,
        'verifier_wall_floor_seconds':verifier,'canonical_score_per_row_floor_seconds':reference['elapsed_seconds']/1024,
        'historical_q0_forward_cost':25600,'new_q0_forwards':0,'historical_reference_rows':1024}


def estimate_whole_stage(smokes,historical,prediction_seconds,load_seconds):
    require(set(smokes)==set(TECHNICAL_KEYS),'All ten independent technical passes required')
    require(prediction_seconds>0 and load_seconds>0,'Full historical prediction timing required')
    floor=historical['full_run_step_floor_seconds'];require(set(floor)=={'raw_sft','exact_kl'} and min(floor.values())>0,'Complete training floors required')
    score=historical['canonical_score_per_row_floor_seconds'];verify=historical['verifier_wall_floor_seconds']
    load=historical['training_load_and_checkpoint_floor_seconds'];per_arm={}
    for key,s in smokes.items():
        expected_steps=2 if key in PATH_SMOKES else 1;expected_visits=8 if key in PATH_SMOKES else 4
        require(s['verified'] is True and s['steps']==expected_steps and s['visits']==expected_visits
                and len(s['step_seconds'])==expected_steps and min(s['step_seconds'])>0
                and s['final_bank_rows']==4 and s['final_bank_seconds']>0 and s['wall_seconds']>0,'Incomplete technical training/timing')
        require(s['effective_adapter_update'] is True and len(s['actual_parameter_update_l2'])==expected_steps
                and all(math.isfinite(v) and v>0 for v in s['actual_parameter_update_l2']),
                'Technical admission requires a real effective update on every smoke/path step')
        score=max(score,s['final_bank_seconds']/4)
        verify=max(verify,s['verify_seconds'])
        if key in PATH_SMOKES:
            require(s['refresh_rows']==4 and s['refresh_seconds']>0,'Both actual refresh-path timings required')
            score=max(score,s['refresh_seconds']/4)
        else:require(s['refresh_rows']==0 and s['refresh_seconds']==0,'Unexpected ordinary-smoke refresh')
        load=max(load,max(0.,s['wall_seconds']-sum(s['step_seconds'])-s['final_bank_seconds']-s['refresh_seconds']))
    for arm in ARMS:
        method='raw_sft' if arm.endswith('_raw_sft') else 'exact_kl'
        technical=smokes['smoke_'+arm]['step_seconds']
        if arm.endswith('_refresh_beta'):technical=technical+smokes['path_smoke_'+arm]['step_seconds']
        per_arm[arm]=max(floor[method],max(technical))
    parts={'eight_128update_trainings':128*sum(per_arm.values()),'eight_training_loads_and_checkpoint_overhead':8*load,
        'two_actual256_reference_refreshes':512*score,'eight_fixed256_final_train_banks':2048*score,
        '4672_predictions_and_16_model_loads':4672*prediction_seconds+16*load_seconds,
        'eight_independent_run_verifications':8*verify}
    require(all(math.isfinite(v) and v>=0 for v in parts.values()),'Nonfinite resource estimate')
    return {'estimated_seconds':1.5*sum(parts.values())+600,'components_seconds':parts,'safety_factor':1.5,
        'fixed_cpu_and_validation_margin_seconds':600,'formal_step_seconds_by_arm':per_arm,
        'canonical_score_per_row_seconds':score,'prediction_per_record_seconds':prediction_seconds,
        'historical_floors':historical,'formal_visits':4096,'formal_updates':1024,
        'formal_reference_rows':512,'formal_final_train_bank_rows':2048,'formal_predictions':4672,
        'all_eight_formal_arms_required':True,
        'new_q0_forwards':0,'historical_shared_q0_forwards_charged_once':25600,
        'policy':'Max actual ten technical timings and complete historical run floors; no score-based stopping or smaller queue.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',required=True,type=Path)
    a=parser.parse_args();policy_path=a.policy.resolve();p=load(policy_path);validate_stage(p);digest=sha256_file(policy_path)
    require(os.environ.get('RECIPEQA_REFRESH_OUTER_WATCHDOG')==digest
            and os.environ.get('RECIPEQA_REFRESH_PGID')==str(os.getpgrp()),'Policy-bound native same-group watchdog required')
    require(Path(f'/proc/{os.getppid()}/exe').resolve().name=='timeout'
            and os.getpgid(os.getppid())==os.getpgrp()==os.getppid(),'Actual GNUtimeout parent/process group required')
    require(ROOT==BASE and ROOT.stat().st_dev!=Path('/').stat().st_dev,'Separate data disk required')
    require(Path(sys.prefix).resolve().is_relative_to(ROOT) and Path(sys.executable).absolute().is_relative_to(ROOT),'Environment escaped data disk')
    for name in ('HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TMPDIR','TORCH_HOME'):
        require(os.environ.get(name) and Path(os.environ[name]).resolve().is_relative_to(ROOT),'Required cache escaped data disk:'+name)
    for name in ('XDG_CACHE_HOME','PIP_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','HF_MODULES_CACHE','TRANSFORMERS_CACHE'):
        require(not os.environ.get(name) or Path(os.environ[name]).resolve().is_relative_to(ROOT),'Cache escaped data disk:'+name)
    require(all(os.environ.get(n)=='4' for n in THREADS),'Four-thread environment must precede this process')
    def confined(value):
        path=(ROOT/value).resolve();require(path.is_relative_to(ROOT),'Path escaped data disk:'+str(value));return path
    output=confined(p['output_directory']);require(output.is_relative_to(ROOT/'exploratory/reference_refresh_20261107') and not output.exists(),'Fresh stage only')
    output.mkdir(parents=True);(output/'logs').mkdir();(output/'authorizations').mkdir()
    state={'status':'preflight','pid':os.getpid(),'pgid':os.getpgrp(),'started_utc':now(),'policy_sha256':digest,
        'first_preflight_started_utc':p['first_preflight_started_utc'],'absolute_deadline_utc':p['absolute_deadline_utc'],
        'active_child_pid':None,'commands':[],'automatic_retry':False,'resume':False,'test_access':False,'confirmation_access':False}
    locks=[];active=None;alarm_previous=None;generated={};started=time.monotonic()
    def update(**values):state.update(values,updated_utc=now());write_json(output/'state.json',state)
    def immutable():
        require(sha256_file(policy_path)==digest,'Stage policy changed')
        for group in ('sources_sha256','artifacts_sha256'):
            for name,h in p[group].items():require(sha256_file(confined(name))==h,'Frozen source/artifact changed:'+name)
        for spec in p['children'].values():require(sha256_file(confined(spec['path']))==spec['sha256'],'Frozen child policy changed')
        for name,h in generated.items():require(sha256_file(name)==h,'Sealed generated evidence changed:'+name)
    def seal(path):generated[str(path)]=sha256_file(path)
    def clear_alarm():
        nonlocal alarm_previous
        if alarm_previous is not None:cancel_budget_alarm(alarm_previous);alarm_previous=None
    def stop_child():
        if active is not None and active.poll() is None:
            active.terminate()
            try:active.wait(timeout=4)
            except subprocess.TimeoutExpired:active.kill();active.wait(timeout=4)
    def run(tag,argv,cpu=False,start_barrier=False):
        nonlocal active
        require(tag==p['command_schedule'][len(state['commands'])],'Command outside fixed complete schedule')
        remaining_seconds(p);immutable();env=dict(os.environ)
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        entry={'stage':tag,'argv':list(map(str,argv)),'started_utc':now(),'returncode':None,'prediction_emissions':[]}
        state['commands'].append(entry);tick=time.monotonic()
        with (output/'logs'/(tag+'.log')).open('x') as log:
            # The final verifier must observe its own persisted PID before it
            # reads the mutable journal.  A one-byte pipe barrier keeps the
            # same PID/PGID and execs only after parent update() has returned.
            command=[sys.executable,'-u',*map(str,argv)]
            if start_barrier:
                require(tag=='stage_global_verify' and cpu,'Barrier reserved for final CPU guard')
                command=[sys.executable,'-u','-c',FINAL_GUARD_BOOTSTRAP,*map(str,argv)]
                entry['start_barrier']='one_byte_pipe_after_persisted_pid_then_same_pid_exec'
            active=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.PIPE if start_barrier else subprocess.DEVNULL,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            require(os.getpgid(active.pid)==os.getpgrp(),'Scientific child escaped owned native process group')
            entry.update(pid=active.pid,pgid=os.getpgid(active.pid));stream=active.stdout
            def consume():
                for line in stream:
                    log.write(line);log.flush()
                    try:
                        v=json.loads(line)
                        if all(k in v for k in ('qid','presentation_id','raw_response','seconds')):
                            entry['prediction_emissions'].append({'qid':v['qid'],'presentation_id':v['presentation_id'],'seconds_since_start':time.monotonic()-tick})
                    except (ValueError,TypeError):pass
            reader=threading.Thread(target=consume,daemon=True);reader.start();update(active_child_pid=active.pid)
            if start_barrier:
                active.stdin.write('1');active.stdin.flush();active.stdin.close()
            try:entry['returncode']=active.wait(timeout=remaining_seconds(p))
            except BaseException:clear_alarm();stop_child();entry['returncode']=active.returncode;raise
            finally:
                reader.join(timeout=2);require(not reader.is_alive(),'Child log stream did not close')
                stream.close();entry.update(wall_seconds=time.monotonic()-tick,finished_utc=now());active=None;update(active_child_pid=None)
        require(entry['returncode']==0,'Technical failure; preserve evidence, no retry:'+tag)
        return entry
    children={};q0_receipts={};technical={}
    def authorize(key):
        child=children[key];phase=child['mode'];path=confined(child['authorization_path'])
        require(not path.exists(),'No reused child authorization')
        auth={'schema_version':1,'child_key':key,'phase':phase,'stage_policy_path':str(policy_path),'stage_policy_sha256':digest,
            'child_policy_sha256':p['children'][key]['sha256'],'issued_utc':now(),
            'first_preflight_started_utc':p['first_preflight_started_utc'],'absolute_deadline_utc':p['absolute_deadline_utc'],
            'cpu_receipt':file_receipt(output/'cpu_preflight.json'),'q0_receipts':q0_receipts,'pgid':os.getpgrp()}
        require(len(q0_receipts)==4 and load(output/'cpu_preflight.json')['passed'],'CPU/four q0 audits must precede every model call')
        if phase=='formal':
            require(set(technical)==set(TECHNICAL_KEYS),'All ten independently passed technical branches required')
            auth.update(whole_gate_receipt=file_receipt(output/'whole_eight_arm_resource_gate.json'),technical_receipts=technical)
            require(load(output/'whole_eight_arm_resource_gate.json')['passed'],'Formal model call before complete resource gate')
        write_json(path,auth);seal(path)
    def train(key):
        authorize(key);child=children[key];path=confined(p['children'][key]['path'])
        command=run(key+'_train',[HERE/'refresh_train.py','--policy',path]);out=confined(child['cli']['output_dir'])
        report=out/'independent_verification.json';verification=run(key+'_verify',[HERE/'verify_refresh_run.py','--policy',path,'--run-dir',out,'--output',report],cpu=True)
        v=load(report);require(v['passed'] is True,'Independent child audit failed')
        require(v['mode']==child['mode'] and v['arm']==child['arm'] and v['flow']==child['flow'],'Wrong verification role')
        if child['mode']!='formal':require(v['effective_adapter_update'] is True,'Technical smoke made no effective adapter update')
        return out,command,verification
    def predict(checkpoint,manifest,seed,tag,orbit=False):
        raw=output/tag/'predictions.jsonl'
        run(tag,[ROOT/'scripts/predict.py','--model-path',confined(p['model_path']),'--revision',p['revision'],'--adapter-path',checkpoint,
            '--manifest',manifest,'--image-root',confined(p['image_root']),'--output',raw,'--run-id',tag,'--seed',str(seed),
            '--temperature','0','--max-new-tokens','32','--min-pixels','65536','--max-pixels','262144','--recipe-token-budget','1536','--max-input-tokens','4096'])
        config=load(str(raw)+'.config.json')
        require(config['weight_identity']=={'base':p['initial_identity']['base'],'adapter':model_weight_identity(checkpoint,adapter=True)}
                and config['input']==p['input'] and config['manifest_sha256']==sha256_file(manifest),'Prediction identity/input mismatch')
        coverage=output/(tag+'_coverage.json');argv=[ROOT/'scripts/validate_run.py','predictions','--manifest',manifest,'--predictions',raw,
            '--run-id',tag,'--checkpoint',checkpoint,'--seed',str(seed),'--expected-questions',str(16 if orbit else 200),
            '--expected-recipes',str(16 if orbit else 200),'--output',coverage]
        if orbit:argv.append('--require-orbit')
        run(tag+'_coverage',argv,cpu=True);require(load(coverage)['passed'] is True,'Complete prediction coverage failed')
        # No rescore/MAP/accuracy reads here. Full-stage analysis is separate.
    previous=install_termination_handlers()
    try:
        alarm_previous=install_budget_alarm(remaining_seconds(p));update();immutable()
        for key,spec in p['children'].items():
            child=load(confined(spec['path']));validate_child(child);children[key]=child
            require(child['sources_sha256']==p['sources_sha256'] and child['artifacts_sha256']==p['artifacts_sha256']
                and child['initial_identity']==p['initial_identity'] and child['first_preflight_started_utc']==p['first_preflight_started_utc']
                and child['absolute_deadline_utc']==p['absolute_deadline_utc'],'Child source/artifact/init/clock differs')
            expected=('smoke_' if child['mode']=='smoke' else 'path_smoke_' if child['mode']=='path_smoke' else '')+child['flow']+'_'+child['arm']
            require(key==expected and confined(child['cli']['output_dir']).is_relative_to(output),'Child role/output mismatch')
        for name in LOCK_NAMES:
            handle=(ROOT/'metadata'/(name+'.lock')).open('a+')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:handle.close();raise
            locks.append(handle)
        idle=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=min(15,remaining_seconds(p)))
        require(idle.returncode==0 and not idle.stdout.strip(),'GPU idle state not established')
        write_json(output/'gpu_idle_preflight.json',{'returncode':idle.returncode,'stdout':idle.stdout,'stderr':idle.stderr,'utc':now()})
        prior_checkpoint=ROOT/'exploratory/principled_ordering_20261105/finite_trials_v2/flow44/raw_sft/checkpoint-64'
        checkpoint_bytes=sum(x.stat().st_size for x in prior_checkpoint.rglob('*') if x.is_file())
        adapter_bytes=sum(x.stat().st_size for x in prior_checkpoint.glob('adapter_model*.safetensors'))
        storage=disk_estimate(checkpoint_bytes,adapter_bytes,shutil.disk_usage(ROOT).free)
        write_json(output/'initial_storage_gate.json',storage);require(storage['passed'],'Complete28checkpoint/8continuity budget fails; no deletion or smaller schedule')
        run('cpu_preflight',[HERE/'run_cpu_preflight.py','--policy',policy_path,'--expected-tests',str(p['expected_cpu_tests']),'--output',output/'cpu_preflight.json'],cpu=True)
        cpu=load(output/'cpu_preflight.json')
        require(cpu['passed'] and cpu['tests_run']==p['expected_cpu_tests'] and cpu['skipped']==cpu['failures']==cpu['errors']==0
                and cpu['cuda_initialized'] is False and cpu['native_watchdog_probe']['passed'],'Exact actual CPU/native gate required');seal(output/'cpu_preflight.json')
        dev=confined(p['dev_manifest']);orbit=output/'orbit16x24.jsonl';jsonl(orbit,make_orbit(read_jsonl(dev)));seal(orbit)
        for flow in FLOWS:
            for mode in ('smoke','formal'):
                c=children[('smoke_' if mode=='smoke' else '')+flow+'_raw_sft'];target=confined(c['cli']['reference_cache'])
                materialize_reference(confined(p['q0_source_cache']),confined(p['q0_source_manifest']),confined(c['cli']['manifest']),target)
                for suffix in ('','.config.json','.completion.json','.lineage.json'):seal(Path(str(target)+suffix))
        generated_receipt=output/'generated_reference_receipt.json'
        write_json(generated_receipt,{'status':'sealed_existing_actual_q0_before_any_model_call','files':dict(generated),
            'stage_policy_sha256':digest,'source_cache':p['q0_source_cache'],'source_cache_sha256':sha256_file(confined(p['q0_source_cache'])),
            'source_manifest_sha256':sha256_file(confined(p['q0_source_manifest'])),
            'source_signature_sha256':sha256_file(confined(p['q0_source_signature'])),
            'source_verification_sha256':sha256_file(confined(p['q0_source_verification'])),
            'historical_shared_q0_forwards_charged_once':25600,'new_q0_model_forwards':0,'optimizer_updates':0,
            'first_preflight_started_utc':p['first_preflight_started_utc'],'absolute_deadline_utc':p['absolute_deadline_utc'],'completed_utc':now()})
        seal(generated_receipt)
        for flow in FLOWS:
            for mode in ('smoke','formal'):
                key=('smoke_' if mode=='smoke' else '')+flow+'_raw_sft';audit=output/f'q0_{mode}_{flow}_verification.json'
                run(f'q0_{mode}_{flow}_verify',[HERE/'verify_refresh_run.py','--q0-only','--policy',confined(p['children'][key]['path']),'--output',audit],cpu=True)
                require(load(audit)['passed'] is True,'Independent q0 derivation audit failed');seal(audit)
                q0_receipts[mode+'_'+flow]=file_receipt(audit)
        smokes={}
        for key in TECHNICAL_KEYS:
            update(status='running_'+key);out,command,verification=train(key);v=load(out/'independent_verification.json')
            steps=read_jsonl(out/'step_times.jsonl');outcome=load(out/'train_outcome.json')
            bank=load(out/'final_train_bank/scores.jsonl.completion.json')
            boundary=children[key]['cli']['refresh_boundary_step'];event=load(out/f'refresh_at_step{boundary}/refresh_event.json') if boundary is not None else None
            smokes[key]={'verified':True,'steps':v['steps'],'visits':v['visits'],'step_seconds':[r['elapsed_seconds'] for r in steps],
                'effective_adapter_update':v['effective_adapter_update'],
                'actual_parameter_update_l2':[r['actual_parameter_update_l2'] for r in steps],
                'wall_seconds':command['wall_seconds'],'verify_seconds':verification['wall_seconds'],
                'final_bank_rows':bank['rows'],'final_bank_seconds':bank['elapsed_seconds'],
                'refresh_rows':event['rows'] if event else 0,'refresh_seconds':event['elapsed_seconds'] if event else 0,
                'independent_verification_sha256':sha256_file(out/'independent_verification.json')}
            technical[key]=file_receipt(out/'independent_verification.json');seal(out/'independent_verification.json')
        hstate=load(confined(p['historical_coverage_state']));runs=p['historical_training_runs']
        history=historical_training_timing(hstate,{k:load(confined(d)/'train_outcome.json') for k,d in runs.items()},
            {k:read_jsonl(confined(d)/'step_times.jsonl') for k,d in runs.items()},load(confined(p['historical_reference_outcome'])))
        timing=historical_prediction_timing(load(confined(p['historical_orbit_state'])))
        gate=estimate_whole_stage(smokes,history,*timing)
        actual_checkpoint=max([checkpoint_bytes]+[sum(f.stat().st_size for f in ck.rglob('*') if f.is_file())
            for key in TECHNICAL_KEYS for ck in confined(children[key]['cli']['output_dir']).glob('checkpoint-*')])
        continuity=max([0]+[f.stat().st_size for key in PATH_SMOKES for f in confined(children[key]['cli']['output_dir']).glob('refresh_at_step*/*.pt')])
        gate.update(remaining_seconds=remaining_seconds(p),storage=disk_estimate(actual_checkpoint,adapter_bytes,
            shutil.disk_usage(ROOT).free,technical_complete=True,continuity_bytes=continuity),technical_receipts=technical,
            completed_utc=now())
        gate['passed']=gate['estimated_seconds']<gate['remaining_seconds'] and gate['storage']['passed']
        write_json(output/'whole_eight_arm_resource_gate.json',gate);seal(output/'whole_eight_arm_resource_gate.json')
        require(gate['passed'],'Complete eight-arm gate failed; no formal launch, arm deletion or budget reset')
        for arm in ARMS:
            update(status='running_'+arm);out,_,_=train(arm);checkpoint=out/'checkpoint-128';seed=children[arm]['cli']['seed']
            predict(checkpoint,dev,seed,arm+'_step128_dev200')
            predict(checkpoint,orbit,seed,arm+'_step128_orbit384',orbit=True)
        remaining_seconds(p);immutable()
        require_science_commands_complete(state,p['command_schedule'])
        require(model_weight_identity(confined(p['model_path']))==p['initial_identity']['base']
                and model_weight_identity(confined(p['initial_adapter']),adapter=True)==p['initial_identity']['adapter'],'Initial weights changed')
        science={'status':'science_complete_pending_global_verification',
            'stage_policy_sha256':digest,'all_formal_arms':list(ARMS),'all_technical_arms':list(TECHNICAL_KEYS),
            'commands_completed':len(state['commands']),'eval_predictions':4672,'final_train_bank_rows':2048,
            'test_access':False,'confirmation_access':False,'score_selection':False,'scientific_metrics_read_by_coordinator':False,
            'completed_utc':now(),'wall_seconds':time.monotonic()-started}
        write_json(output/'science_completion.json',science);seal(output/'science_completion.json')
        update(status='running_stage_global_verify',science_completion_sha256=sha256_file(output/'science_completion.json'))
        global_report=output/'stage_global_verification.json'
        run('stage_global_verify',[HERE/'verify_refresh_run.py','--stage-only','--policy',policy_path,
            '--output',global_report],cpu=True,start_barrier=True)
        proof=load(global_report)
        require(proof['passed'] is True and proof['status']=='complete_reference_refresh_stage_analysis_gate_passed'
                and proof['locked_policy_sha256']==digest,'Actual complete independent global audit required')
        seal(global_report);remaining_seconds(p);immutable()
        require([r['stage'] for r in state['commands']]==p['command_schedule']
                and all(r['returncode']==0 for r in state['commands']),'All74 commands must complete before terminal state')
        complete={**science,'status':'completed_eight_reference_refresh_endpoints_independently_verified',
            'commands':len(state['commands']),'global_verification_path':str(global_report),
            'global_verification_sha256':sha256_file(global_report),
            'completed_utc':now(),'wall_seconds':time.monotonic()-started}
        write_json(output/'completion.json',complete);update(status=complete['status'],completion_sha256=sha256_file(output/'completion.json'))
    except BaseException as error:
        clear_alarm();stop_child();update(status='failed_or_budget_stopped_no_retry',error=type(error).__name__+': '+str(error));raise
    finally:
        clear_alarm()
        for handle in reversed(locks):fcntl.flock(handle,fcntl.LOCK_UN);handle.close()
        restore_termination_handlers(previous)


if __name__=='__main__':main()
