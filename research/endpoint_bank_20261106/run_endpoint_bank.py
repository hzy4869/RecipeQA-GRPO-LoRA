#!/usr/bin/env python3
"""All nine fixed likelihood tasks; independent pilot gates precede formal work."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
from endpoint_bank_contract import (BASE,ROOT,ENDPOINTS,authorization_path,bank_records,confined,load,now,
    remaining_seconds,require,sha256_file,validate_environment,validate_policy,whole_gate)
from recipeqa.modeling import model_weight_identity,write_json
from score_endpoint_bank import cached_initial_train,check_frozen_sources

LOCKS=('formal_training_pipeline','evaluation_pipeline','evaluation_handoff','seed43_pipeline',
 'reward_search_20261105','finite_trials_20261105','orbit_diagnostic_20261106','projection_20261106',
 'grounding_decomposition_20261106','coverage_20261106','endpoint_bank_20261106')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True)
    a=parser.parse_args();policy_path=a.policy.resolve();p=load(policy_path)
    validate_policy(p);validate_environment();check_frozen_sources(p);digest=sha256_file(policy_path)
    require(os.environ.get('RECIPEQA_ENDPOINT_BANK_WATCHDOG')==digest,'Native policy-bound outer watchdog required')
    require(Path(f'/proc/{os.getppid()}/exe').resolve().name=='timeout' and os.getpgrp()==os.getppid(),
            'Coordinator must be an actual child of GNUtimeout in its owned process group')
    os.environ['RECIPEQA_ENDPOINT_BANK_PGID']=str(os.getpgrp())
    output=confined(p['output_directory']);require(not output.exists(),'Fresh stage only; no resume or retry')
    output.mkdir(parents=True);(output/'logs').mkdir()
    state={'status':'preflight','pid':os.getpid(),'pgid':os.getpgrp(),'active_child_pid':None,
        'started_utc':now(),'policy_sha256':digest,'first_preflight_started_utc':p['first_preflight_started_utc'],
        'absolute_deadline_utc':p['absolute_deadline_utc'],'commands':[],'automatic_retry':False,
        'resume':False,'test_access':False,'confirmation_access':False,'optimizer_steps':0,'backward_calls':0,'generation_calls':0}
    started=time.monotonic();locks=[];active=None
    def update(**values):state.update(values,updated_utc=now());write_json(output/'state.json',state)
    def immutable():
        require(sha256_file(policy_path)==digest,'Stage policy changed');check_frozen_sources(p);remaining_seconds(p)
    def run(tag,argv,cpu=False):
        nonlocal active
        immutable();env=dict(os.environ)
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        row={'stage':tag,'argv':list(map(str,argv)),'started_utc':now(),'returncode':None};state['commands'].append(row)
        tick=time.monotonic()
        with (output/'logs'/(tag+'.log')).open('x') as log:
            active=subprocess.Popen([sys.executable,'-u',*map(str,argv)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            require(os.getpgid(active.pid)==os.getpgrp(),'Scientific child escaped owned native group')
            row.update(pid=active.pid,pgid=os.getpgid(active.pid));update(active_child_pid=active.pid)
            try:row['returncode']=active.wait(timeout=remaining_seconds(p))
            except BaseException:
                if active.poll() is None:active.terminate()
                try:active.wait(timeout=4)
                except subprocess.TimeoutExpired:active.kill();active.wait(timeout=4)
                row['returncode']=active.returncode;raise
            finally:
                row.update(finished_utc=now(),wall_seconds=time.monotonic()-tick);active=None;update(active_child_pid=None)
        require(row['returncode']==0,'Technical failure, preserve evidence, no automatic retry:'+tag)
        return row
    def verify(task,phase,directory):
        report=directory/'independent_verification.json'
        run(phase+'_'+task+'_verify',[HERE/'verify_endpoint_scores.py','--policy',policy_path,'--phase',phase,
            '--task-id',task,'--run-dir',directory,'--output',report],cpu=True)
        result=load(report);require(result['passed'] is True,'Independent likelihood audit failed')
        return result
    def authorize(task,phase):
        path=authorization_path(p,task['id'],phase);require(not path.exists(),'Authorization already exists; no retry')
        write_json(path,{'policy_sha256':digest,'task_id':task['id'],'phase':phase,
            'rows':2 if phase=='pilot' else task['rows'],'pgid':os.getpgrp(),'issued_utc':now(),
            'cpu_preflight_sha256':sha256_file(output/'cpu_preflight.json'),
            'cached_initial_train_audit_sha256':sha256_file(output/'cached_initial_train'/'independent_verification.json'),
            'whole_gate_sha256':sha256_file(output/'whole_nine_task_gate.json') if phase=='formal' else None})
    try:
        update()
        for name in LOCKS:
            handle=(BASE/'metadata'/(name+'.lock')).open('a+')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:handle.close();raise
            locks.append(handle)
        idle=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,
                            timeout=min(15,remaining_seconds(p)))
        require(idle.returncode==0 and not idle.stdout.strip(),'GPU idle state not established')
        write_json(output/'gpu_idle_preflight.json',{'returncode':idle.returncode,'stdout':idle.stdout,'stderr':idle.stderr,'utc':now()})
        run('cpu_preflight',[HERE/'run_cpu_preflight.py','--policy',policy_path,'--expected-tests',str(p['expected_cpu_tests']),
                            '--output',output/'cpu_preflight.json'],cpu=True)
        cpu=load(output/'cpu_preflight.json');require(cpu['passed'] and cpu['native_watchdog_probe']['passed'],'CPU/native gate not complete')
        base=model_weight_identity(confined(p['model_path']))
        physical={}
        for endpoint in ENDPOINTS:
            item=p['endpoints'][endpoint];actual={'base':base,'adapter':model_weight_identity(confined(item['adapter_path']),adapter=True)}
            require(actual==item['weight_identity'],'Missing/changed fixed endpoint:'+endpoint);physical[endpoint]=actual
        write_json(output/'all_five_physical_identities.json',physical)
        for bank in p['banks']:bank_records(p,bank)
        update(status='cached_initial_train');immutable();cached=output/'cached_initial_train'
        cached_initial_train(p,policy_path,cached);verify('initial_sft768__train256','cached',cached)
        pilots={}
        for task in p['tasks']:
            authorize(task,'pilot');update(status='pilot_'+task['id']);command=run('pilot_'+task['id'],[HERE/'score_endpoint_bank.py','--policy',policy_path,
                    '--task-id',task['id'],'--phase','pilot'])
            directory=output/'pilot'/task['id'];result=verify(task['id'],'pilot',directory)
            # Outer command wall includes model load, hashes and fingerprints.
            result={**result,'command_wall_seconds':command['wall_seconds']};pilots[task['id']]=result
        historical=p['historical_timing'];actual=load(confined(historical['reference_outcome']))
        require(sha256_file(confined(historical['reference_outcome']))==historical['sha256']
                and actual['rows']==historical['rows']==1024 and actual['status']=='completed_reference_only'
                and actual['updates']==0 and actual['elapsed_seconds']==historical['elapsed_seconds'],
                'Historical floor must come from the actual immutable completed reference outcome')
        immutable();gate=whole_gate(pilots,p['tasks'],remaining_seconds(p),shutil.disk_usage(BASE).free,historical)
        gate['pilot_receipts_sha256']={t['id']:sha256_file(output/'pilot'/t['id']/'independent_verification.json') for t in p['tasks']}
        gate['completed_utc']=now();write_json(output/'whole_nine_task_gate.json',gate)
        require(gate['passed'],'All-nine timing/storage gate failed; no formal task may start')
        for task in p['tasks']:
            authorize(task,'formal');update(status='formal_'+task['id']);run('formal_'+task['id'],[HERE/'score_endpoint_bank.py','--policy',policy_path,
                    '--task-id',task['id'],'--phase','formal'])
            verify(task['id'],'formal',output/'formal'/task['id'])
        immutable();update(status='completed_endpoint_bank_zero_update',formal_tasks=9,formal_rows=2024,
            pilot_tasks=9,pilot_rows=18,cached_initial_train_rows=256,formal_candidate_forwards=50600,
            pilot_candidate_forwards=450,candidate_forwards=51050,cached_new_candidate_forwards=0,
            model_loads=18,wall_seconds=time.monotonic()-started)
    except BaseException as exc:
        update(status='failed_preserved_no_retry',error=repr(exc),wall_seconds=time.monotonic()-started)
        raise
    finally:
        for handle in reversed(locks):fcntl.flock(handle,fcntl.LOCK_UN);handle.close()


if __name__=='__main__':main()
