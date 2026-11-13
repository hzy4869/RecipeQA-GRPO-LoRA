#!/usr/bin/env python3
"""Native GNUtimeout for the single original24h clock, including CPU preflight."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from recipeqa.modeling import sha256_file
from refresh_contract import BASE,remaining_seconds,require


def stopped(pid):
    path=Path(f'/proc/{pid}/stat')
    if path.exists():
        try:return path.read_text().split(') ',1)[1].split()[0]=='Z'
        except FileNotFoundError:return True
    try:os.kill(pid,0)
    except ProcessLookupError:return True
    return False


def native_watchdog_probe():
    """Real bounded CPU child/grandchild TERM-ignore and native group-KILL proof."""
    executable=shutil.which('timeout');require(executable is not None,'GNUtimeout missing; cannot skip native cleanup proof')
    leaf="import os,signal,time,json;signal.signal(signal.SIGTERM,signal.SIG_IGN);print(json.dumps({'role':'grandchild','pid':os.getpid(),'pgid':os.getpgrp()}),flush=True);time.sleep(30)"
    parent="import os,signal,subprocess,sys,time,json;signal.signal(signal.SIGTERM,signal.SIG_IGN);print(json.dumps({'role':'child','pid':os.getpid(),'pgid':os.getpgrp()}),flush=True);subprocess.Popen([sys.executable,'-c',"+repr(leaf)+"]);time.sleep(30)"
    begun=time.monotonic();p=subprocess.Popen([executable,'--signal=TERM','--kill-after=1s','0.3s',sys.executable,'-u','-c',parent],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:stdout,stderr=p.communicate(timeout=5)
    except BaseException:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait(timeout=2);raise
    events=[json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
    require(len(events)==2 and {e['role'] for e in events}=={'child','grandchild'},'Actual child and grandchild required')
    require(all(e['pgid']==p.pid for e in events),'Native CPU probe descendants escaped test group')
    require(p.returncode in (-signal.SIGKILL,137) and time.monotonic()-begun<5,'Native KILL fallback not observed within bounded probe')
    require(all(stopped(e['pid']) for e in events),'TERM-ignoring descendant survived native deadline')
    return {'passed':True,'timeout_executable':executable,'returncode':p.returncode,'seconds':time.monotonic()-begun,
        'events':events,'all_descendants_stopped':True,'test_group_own_hard_limit_seconds':1.3,'model_gpu_calls':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',required=True,type=Path)
    a=parser.parse_args();path=a.policy.resolve();p=json.loads(path.read_text())
    require(ROOT==BASE and ROOT.stat().st_dev!=Path('/').stat().st_dev and path.is_relative_to(ROOT),'Separate existing data disk required')
    from refresh_pipeline import validate_stage
    validate_stage(p)
    for group in ('sources_sha256','artifacts_sha256'):
        for name,digest in p[group].items():
            source=(ROOT/name).resolve();require(source.is_relative_to(ROOT) and sha256_file(source)==digest,'Frozen dependency changed:'+name)
    for child in p['children'].values():require(sha256_file(ROOT/child['path'])==child['sha256'],'Frozen child policy changed')
    timeout=shutil.which('timeout');require(timeout,'Native GNUtimeout required')
    seconds=remaining_seconds(p);digest=sha256_file(path)
    os.environ['RECIPEQA_REFRESH_OUTER_WATCHDOG']=digest
    # GNUtimeout, coordinator, CPU and scientific descendants inherit this group.
    # GNUtimeout normally creates its own group equal to its PID before exec child.
    os.environ['RECIPEQA_REFRESH_PGID']=str(os.getpid())
    print(json.dumps({'status':'exec_native_reference_refresh_watchdog','policy_sha256':digest,
        'remaining_original_clock_seconds':seconds,'first_preflight_started_utc':p['first_preflight_started_utc'],
        'absolute_deadline_utc':p['absolute_deadline_utc'],'kill_after_seconds':10,'timeout':timeout,
        'scientific_children_same_process_group':True}),flush=True)
    os.execv(timeout,[timeout,'--signal=TERM','--kill-after=10s',f'{seconds:.6f}s',sys.executable,'-u',str(HERE/'refresh_pipeline.py'),'--policy',str(path)])


if __name__=='__main__':main()
