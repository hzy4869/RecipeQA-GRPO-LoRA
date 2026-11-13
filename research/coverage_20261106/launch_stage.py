#!/usr/bin/env python3
"""Native16h deadline watchdog; child stages never create separate sessions."""
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
from coverage_contract import BASE,remaining_seconds,require


def stopped(pid):
    path=Path(f'/proc/{pid}/stat')
    if path.exists():
        # Zombie is no longer executing; init may reap its adopted process later.
        try:return path.read_text().split(') ',1)[1].split()[0]=='Z'
        except FileNotFoundError:return True  # init reaped it between stat/read
    try:os.kill(pid,0)
    except ProcessLookupError:return True
    return False


def native_watchdog_probe():
    """Real GNUtimeout TERM-ignoring parent+grandchild CPU cleanup proof.

    This isolated <=2second testgroup owns its own1second fallback. It never
    performs model work and its explicit timeout cannot survive stagecleanup.
    Scientific subprocesses remain in the outer stagegroup throughout.
    """
    timeout=shutil.which('timeout');require(timeout is not None,'GNUtimeout missing; native probe cannot be silently skipped')
    leaf="import os,signal,time,json;signal.signal(signal.SIGTERM,signal.SIG_IGN);print(json.dumps({'role':'grandchild','pid':os.getpid(),'pgid':os.getpgrp()}),flush=True);time.sleep(30)"
    parent="import os,signal,subprocess,sys,time,json;signal.signal(signal.SIGTERM,signal.SIG_IGN);print(json.dumps({'role':'child','pid':os.getpid(),'pgid':os.getpgrp()}),flush=True);subprocess.Popen([sys.executable,'-c',"+repr(leaf)+"]);time.sleep(30)"
    begun=time.monotonic();process=subprocess.Popen([timeout,'--signal=TERM','--kill-after=1s','0.3s',sys.executable,'-u','-c',parent],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:stdout,stderr=process.communicate(timeout=5)
    except BaseException:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=2);raise
    events=[json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
    require(len(events)==2 and {e['role'] for e in events}=={'child','grandchild'},'CPU watchdog probe must actually create child and grandchild')
    require(all(e['pgid']==process.pid for e in events),'GNUtimeout descendants escaped its owned processgroup')
    require(process.returncode in (-signal.SIGKILL,137) and time.monotonic()-begun<5,'Native KILL fallback did not fire within bounded probe')
    require(all(stopped(e['pid']) for e in events),'A TERM-ignoring grandchild survived native groupdeadline')
    return {'passed':True,'timeout_executable':timeout,'returncode':process.returncode,'seconds':time.monotonic()-begun,
        'events':events,'all_descendants_stopped':True,'test_group_own_hard_limit_seconds':1.3,'model_gpu_calls':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',required=True,type=Path)
    args=parser.parse_args();path=args.policy.resolve();policy=json.loads(path.read_text())
    require(ROOT==BASE and ROOT.stat().st_dev!=Path('/').stat().st_dev and path.is_relative_to(ROOT),'Separate existing data disk required')
    require(policy['status']=='locked_before_coverage_stage','Draft/unreviewed stage rejected')
    for name,digest in policy['sources_sha256'].items():
        source=(ROOT/name).resolve();require(source.is_relative_to(ROOT) and sha256_file(source)==digest,'Frozen source changed:'+name)
    timeout=shutil.which('timeout');require(timeout,'Native GNUtimeout required')
    seconds=remaining_seconds(policy);digest=sha256_file(path)
    os.environ['RECIPEQA_COVERAGE_OUTER_WATCHDOG']=digest
    print(json.dumps({'status':'exec_native_coverage_watchdog','policy_sha256':digest,
        'remaining_original_clock_seconds':seconds,'first_preflight_started_utc':policy['first_preflight_started_utc'],
        'absolute_deadline_utc':policy['absolute_deadline_utc'],'kill_after_seconds':10,'timeout':timeout,
        'scientific_children_same_process_group':True}),flush=True)
    os.execv(timeout,[timeout,'--signal=TERM','--kill-after=10s',f'{seconds:.6f}s',sys.executable,'-u',
        str(HERE/'coverage_pipeline.py'),'--policy',str(path)])


if __name__=='__main__':main()
