#!/usr/bin/env python3
"""Native GNUtimeout owns the shared scientific process group and3h deadline."""
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
HERE=Path(__file__).resolve().parent
from endpoint_bank_contract import (BASE,ROOT,confined,load,remaining_seconds,
    require,sha256_file,validate_environment,validate_policy)


def stopped(pid):
    try:
        stat=Path(f'/proc/{pid}/stat')
        if stat.exists():return stat.read_text().split(') ',1)[1].split()[0]=='Z'
        os.kill(pid,0)
    except (ProcessLookupError,FileNotFoundError):return True
    return False


def native_watchdog_probe():
    """CPU proof: TERM-ignoring child and grandchild are killed in one group."""
    timeout=shutil.which('timeout');require(timeout,'GNUtimeout missing; probe cannot be skipped')
    leaf="import os,signal,time,json;signal.signal(signal.SIGTERM,signal.SIG_IGN);print(json.dumps({'role':'grandchild','pid':os.getpid(),'pgid':os.getpgrp()}),flush=True);time.sleep(30)"
    parent="import os,signal,subprocess,sys,time,json;signal.signal(signal.SIGTERM,signal.SIG_IGN);print(json.dumps({'role':'child','pid':os.getpid(),'pgid':os.getpgrp()}),flush=True);subprocess.Popen([sys.executable,'-c',"+repr(leaf)+"]);time.sleep(30)"
    start=time.monotonic();p=subprocess.Popen([timeout,'--signal=TERM','--kill-after=1s','0.3s',sys.executable,'-u','-c',parent],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:stdout,stderr=p.communicate(timeout=5)
    except BaseException:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait(timeout=2);raise
    events=[json.loads(x) for x in stdout.splitlines() if x.startswith('{')]
    require(len(events)==2 and {x['role'] for x in events}=={'child','grandchild'},'Actual CPU child/grandchild proof missing')
    require(all(x['pgid']==p.pid for x in events),'Watchdog descendants escaped owned group')
    require(p.returncode in (-signal.SIGKILL,137) and time.monotonic()-start<5,'Native KILL fallback failed')
    require(all(stopped(x['pid']) for x in events),'A TERM-ignoring descendant survived')
    return {'passed':True,'events':events,'returncode':p.returncode,'seconds':time.monotonic()-start,
            'all_descendants_stopped':True,'model_gpu_calls':0,'probe_hard_limit_seconds':1.3}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',type=Path,required=True)
    a=parser.parse_args();path=a.policy.resolve();p=load(path)
    validate_policy(p);validate_environment();confined(path)
    for name,digest in p['sources_sha256'].items():require(sha256_file(confined(name))==digest,'Frozen source changed:'+name)
    timeout=shutil.which('timeout');require(timeout,'Native GNUtimeout is required')
    seconds=remaining_seconds(p);digest=sha256_file(path);os.environ['RECIPEQA_ENDPOINT_BANK_WATCHDOG']=digest
    print(json.dumps({'status':'exec_native_endpoint_bank_watchdog','policy_sha256':digest,
        'remaining_original_clock_seconds':seconds,'kill_after_seconds':10,'scientific_children_same_process_group':True}),flush=True)
    os.execv(timeout,[timeout,'--signal=TERM','--kill-after=10s',f'{seconds:.6f}s',sys.executable,'-u',
        str(HERE/'run_endpoint_bank.py'),'--policy',str(path)])


if __name__=='__main__':main()
