#!/usr/bin/env python3
"""Outer GNU-timeout watchdog for the whole frozen read-only stage.

The timeout process supervises the Python runner and its CPU test subprocess in
one process group. Native CUDA calls cannot defer the final SIGKILL indefinitely.
No time is reset here; CPU preflight consumes the original frozen30min budget.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from recipeqa.modeling import sha256_file
from decomposition import require


def remaining_seconds(policy,utc_now=None):
    now=datetime.now(timezone.utc) if utc_now is None else utc_now
    start=datetime.fromisoformat(policy['first_preflight_started_utc'].replace('Z','+00:00'))
    absolute=datetime.fromisoformat(policy['absolute_deadline_utc'].replace('Z','+00:00'))
    require(start.tzinfo is not None and absolute.tzinfo is not None and now.tzinfo is not None,'Aware UTC clocks required')
    require(policy['wall_limit_seconds']==1800 and 0<(absolute-start).total_seconds()<=1800 and start<=now,'Frozen first-preflight30min clock required')
    remaining=min(1800-(now-start).total_seconds(),(absolute-now).total_seconds())
    require(remaining>0,'Original phase clock expired; never restart it')
    return remaining


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',required=True,type=Path)
    args=parser.parse_args();path=args.policy.resolve();policy=json.loads(path.read_text())
    require(ROOT==Path('/root/autodl-tmp/rl_recipeqa') and ROOT.stat().st_dev!=Path('/').stat().st_dev,'Separate remote data disk required')
    require(path.is_relative_to(ROOT) and policy['status']=='locked_before_paired_grounding_decomposition','Only separately reviewed frozen remote policy')
    for source,digest in policy['sources_sha256'].items():
        source_path=(ROOT/source).resolve();require(source_path.is_relative_to(ROOT) and sha256_file(source_path)==digest,'Reviewed source changed before watchdog: '+source)
    timeout=shutil.which('timeout');require(timeout is not None,'GNU timeout must already be installed; do not silently remove native watchdog')
    seconds=remaining_seconds(policy)
    print(json.dumps({'status':'exec_outer_timeout','utc':datetime.now(timezone.utc).isoformat(),
        'remaining_original_clock_seconds':seconds,'policy_sha256':sha256_file(path),
        'absolute_deadline_utc':policy['absolute_deadline_utc'],'timeout_executable':timeout,
        'signal':'TERM','kill_after_seconds':10,'native_cuda_watchdog':True,
        'cleanup_only_after_deadline':True}),flush=True)
    os.environ['RECIPEQA_GROUNDING_OUTER_WATCHDOG_POLICY_SHA256']=sha256_file(path)
    # GNU timeout without --foreground supervises a process group. The inherited
    # runner's own SIGALRM handles cooperative paths; this is the native fallback.
    os.execv(timeout,[timeout,'--signal=TERM','--kill-after=10s',f'{seconds:.6f}s',
        sys.executable,str(HERE/'run_decomposition.py'),'run','--policy',str(path)])


if __name__=='__main__':main()
