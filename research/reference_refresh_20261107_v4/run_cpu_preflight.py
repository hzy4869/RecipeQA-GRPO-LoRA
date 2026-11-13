#!/usr/bin/env python3
"""Exact zero-skip CPU suite and actual bounded native descendant-cleanup test."""
from __future__ import annotations
import argparse
import importlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'exploratory/principled_ordering_20261105'))
from recipeqa.modeling import sha256_file,write_json
from refresh_contract import require,validate_reuse
from refresh_pipeline import validate_stage,now
from launch_stage import native_watchdog_probe
MODULES=('test_refresh_contract','test_refresh_pipeline','test_verify_refresh_run','test_finite_policy_tensor_reference')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--policy',required=True,type=Path)
    p.add_argument('--expected-tests',required=True,type=int);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();policy=json.loads(a.policy.read_text());validate_stage(policy)
    require(a.expected_tests==policy['expected_cpu_tests']>0 and not a.output.exists(),'Fresh exact nonzero CPU receipt required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preflight must explicitly hide GPU')
    require(os.environ.get('RECIPEQA_REFRESH_OUTER_WATCHDOG')==sha256_file(a.policy)
            and os.environ.get('RECIPEQA_REFRESH_PGID')==str(os.getpgrp()),'CPU preflight escaped native outer group')
    for name,digest in policy['sources_sha256'].items():require(sha256_file(ROOT/name)==digest,'Frozen CPU source changed:'+name)
    suite=unittest.TestSuite();sources={}
    for name in MODULES:
        module=importlib.import_module(name);suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        sources[name]=sha256_file(module.__file__)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    receipt={'status':'cpu_preflight_completed','tests_run':result.testsRun,'expected_tests':a.expected_tests,
        'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'expected_failures':len(result.expectedFailures),'unexpected_successes':len(result.unexpectedSuccesses),
        'suite_sources_sha256':sources,'report':stream.getvalue(),'cuda_initialized':None,'native_watchdog_probe':None,
        'policy_sha256':sha256_file(a.policy),'sources_sha256':policy['sources_sha256'],
        'first_preflight_started_utc':policy['first_preflight_started_utc'],'absolute_deadline_utc':policy['absolute_deadline_utc'],
        'pid':os.getpid(),'pgid':os.getpgrp(),'model_calls':0,'updates':0,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
    receipt['passed']=result.wasSuccessful() and result.testsRun==a.expected_tests and not any(receipt[k] for k in ('failures','errors','skipped','expected_failures','unexpected_successes'))
    try:
        require(receipt['passed'],'Exact full CPU suite must pass with zero skips/errors/failures')
        import torch
        receipt['cuda_initialized']=torch.cuda.is_initialized();require(receipt['cuda_initialized'] is False,'CPU gate initialized CUDA')
        receipt['native_watchdog_probe']=native_watchdog_probe()
        receipt['historical_reuse_verification']=validate_reuse(policy,ROOT,physical=True)
    except BaseException as exc:
        receipt.update(passed=False,error=type(exc).__name__+': '+str(exc),completed_utc=now())
        write_json(a.output,receipt);print(stream.getvalue(),flush=True);raise
    receipt['completed_utc']=now();write_json(a.output,receipt)
    print(json.dumps({k:receipt[k] for k in ('passed','tests_run','failures','errors','skipped','cuda_initialized')}),flush=True)


if __name__=='__main__':main()
