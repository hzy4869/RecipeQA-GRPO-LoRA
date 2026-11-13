#!/usr/bin/env python3
"""Nonvacuous CPU suite plus real bounded native process-group cleanup proof."""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent
from endpoint_bank_contract import confined,load,now,require,sha256_file,validate_policy
from recipeqa.modeling import write_json
from launch_endpoint_bank import native_watchdog_probe

MODULES=('test_endpoint_bank_contract','test_bank_math','test_verify_endpoint_scores','test_analyze_endpoint_bank')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-tests',type=int,required=True)
    parser.add_argument('--policy',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    p=load(a.policy);validate_policy(p);require(a.expected_tests==p['expected_cpu_tests'],'CPU suite count differs from locked policy')
    for name,digest in p['sources_sha256'].items():require(sha256_file(confined(name))==digest,'CPU source changed:'+name)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU gate must hide CUDA explicitly')
    require(not a.output.exists(),'Preserve prior CPU evidence; no overwrite')
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in MODULES)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    receipt={'status':'cpu_preflight_completed','tests':result.testsRun,'expected_tests':a.expected_tests,
        'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'expected_failures':len(result.expectedFailures),'unexpected_successes':len(result.unexpectedSuccesses),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch_imported':'torch' in sys.modules,
        'model_gpu_calls':0,'unittest_output':stream.getvalue(),'policy_sha256':sha256_file(a.policy),
        'first_preflight_started_utc':p['first_preflight_started_utc'],'absolute_deadline_utc':p['absolute_deadline_utc'],
        'sources_sha256':p['sources_sha256'],'source_sha256':{str((HERE/(n+'.py')).relative_to(HERE.parents[1])):sha256_file(HERE/(n+'.py')) for n in MODULES},'completed_utc':now()}
    receipt['passed']=result.wasSuccessful() and result.testsRun==a.expected_tests>0 and not any(receipt[k] for k in ('failures','errors','skipped','expected_failures','unexpected_successes')) and not receipt['torch_imported']
    write_json(a.output,receipt);require(receipt['passed'],'CPU suite failed/incomplete/skipped; preserve output')
    try:receipt['native_watchdog_probe']=native_watchdog_probe()
    except BaseException as exc:
        receipt.update(passed=False,native_watchdog_error=repr(exc),completed_utc=now());write_json(a.output,receipt);raise
    receipt['completed_utc']=now();write_json(a.output,receipt)
    print(json.dumps({k:receipt[k] for k in ('passed','tests','failures','errors','skipped','torch_imported')}),flush=True)


if __name__=='__main__':main()
