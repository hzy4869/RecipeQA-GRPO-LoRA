#!/usr/bin/env python3
"""Mandatory exact-count/zero-skip CPU tests, real Torch, and native subtree gate."""
from __future__ import annotations
import argparse
import importlib
import io
import json
from pathlib import Path
import sys
import unittest
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'exploratory/principled_ordering_20261105'))
from recipeqa.modeling import sha256_file,write_json
from coverage_contract import require
from launch_stage import native_watchdog_probe

MODULES=('test_coverage_contract','test_coverage_pipeline','test_verify_coverage_run','test_finite_policy_tensor_reference')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-tests',required=True,type=int)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    require(not args.output.exists() and args.expected_tests>0,'Fresh exact nonzero CPUreceipt required')
    suite=unittest.TestSuite();loader=unittest.TestLoader();sources={}
    for name in MODULES:
        module=importlib.import_module(name);suite.addTests(loader.loadTestsFromModule(module))
        sources[name]=sha256_file(module.__file__)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    receipt={'tests_run':result.testsRun,'expected_tests':args.expected_tests,'skipped':len(result.skipped),
        'failures':len(result.failures),'errors':len(result.errors),'suite_sources_sha256':sources,
        'passed':result.wasSuccessful() and result.testsRun==args.expected_tests and not result.skipped,
        'report':stream.getvalue(),'cuda_initialized':None,'native_watchdog_probe':None,'model_calls':0,'updates':0}
    try:
        require(receipt['passed'],'Exact nonvacuous suite must pass with zero skips/errors/failures')
        import torch
        receipt['cuda_initialized']=torch.cuda.is_initialized();require(receipt['cuda_initialized'] is False,'CPU preflight initialized CUDA')
        receipt['native_watchdog_probe']=native_watchdog_probe()
    except BaseException as error:
        receipt.update(passed=False,error=type(error).__name__+': '+str(error));write_json(args.output,receipt);print(stream.getvalue(),flush=True);raise
    write_json(args.output,receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
