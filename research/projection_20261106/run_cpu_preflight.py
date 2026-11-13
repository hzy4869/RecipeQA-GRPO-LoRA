#!/usr/bin/env python3
"""Remote CPU-only discovery receipt; skips or any CUDA initialization fail closed."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from recipeqa.modeling import ensure_data_disk, write_json


def validate_report(report, expected):
    return (report['tests_run']==expected and report['successful'] is True
            and report['failures']==report['errors']==report['skipped']==0
            and report['cuda_initialized'] is False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-tests',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    ensure_data_disk(args.output)
    if args.output.exists():
        raise RuntimeError('Fresh CPU receipt required; no overwrite or retry')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU suite requires explicit CUDA_VISIBLE_DEVICES empty')
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test*.py',top_level_dir=str(HERE))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    import torch
    report={'tests_run':result.testsRun,'successful':result.wasSuccessful(),
            'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
            'cuda_initialized':torch.cuda.is_initialized(),'expected_tests':args.expected_tests,
            'scope':'actual remote CPU Torch tests; no model load, CUDA, training, data selection or SSH'}
    report['passed']=validate_report(report,args.expected_tests)
    write_json(args.output,report)
    print(json.dumps(report))
    if not report['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
