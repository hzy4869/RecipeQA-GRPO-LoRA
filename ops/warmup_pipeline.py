#!/usr/bin/env python3
"""Run the predeclared common-W preparation sequentially, stopping on failure.

This runs training-only diagnostics and development evaluation; no test or
formal branch is launched. Invoke once in the data-disk environment.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/rl_recipeqa'))
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to('/root/autodl-tmp'):
        raise ValueError('The project must remain on the data disk')
    os.chdir(root)
    sys.path.insert(0, str(root / 'src'))
    from recipeqa.modeling import model_weight_identity, write_json

    lock = (root / 'metadata/warmup_pipeline.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'running', 'pid': os.getpid(), 'commands': [],
             'started_at': datetime.now(timezone.utc).isoformat()}

    def update(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(root / 'metadata/warmup_pipeline.json', state)

    def run(stage, arguments):
        log = root / f'metadata/{stage}.log'
        entry = {'stage': stage, 'arguments': arguments, 'log': str(log)}
        state['commands'].append(entry)
        update(phase=stage)
        started = time.monotonic()
        with log.open('a') as stream:
            result = subprocess.run([sys.executable, *arguments], stdout=stream,
                                    stderr=subprocess.STDOUT)
        entry.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - started)
        update()
        if result.returncode:
            raise RuntimeError(f'{stage} exited {result.returncode}: {log}')

    model = str(root / 'models/Qwen3-VL-2B-Instruct')
    images = str(root / 'data/images')
    revision = '89644892e4d85e24eaac8bacfd4f463576704203'
    common = ['--model-path', model, '--revision', revision, '--image-root', images]
    try:
        outcome = json.loads((root / 'runs/pilot_sft_debug_v2/train_outcome.json').read_text())
        if outcome.get('global_step') != 20 or outcome.get('resume_global_step') != 18:
            raise RuntimeError('The diagnostic resume must finish before common-W preparation')
        debug_output = root / 'runs/pilot_sft_debug_v2_eval/predictions.jsonl'
        run('pilot_sft_debug_v2_eval', ['scripts/predict.py', *common,
            '--adapter-path', str(root / 'runs/pilot_sft_debug_v2/adapter_final'),
            '--manifest', str(root / 'data/manifests/debug.jsonl'),
            '--output', str(debug_output), '--run-id', 'pilot_sft_debug_v2_eval'])
        run('debug_before_after_summary', ['scripts/summarize_predictions.py',
            str(root / 'runs/base_debug/predictions.jsonl'), str(debug_output),
            '--paired', '--output-dir', str(root / 'results/debug_before_after')])
        warmup = root / 'runs/warmup_v1'
        run('warmup_v1', ['scripts/train_sft.py', *common,
            '--manifest', str(root / 'data/manifests/warmup.jsonl'),
            '--output-dir', str(warmup), '--max-steps', '64', '--save-steps', '16'])
        run('warmup_v1_validation', ['scripts/validate_run.py', 'training',
            '--run-dir', str(warmup), '--min-steps', '64',
            '--output', str(warmup / 'artifact_validation.json')])
        candidates, chosen = [], None
        for step in [16, 32, 48, 64]:
            checkpoint = warmup / f'checkpoint-{step}'
            run_id = f'warmup_dev_step_{step}'
            output = root / f'runs/{run_id}/predictions.jsonl'
            run(run_id, ['scripts/predict.py', *common, '--adapter-path', str(checkpoint),
                '--manifest', str(root / 'data/manifests/val.jsonl'),
                '--output', str(output), '--run-id', run_id])
            rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
            if len(rows) != 200 or any(r.get('technical_error') for r in rows):
                raise RuntimeError(f'{run_id} lacks 200 technical successes')
            run(f'{run_id}_coverage', ['scripts/validate_run.py', 'predictions',
                '--manifest', str(root / 'data/manifests/val.jsonl'), '--predictions', str(output),
                '--run-id', run_id, '--checkpoint', str(checkpoint), '--seed', '42',
                '--expected-questions', '200', '--expected-recipes', '200',
                '--output', str(output.parent / 'coverage.json')])
            candidate = {'step': step, 'checkpoint': str(checkpoint), 'n': len(rows),
                         'valid': sum(r['score']['valid'] for r in rows) / 200,
                         'exact': sum(r['score']['exact'] for r in rows) / 200,
                         'pair': sum(r['score']['pair'] for r in rows) / 200}
            candidates.append(candidate)
            update(evaluated_candidates=candidates)
            if candidate['valid'] >= .98:
                chosen = candidate
                break
        if chosen is None:
            chosen = max(candidates, key=lambda r: (r['valid'], -r['step']))
        selection = {'policy_file': str(root / 'metadata/w_starting_policy.json'),
                     'candidates': candidates, 'selected': chosen,
                     'status': 'selected' if chosen['valid'] >= .95 else 'below_minimum_valid',
                     'format_partially_stable': .95 <= chosen['valid'] < .98,
                     'selected_identity': model_weight_identity(chosen['checkpoint'], adapter=True),
                     'created_at': datetime.now(timezone.utc).isoformat()}
        write_json(root / 'metadata/w_selection.json', selection)
        if chosen['valid'] < .95:
            raise RuntimeError('W valid is below 95%; debug before launching formal RL')
        update(status='completed', phase='W_selected', selection=selection,
               next_action='Inspect evidence, run same-W SFT/Exact/Pair pilots; no formal branch launched')
    except BaseException as error:
        update(status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
