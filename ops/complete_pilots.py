#!/usr/bin/env python3
"""Finish same-W SFT/Exact/Pair pilots and W bank; never launch formal runs."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/rl_recipeqa'))
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to('/root/autodl-tmp'):
        raise ValueError('Use the project data disk')
    os.chdir(root)
    sys.path.insert(0, str(root / 'src'))
    from recipeqa.modeling import write_json, model_weight_identity
    lock = (root / 'metadata/complete_pilots.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'running', 'pid': os.getpid(), 'commands': [],
             'started_at': datetime.now(timezone.utc).isoformat()}

    def update(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(root / 'metadata/complete_pilots.json', state)

    def run(stage, command):
        log = root / f'metadata/{stage}.log'
        entry = {'stage': stage, 'arguments': command, 'log': str(log)}
        state['commands'].append(entry)
        update(phase=stage)
        started = time.monotonic()
        with log.open('a') as stream:
            result = subprocess.run([sys.executable, *command], stdout=stream, stderr=subprocess.STDOUT)
        entry.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - started)
        update()
        if result.returncode:
            raise RuntimeError(f'{stage} exited {result.returncode}; inspect {log}')

    model = str(root / 'models/Qwen3-VL-2B-Instruct')
    selection = json.loads((root / 'metadata/w_selection.json').read_text())
    if selection['status'] != 'selected':
        raise RuntimeError('W must pass the predeclared selection gate')
    w = selection['selected']['checkpoint']
    if model_weight_identity(w, adapter=True) != selection['selected_identity']:
        raise RuntimeError('Selected W contents changed')
    common = ['--model-path', model, '--revision', '89644892e4d85e24eaac8bacfd4f463576704203',
              '--image-root', str(root / 'data/images')]
    visits = str(root / 'data/manifests/train_visits.jsonl')
    pair = root / 'runs/pilot_pair_w_v1'
    exact = root / 'runs/pilot_exact_w_v1'
    sft = root / 'runs/pilot_sft_w_v1'

    def train_command(method, output):
        command = [f'scripts/train_{"sft" if method == "sft" else "grpo"}.py', *common,
                   '--manifest', visits, '--limit', '160' if method == 'sft' else '40',
                   '--initial-adapter', w, '--output-dir', str(output),
                   '--max-steps', '20', '--save-steps', '2']
        if method != 'sft':
            command += ['--reward', method]
        return command

    def validate(name, directory):
        run(name, ['scripts/validate_run.py', 'training', '--run-dir', str(directory),
                   '--min-steps', '20', '--output', str(directory / f'{name}.json')])

    try:
        update(phase='observe_running_pair_pilot')
        pid = int((root / 'metadata/pilot_pair_w_v1.pid').read_text())
        started = time.monotonic()
        while True:
            process = Path(f'/proc/{pid}/stat')
            status = process.read_text().rsplit(')', 1)[-1].split()[0] if process.exists() else 'missing'
            if status in ('missing', 'Z', 'X'):
                break
            if time.monotonic() - started > 3600:
                raise TimeoutError('Original pair pilot still running; do not restart it')
            time.sleep(5)
        validate('pair_w_before_resume_validation', pair)
        run('pilot_pair_w_v1_resume', [*train_command('pair', pair), '--resume', str(pair / 'checkpoint-18')])
        validate('pair_w_after_resume_validation', pair)
        run('pair_w_resume_comparison', ['scripts/compare_resume.py', '--run-dir', str(pair),
            '--output', str(pair / 'resume_comparison.json')])
        run('pilot_exact_w_v1', train_command('exact', exact))
        validate('exact_w_validation', exact)
        run('pilot_sft_w_v1', train_command('sft', sft))
        validate('sft_w_before_resume_validation', sft)
        run('pilot_sft_w_v1_resume', [*train_command('sft', sft), '--resume', str(sft / 'checkpoint-18')])
        validate('sft_w_after_resume_validation', sft)
        run('sft_w_resume_comparison', ['scripts/compare_resume.py', '--run-dir', str(sft),
            '--output', str(sft / 'resume_comparison.json')])
        bank = root / 'runs/w_candidate_bank/predictions.jsonl'
        run('w_candidate_bank', ['scripts/predict.py', *common, '--adapter-path', w,
            '--manifest', str(root / 'data/manifests/bank.jsonl'), '--output', str(bank),
            '--run-id', 'w_candidate_bank', '--num-samples', '4', '--temperature', '1'])
        run('w_candidate_bank_validation', ['scripts/validate_run.py', 'predictions',
            '--manifest', str(root / 'data/manifests/bank.jsonl'), '--predictions', str(bank),
            '--run-id', 'w_candidate_bank', '--checkpoint', w, '--seed', '42',
            '--samples-per-input', '4', '--expected-questions', '100', '--expected-recipes', '100',
            '--output', str(bank.parent / 'coverage.json')])
        run('w_candidate_bank_summary', ['scripts/summarize_predictions.py', str(bank),
            '--candidate-bank', '--output-dir', str(root / 'results/e1_w_bank')])
        run('same_w_pilot_summary', ['scripts/summarize_pilot.py', '--run-dir',
            str(sft), str(exact), str(pair), '--output', str(root / 'results/same_w_pilot_summary.json')])
        update(status='completed', phase='core_pilots_complete',
               next_action='Inspect gradients/groups/resume/cost, complete sanity controls, freeze protocol before formal training')
    except BaseException as error:
        update(status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
