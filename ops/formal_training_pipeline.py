#!/usr/bin/env python3
"""Execute the frozen three-branch protocol and development selection only."""
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
        raise ValueError('All artifacts must stay on the data disk')
    os.chdir(root)
    sys.path.insert(0, str(root / 'src'))
    from recipeqa.modeling import model_weight_identity, sha256_file, write_json
    lock_handle = (root / 'metadata/formal_training_pipeline.lock').open('a+')
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((root / 'configs/protocol_lock.json').read_text())
    if protocol['status'] != 'frozen':
        raise RuntimeError('A verified protocol lock is required')
    for name, expected in protocol['file_sha256'].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f'Frozen source or manifest changed: {name}')
    w = protocol['W']['directory']
    if model_weight_identity(w, adapter=True) != protocol['W']:
        raise RuntimeError('Common W contents changed')
    state = {'status': 'running', 'pid': os.getpid(), 'commands': [], 'selected': {},
             'protocol_sha256': sha256_file(root / 'configs/protocol_lock.json'),
             'started_at': datetime.now(timezone.utc).isoformat()}

    def update(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(root / 'metadata/formal_training_pipeline.json', state)

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

    common = ['--model-path', str(root / 'models/Qwen3-VL-2B-Instruct'),
              '--revision', '89644892e4d85e24eaac8bacfd4f463576704203',
              '--image-root', str(root / 'data/images')]

    def train(name, method, manifest, steps, save_steps):
        output = root / f'runs/{name}'
        command = [f'scripts/train_{"sft" if method == "sft" else "grpo"}.py', *common,
                   '--manifest', str(root / manifest), '--output-dir', str(output),
                   '--initial-adapter', w, '--max-steps', str(steps),
                   '--save-steps', str(save_steps), '--seed', '42']
        if method != 'sft':
            command += ['--reward', method]
        run(name, command)
        run(f'{name}_validation', ['scripts/validate_run.py', 'training', '--run-dir', str(output),
            '--min-steps', str(steps), '--output', str(output / 'artifact_validation.json')])
        return output

    def select(name, directory, checkpoints):
        candidates = []
        for step in checkpoints:
            checkpoint = directory / f'checkpoint-{step}'
            run_id = f'{name}_dev_{step}'
            output = root / f'runs/dev/{run_id}/predictions.jsonl'
            run(run_id, ['scripts/predict.py', *common, '--adapter-path', str(checkpoint),
                '--manifest', str(root / 'data/manifests/val.jsonl'), '--output', str(output),
                '--run-id', run_id, '--seed', '42'])
            run(f'{run_id}_coverage', ['scripts/validate_run.py', 'predictions',
                '--manifest', str(root / 'data/manifests/val.jsonl'), '--predictions', str(output),
                '--run-id', run_id, '--checkpoint', str(checkpoint), '--seed', '42',
                '--expected-questions', '200', '--expected-recipes', '200',
                '--output', str(output.parent / 'coverage.json')])
            rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
            if len(rows) != 200 or len({r['recipe_id'] for r in rows}) != 200:
                raise RuntimeError('Development mean requires exactly one question per recipe')
            if any(row.get('technical_error') for row in rows):
                raise RuntimeError('Resolve all technical development failures before selection')
            candidates.append({'step': step, 'checkpoint': str(checkpoint), 'predictions': str(output),
                **{metric: sum(row['score'][metric] for row in rows) / 200
                   for metric in ['exact', 'pair', 'valid']}})
            update(development_candidates={**state.get('development_candidates', {}), name: candidates})
        chosen = max(candidates, key=lambda row: (row['exact'], row['pair'], -row['step']))
        result = {'selection_rule': 'recipe-macro Exact, then Pair, then earlier; fixed val200',
                  'candidates': candidates, 'selected': chosen,
                  'selected_identity': model_weight_identity(chosen['checkpoint'], adapter=True)}
        write_json(directory / 'best_checkpoint.json', result)
        state['selected'][name] = chosen
        update()
        return result

    try:
        main_sft = train('sft_seed42', 'sft', 'data/manifests/train_visits.jsonl', 256, 64)
        sft = select('sft_seed42', main_sft, [64, 128, 192, 256])
        candidates = sft['candidates']
        # Compare integer correct counts: >=4/200, avoiding float boundary drift.
        improvement = round((candidates[-1]['exact'] - candidates[-2]['exact']) * 200)
        triggered = sft['selected']['step'] == 256 and improvement >= 4
        write_json(main_sft / 'sufficiency_trigger.json', {
            'triggered': triggered, 'net_correct_256_minus_192': improvement,
            'policy': protocol['sft_sufficiency_policy'], 'test_predictions_used': False})
        if triggered:
            extended = train('sft_4pass_seed42', 'sft', 'data/manifests/train_visits_4pass.jsonl', 512, 128)
            select('sft_4pass_seed42', extended, [128, 256, 384, 512])
        for method in ['exact', 'pair']:
            name = f'{method}_seed42'
            output = train(name, method, 'data/manifests/train_visits.jsonl', 1024, 256)
            select(name, output, [256, 512, 768, 1024])
        update(status='completed', phase='three_branches_selected',
               next_action='Run locked E0-E4 evaluation; human labels remain pending if unavailable')
    except BaseException as error:
        update(status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
