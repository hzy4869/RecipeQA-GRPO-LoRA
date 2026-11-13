#!/usr/bin/env python3
"""Resume data-disk preparation after live downloads, then run a tiny GPU smoke.

No formal training or protocol freeze is performed. All paths stay on the data
disk; dependency PIDs are observed rather than silently restarted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(payload, indent=2) + '\n')
    partial.replace(path)


def process_state(pid):
    proc = Path(f'/proc/{pid}')
    try:
        state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except FileNotFoundError:
        return {'pid': pid, 'state': 'missing', 'command': ''}
    return {'pid': pid, 'state': state, 'command': command}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/rl_recipeqa'))
    parser.add_argument('--max-wait-seconds', type=int, default=10800)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to('/root/autodl-tmp') or os.stat(root).st_dev == os.stat('/').st_dev:
        raise ValueError('A separately mounted /root/autodl-tmp data-disk root is required')
    os.chdir(root)
    if Path(sys.executable).resolve().is_relative_to('/root/autodl-tmp') is False:
        # venv python may be a symlink to the preinstalled interpreter; use
        # sys.prefix, the actual environment, to decide where packages are kept.
        if not Path(sys.prefix).resolve().is_relative_to(root):
            raise ValueError('Use the environment installed on the data disk')
    for variable in ['TMPDIR', 'HF_HOME', 'HF_HUB_CACHE', 'PIP_CACHE_DIR']:
        if not Path(os.environ[variable]).resolve().is_relative_to(root):
            raise ValueError(f'{variable} is not on the task data disk')

    status_path = root / 'metadata/preparation_pipeline.json'
    pipeline_lock = (root / 'metadata/preparation_pipeline.lock').open('a+')
    fcntl.flock(pipeline_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'started_at': timestamp(), 'pid': os.getpid(), 'status': 'running',
             'phase': 'observe_dependencies', 'root': str(root), 'commands': []}

    def update(**changes):
        state.update(changes, updated_at=timestamp())
        atomic_json(status_path, state)
        print(json.dumps({k: state.get(k) for k in ['updated_at', 'status', 'phase', 'error']}), flush=True)

    def wait_for(name, condition, pid_filename):
        started = time.monotonic()
        while not condition():
            pid = int((root / pid_filename).read_text())
            observation = process_state(pid)
            update(phase=f'wait_{name}', dependency=observation)
            if observation['state'] in ('missing', 'Z', 'X'):
                raise RuntimeError(f'{name} process is terminal but required completion evidence is absent: {observation}')
            if time.monotonic() - started > args.max_wait_seconds:
                raise TimeoutError(f'{name} is still unready; observe its original PID before any restart')
            time.sleep(10)

    def run(name, arguments):
        marker = root / 'metadata' / f'{name}.completed.json'
        # Markers are receipts only. Each stage's outputs are checked below.
        update(phase=name)
        log_path = root / 'metadata' / f'{name}.log'
        entry = {'stage': name, 'command': arguments, 'started_at': timestamp(), 'log': str(log_path)}
        state['commands'].append(entry)
        update()
        with log_path.open('a') as log:
            result = subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, check=False)
        entry.update(returncode=result.returncode, ended_at=timestamp())
        if result.returncode:
            raise RuntimeError(f'{name} exited {result.returncode}; inspect {log_path}')
        atomic_json(marker, entry)

    try:
        update()

        def terminal(pid_filename):
            pid = int((root / pid_filename).read_text())
            return process_state(pid)['state'] in ('missing', 'Z', 'X')

        wait_for('json_download', lambda: terminal('metadata/json_download.pid') and
                 all((root / f'data/raw/{split}.json').is_file() for split in ('train', 'val', 'test')),
                 'metadata/json_download.pid')

        def images_received():
            p = root / 'data/raw/images.zip'
            if p.is_file() and p.stat().st_size == 2271690314 and terminal('metadata/images_download.pid'):
                return True
            p = root / 'data/raw/images.zip.part'
            if not p.is_file() or p.stat().st_size != 2271690314:
                return False
            pid = int((root / 'metadata/images_download.pid').read_text())
            return process_state(pid)['state'] in ('missing', 'Z', 'X')

        wait_for('images_download', images_received, 'metadata/images_download.pid')
        run('verify_raw_assets', [sys.executable, 'ops/fetch_recipeqa.py', '--root', str(root), '--download-only'])
        run('extract_images', [sys.executable, 'ops/fetch_recipeqa.py', '--root', str(root), '--extract-only'])
        wait_for('environment_install', lambda: terminal('metadata/install.pid'), 'metadata/install.pid')
        run('environment_imports', [sys.executable, '-c',
            'import torch,transformers,trl,peft,accelerate,datasets; '
            'print({m.__name__:m.__version__ for m in [torch,transformers,trl,peft,accelerate,datasets]})'])
        run('prepare_manifests', [sys.executable, 'scripts/prepare_data.py', '--data-dir', str(root/'data/raw'),
             '--image-root', str(root/'data/images'), '--output-dir', str(root/'data/manifests')])
        run('unit_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
        run('input_audit_packet', [sys.executable, 'scripts/build_audit_packet.py', '--manifest',
             str(root/'data/manifests/audit.jsonl'), '--image-root', str(root/'data/images'),
             '--output-dir', str(root/'review/input_audit')])

        model = root/'models/Qwen3-VL-2B-Instruct'
        wait_for('model_download', lambda: (model/'DOWNLOAD_COMPLETE.json').is_file(), 'metadata/model_download.pid')

        run('base_smoke', [sys.executable, 'scripts/predict.py', '--model-path', str(model),
             '--manifest', str(root/'data/manifests/debug.jsonl'), '--image-root', str(root/'data/images'),
             '--output', str(root/'runs/base_smoke/predictions.jsonl'), '--run-id', 'base_smoke', '--limit', '2'])
        output = root/'runs/base_smoke/predictions.jsonl'
        rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
        if len(rows) != 2 or any(row.get('technical_error') for row in rows):
            raise RuntimeError('Base smoke does not have exactly two successful technical predictions')
        update(status='completed', phase='base_smoke_complete', next_action='Inspect smoke and run supervised/GRPO pilot; no formal training started')
    except BaseException as error:
        update(status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
