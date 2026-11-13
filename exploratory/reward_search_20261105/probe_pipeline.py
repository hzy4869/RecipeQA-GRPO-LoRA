"""Development-only stronger-SFT candidate probe; no training or test access."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/rl_recipeqa')
HERE = ROOT / 'exploratory/reward_search_20261105'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'ops'))
from recipeqa.modeling import sha256_file, write_json
from wait_and_evaluate import require_gpu_idle


def main():
    if ROOT.stat().st_dev == Path('/').stat().st_dev:
        raise RuntimeError('Project must be on the mounted data disk')
    os.chdir(ROOT)
    policy = json.loads((HERE / 'PROBE_PROTOCOL.json').read_text())
    initial = Path(policy['initial_adapter'])
    if sha256_file(initial / 'adapter_model.safetensors') != policy['initial_adapter_weight_sha256']:
        raise RuntimeError('Strong-SFT checkpoint changed')
    if sha256_file(ROOT / policy['manifest']) != policy['manifest_sha256']:
        raise RuntimeError('Development bank changed')
    locked = json.loads((ROOT / 'configs/protocol_lock.json').read_text())
    for name, digest in locked['file_sha256'].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError('Original frozen file changed: ' + name)
    locks = []
    for name in ['formal_training_pipeline', 'evaluation_pipeline', 'evaluation_handoff', 'seed43_pipeline', 'reward_search_20261105']:
        handle = (ROOT / ('metadata/' + name + '.lock')).open('a+')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    require_gpu_idle()
    path = HERE / 'probe_state.json'
    if path.exists():
        raise RuntimeError('Probe state exists; do not silently restart or overwrite')
    state = {'status': 'running', 'pid': os.getpid(), 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'policy_sha256': sha256_file(HERE / 'PROBE_PROTOCOL.json'), 'commands': [], 'test_access': False}
    write_json(path, state)
    for temperature in policy['temperatures']:
        tag = str(temperature).replace('.', 'p')
        output = HERE / ('probe_t' + tag + '/predictions.jsonl')
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, 'scripts/predict.py', '--model-path', str(ROOT / 'models/Qwen3-VL-2B-Instruct'),
            '--revision', '89644892e4d85e24eaac8bacfd4f463576704203', '--image-root', str(ROOT / 'data/images'),
            '--adapter-path', str(initial), '--manifest', str(ROOT / policy['manifest']), '--output', str(output),
            '--run-id', 'explore_strong_sft_probe_t' + tag, '--num-samples', '8', '--temperature', str(temperature), '--seed', str(policy['seed'])]
        entry = {'temperature': temperature, 'arguments': command, 'started_unix': time.time()}
        state['commands'].append(entry)
        log = HERE / ('probe_t' + tag + '.log')
        try:
            require_gpu_idle()
            with log.open('w') as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                state.update(active_child_pid=child.pid, phase='temperature_' + tag)
                write_json(path, state)
                entry['returncode'] = child.wait(timeout=policy['gpu_probe_wall_limit_seconds'])
            if entry['returncode']:
                raise RuntimeError('Probe failed; inspect ' + str(log))
            rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
            if len(rows) != 800 or any(r.get('technical_error') for r in rows):
                raise RuntimeError('Probe coverage is incomplete')
            entry.update(elapsed_seconds=time.time()-entry['started_unix'], output_sha256=sha256_file(output), rows=len(rows))
            state.update(active_child_pid=None)
            write_json(path, state)
        except BaseException as error:
            if 'child' in locals() and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
            state.update(status='failed', error=str(error), active_child_pid=None)
            write_json(path, state)
            raise
    state.update(status='completed_development_probe', phase='both_temperatures_complete', active_child_pid=None,
                 finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_json(path, state)


if __name__ == '__main__':
    main()
