"""Run a bounded, development-only finite-action probability diagnostic."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'ops'))
from recipeqa.modeling import sha256_file, write_json
from wait_and_evaluate import require_gpu_idle


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    os.chdir(ROOT)
    if not ROOT.is_relative_to('/root/autodl-tmp') or ROOT.stat().st_dev == Path('/').stat().st_dev:
        raise RuntimeError('Diagnostics must remain on the mounted data disk')
    policy_path = HERE / 'DIAGNOSTIC_PROTOCOL.json'
    policy = json.loads(policy_path.read_text())
    policy_sha = sha256_file(policy_path)
    if policy['status'] != 'locked_before_new_gpu_diagnostics':
        raise RuntimeError('Diagnostic policy is not locked')
    expected_sources = {
        HERE / 'score_permutations.py': policy['scorer_source_sha256'],
        ROOT / 'src/recipeqa/metrics.py': policy['metric_source_sha256'],
        ROOT / 'src/recipeqa/modeling.py': policy['modeling_source_sha256'],
        Path(__file__): sha256_file(__file__),
    }
    for path, digest in expected_sources.items():
        if sha256_file(path) != digest:
            raise RuntimeError('Diagnostic source changed: ' + str(path))
    if sha256_file(ROOT / policy['manifest']) != policy['manifest_sha256']:
        raise RuntimeError('Development manifest changed')
    destination = HERE / 'diagnostics'
    if destination.exists():
        raise RuntimeError('Diagnostic output exists; inspect instead of automatic restart')
    destination.mkdir()
    state_path = destination / 'state.json'
    state = {'status': 'waiting_exact_completion', 'pid': os.getpid(), 'started_utc': now(),
             'commands': [], 'policy_sha256': policy_sha, 'test_access': False,
             'training': False, 'source_sha256': {str(p.relative_to(ROOT)): h for p, h in expected_sources.items()}}
    write_json(state_path, state)
    locks = []
    child = None
    try:
        waiting = time.monotonic()
        prior_path = HERE / 'exact_completion/state.json'
        while True:
            if prior_path.exists():
                prior = json.loads(prior_path.read_text())
                if prior['status'] == 'completed_exact_only':
                    break
                if prior['status'] == 'failed_stopped_no_retry':
                    raise RuntimeError('Exact completion failed; inspect before any diagnostic GPU work')
            if time.monotonic() - waiting > 2100:
                raise RuntimeError('Timed out waiting for the bounded Exact-only completion')
            time.sleep(10)
        for name in ('formal_training_pipeline', 'evaluation_pipeline', 'evaluation_handoff', 'seed43_pipeline', 'reward_search_20261105'):
            handle = (ROOT / f'metadata/{name}.lock').open('a+')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
        require_gpu_idle()
        deadline = time.monotonic() + policy['diagnostic_wall_cap_seconds']
        state.update(status='running', exact_completion_sha256=sha256_file(prior_path))
        write_json(state_path, state)
        models = {row['name']: row for row in policy['model_order']}
        stages = [('sft256', 2), ('sft256', 32), ('pair1024', 32), ('W16', 32), ('sft256', 200), ('pair1024', 200)]
        for name, limit in stages:
            if sha256_file(policy_path) != policy_sha:
                raise RuntimeError('Locked diagnostic policy changed')
            for path, digest in expected_sources.items():
                if sha256_file(path) != digest:
                    raise RuntimeError('Diagnostic source changed after launch')
            require_gpu_idle()
            model = models[name]
            if model.get('weight_sha256') and sha256_file(ROOT / model['adapter'] / 'adapter_model.safetensors') != model['weight_sha256']:
                raise RuntimeError('Initial model weights changed')
            tag = f'{name}_dev{limit}'
            output = destination / tag / 'scores.jsonl'
            output.parent.mkdir()
            command = [sys.executable, str(HERE / 'score_permutations.py'),
                '--model-path', str(ROOT / 'models/Qwen3-VL-2B-Instruct'), '--revision', '89644892e4d85e24eaac8bacfd4f463576704203',
                '--adapter-path', str(ROOT / model['adapter']), '--image-root', str(ROOT / 'data/images'),
                '--manifest', str(ROOT / policy['manifest']), '--greedy-predictions', str(ROOT / model['greedy']),
                '--output', str(output), '--run-id', f'principled_{tag}', '--limit', str(limit), '--microbatch', '1']
            entry = {'stage': tag, 'arguments': command, 'started_utc': now()}
            state['commands'].append(entry)
            started = time.monotonic()
            remaining = deadline - started
            if remaining <= 1:
                raise RuntimeError('Diagnostic two-hour wall budget exhausted')
            log = destination / f'{tag}.log'
            with log.open('x') as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                entry['pid'] = child.pid
                state.update(phase=tag, active_child_pid=child.pid)
                write_json(state_path, state)
                entry['returncode'] = child.wait(timeout=remaining)
            entry.update(elapsed_seconds=time.monotonic()-started, log_sha256=sha256_file(log))
            state.update(active_child_pid=None)
            write_json(state_path, state)
            if entry['returncode']:
                raise RuntimeError(tag + ' failed; no automatic retry or tolerance relaxation')
            completion = output.with_suffix('.jsonl.completion.json')
            report = json.loads(completion.read_text())
            if report['status'] != 'complete' or report['rows'] != limit or report['expected_rows'] != limit:
                raise RuntimeError('Incomplete canonical-action diagnostic')
            entry['artifacts'] = {str(p.relative_to(ROOT)): {'sha256': sha256_file(p), 'bytes': p.stat().st_size}
                for p in (output, completion, output.with_suffix('.jsonl.config.json'))}
            entry['result'] = report
            write_json(state_path, state)
        state.update(status='completed_development_diagnostics', phase='all_fixed_models_scored', finished_utc=now(), active_child_pid=None)
        write_json(state_path, state)
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        state.update(status='failed_stopped_no_retry', error=f'{type(error).__name__}: {error}', finished_utc=now(), active_child_pid=None)
        write_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
