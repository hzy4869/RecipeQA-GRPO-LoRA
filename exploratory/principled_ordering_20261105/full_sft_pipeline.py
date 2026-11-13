"""One full-data SFT baseline, independently checked; development only."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'ops'))
from recipeqa.metrics import score_order
from recipeqa.modeling import model_weight_identity, package_versions, read_jsonl, sha256_file, write_json
from wait_and_evaluate import process_info, require_gpu_idle


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def process_state(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return None


def main():
    os.chdir(ROOT)
    require(ROOT.is_relative_to('/root/autodl-tmp') and ROOT.stat().st_dev != Path('/').stat().st_dev,
            'All inputs, output, cache and environments must remain on the separate data disk')
    for name in ('HF_HOME', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE', 'TMPDIR', 'TORCH_HOME'):
        require(os.environ.get(name) and Path(os.environ[name]).resolve().is_relative_to(ROOT),
                'Missing data-disk environment redirection: ' + name)
    for name in ('XDG_CACHE_HOME', 'PIP_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH'):
        require(not os.environ.get(name) or Path(os.environ[name]).resolve().is_relative_to(ROOT),
                'Active cache escaped data disk: ' + name)
    require(Path(sys.prefix).resolve().is_relative_to(ROOT)
            and Path(sys.executable).absolute().is_relative_to(ROOT), 'Python virtual environment escaped data disk')
    policy_path = HERE / 'FULL_SFT_PROTOCOL.json'
    policy = load(policy_path)
    policy_sha = sha256_file(policy_path)
    require(policy['status'] == 'locked_before_full_data_training', 'Full-data policy not locked')
    destination = HERE / 'full_sft'
    require(not destination.exists(), 'Do not overwrite or automatically restart a full-data run')
    destination.mkdir()
    state_path = destination / 'state.json'
    state = {'status': 'waiting_parameter_diagnostics', 'pid': os.getpid(), 'started_utc': now(),
             'commands': [], 'test_access': False, 'confirmation_access': False,
             'policy_sha256': policy_sha, 'automatic_restart': False, 'post_hoc_exploration': True}
    child = None
    locks = []
    deadline = None

    def interrupted(signum, frame):
        raise RuntimeError(f'Coordinator interrupted by signal {signum}; preserve owned child evidence')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def update(**changes):
        state.update(changes, updated_utc=now())
        write_json(state_path, state)

    def check_sources():
        require(sha256_file(policy_path) == policy_sha, 'Full-data policy changed after launch')
        for name, digest in policy['sources_sha256'].items():
            require(sha256_file(ROOT / name) == digest, 'Bound source/input changed: ' + name)

    def run(tag, arguments):
        nonlocal child
        check_sources()
        require_gpu_idle()
        require(time.monotonic() < deadline, 'Full-data training/evaluation wall budget exhausted')
        entry = {'stage': tag, 'arguments': [sys.executable, *map(str, arguments)], 'started_utc': now()}
        state['commands'].append(entry)
        log = destination / f'{tag}.log'
        started = time.monotonic()
        with log.open('x') as handle:
            child = subprocess.Popen(entry['arguments'], stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            entry['pid'] = child.pid
            update(status='running', phase=tag, active_child_pid=child.pid)
            try:
                entry['returncode'] = child.wait(timeout=deadline - time.monotonic())
            finally:
                entry.update(elapsed_seconds=time.monotonic() - started, finished_utc=now())
                entry['log_sha256'] = sha256_file(log)
                update()
        entry['log_sha256'] = sha256_file(log)
        update(active_child_pid=None)
        require(entry['returncode'] == 0, tag + ' failed; preserve evidence and inspect, no automatic retry')

    try:
        update()
        check_sources()
        waiting = time.monotonic()
        prior_path = HERE / 'parameter_diagnostics/state.json'
        prior_pid = None
        prior_start_ticks = None
        while True:
            if prior_path.exists():
                prior = load(prior_path)
                require(prior_pid is None or prior['pid'] == prior_pid, 'Predecessor PID changed')
                prior_pid = prior['pid']
                info = process_info(prior_pid)
                if info is not None and info['state'] not in ('Z', 'X'):
                    require(info['cwd'] == str(ROOT)
                            and any(Path(arg).name == 'parameter_diagnostic_pipeline.py' for arg in info['argv']),
                            'Predecessor PID is not the expected project coordinator')
                    require(prior_start_ticks is None or prior_start_ticks == info['start_ticks'],
                            'Predecessor PID was reused')
                    prior_start_ticks = info['start_ticks']
                require(prior['status'] not in ('failed_stopped_no_retry', 'stopped_budget'),
                        'Parameter diagnostic stopped; inspect before training')
                if prior['status'] == 'completed_parameter_diagnostics' and (info is None or info['state'] in ('Z', 'X')):
                    require(prior.get('active_child_pid') is None
                            and prior.get('commands') and all(x.get('returncode') == 0 for x in prior['commands']),
                            'Parameter handoff has an active/failed/incomplete command')
                    update(prior_state_sha256=sha256_file(prior_path), prior_state=prior)
                    break
            require(time.monotonic() - waiting < 9000, 'Timed out waiting for reviewed parameter diagnostic')
            time.sleep(10)
        update(status='waiting_gpu_locks')
        lock_started = time.monotonic()
        while not locks:
            acquired = []
            try:
                for name in ('formal_training_pipeline', 'evaluation_pipeline', 'evaluation_handoff',
                             'seed43_pipeline', 'reward_search_20261105'):
                    handle = (ROOT / f'metadata/{name}.lock').open('a+')
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BaseException:
                        handle.close()
                        raise
                    acquired.append(handle)
                locks = acquired
            except BlockingIOError:
                for handle in acquired:
                    handle.close()
                require(time.monotonic() - lock_started < 120, 'GPU locks did not become available')
                time.sleep(2)
        require_gpu_idle()
        deadline = time.monotonic() + policy['wall_limit_seconds']
        require(shutil.disk_usage(ROOT).free > 4 * 1024**3, 'Require 4GiB free on the data disk before training')
        remainder = load(HERE / 'remainder_probe/result.json')
        require(remainder['passed'] is True and remainder['global_step'] == 2
                and len(remainder['optimizer_events']) == 2
                and remainder['versions'] == package_versions()
                and remainder['source_sha256'] == sha256_file(HERE / 'probe_remainder.py'),
                'Installed Trainer remainder handling not verified')
        report = load(HERE / 'full_train_data/manifest_report.json')
        require(report['status'] == 'ready_full_training_data_no_training_started', 'Full training data not ready')
        manifest = ROOT / policy['train_manifest']
        records = read_jsonl(manifest)
        require(len(records) == len({r['qid'] for r in records}) == 7558
                and len({r['recipe_id'] for r in records}) == 5300, 'Full training coverage changed')
        require(all(r['source_split'] == 'train' and r['manifest_set'] == 'train_all7558'
                    and r['technical_status'] == 'ok' for r in records), 'Non-training or damaged row')
        initial = ROOT / policy['initial_adapter']
        initial_identity = model_weight_identity(initial, adapter=True)
        require(model_weight_identity(ROOT / 'models/Qwen3-VL-2B-Instruct')['bundle_sha256']
                == policy['base_bundle_sha256'], 'Base weights or processor bytes changed')
        require(sha256_file(initial / 'adapter_model.safetensors') == policy['initial_weight_sha256'],
                'Common W16 start changed')
        baseline = ROOT / policy['baseline_predictions']
        baseline_config = load(str(baseline) + '.config.json')
        require(baseline_config['manifest_sha256'] == sha256_file(ROOT / policy['dev_manifest'])
                and baseline_config['weight_identity']['adapter'] == initial_identity
                and baseline_config['weight_identity']['base']['bundle_sha256'] == policy['base_bundle_sha256']
                and baseline_config['generation']['do_sample'] is False
                and baseline_config['generation']['max_new_tokens'] == 32
                and baseline_config['input'] == policy['input'], 'W16 baseline uses different inputs/weights/decoding')
        baseline_gate = destination / 'baseline_coverage.json'
        run('baseline_coverage', [ROOT / 'scripts/validate_run.py', 'predictions',
             '--manifest', ROOT / policy['dev_manifest'], '--predictions', baseline,
             '--run-id', baseline_config['cli']['run_id'], '--checkpoint', initial,
             '--seed', str(baseline_config['cli']['seed']), '--expected-questions', '200',
             '--expected-recipes', '200', '--output', baseline_gate])
        require(load(baseline_gate)['passed'], 'W16 baseline coverage failed')
        run_dir = destination / 'train'
        common = ['--model-path', ROOT / 'models/Qwen3-VL-2B-Instruct',
                  '--revision', policy['revision'], '--image-root', ROOT / 'data/images']
        run('train', [ROOT / 'scripts/train_sft.py', *common, '--manifest', manifest,
             '--output-dir', run_dir, '--initial-adapter', initial, '--max-steps', '-1',
             '--save-steps', '256', '--learning-rate', '0.00005',
             '--gradient-accumulation-steps', '8', '--seed', str(policy['seed'])])
        gate = destination / 'training_artifact_gate.json'
        run('training_artifact_gate', [ROOT / 'scripts/validate_run.py', 'training', '--run-dir', run_dir,
             '--min-steps', '945', '--output', gate])
        require(load(gate)['passed'], 'Full training artifacts failed validation')
        outcome, config = load(run_dir / 'train_outcome.json'), load(run_dir / 'resolved_config.json')
        require(outcome['global_step'] == 945 and outcome['status'] == 'completed', 'Expected one pass / 945 updates')
        require(config['manifest_sha256'] == sha256_file(manifest)
                and config['cli']['max_steps'] == -1 and config['trainer']['num_train_epochs'] == 1,
                'Training budget changed')
        require(model_weight_identity(initial, adapter=True) == initial_identity, 'W16 changed during run')
        require(load(run_dir / 'run_signature.json')['payload']['initial_W_adapter_identity'] == initial_identity,
                'Actual initial adapter differs from policy')
        visits, audits = read_jsonl(run_dir / 'actual_prompt_visits.jsonl'), read_jsonl(run_dir / 'input_audit.jsonl')
        require(len(visits) == len(audits) == len(records), 'Missing/repeated full-data exposure')
        for index, (record, visit, audit) in enumerate(zip(records, visits, audits)):
            require(all(record[k] == visit[k] == audit[k] for k in ('qid', 'recipe_id', 'presentation_id')),
                    'Actual training order/input identity changed')
            require(visit['visit_id'] == record['visit_id'] and visit['input_hash'] == audit['input_hash']
                    and visit['global_step_before_update'] == index // 8, 'Wrong visit or accumulation boundary')
        write_json(destination / 'exposure_verification.json', {
            'passed': True, 'actual_visits': len(visits), 'unique_questions': 7558, 'unique_recipes': 5300,
            'optimizer_updates': 945, 'last_accumulation_examples': 6,
            'sources_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in
                (manifest, run_dir / 'actual_prompt_visits.jsonl', run_dir / 'input_audit.jsonl')},
            'loss_normalization': 'Unchanged frozen SFTTrainer runner and recorded TRL version/config; no custom rescaling for final6',
            'disclosure': 'Data, exposure and seed differ from original small-data SFT; this is a scale-up, not an isolated coverage effect.'})
        require(config['input'] == policy['input'], 'Training input policy changed')
        candidates = []
        points = [(0, initial, baseline), (256, run_dir / 'checkpoint-256', None),
                  (512, run_dir / 'checkpoint-512', None), (768, run_dir / 'checkpoint-768', None),
                  (945, run_dir / 'adapter_final', None)]
        for step, checkpoint, predictions in points:
            if predictions is None:
                predictions = destination / f'dev{step}' / 'predictions.jsonl'
                predictions.parent.mkdir()
                run_id = f'full7558_sft_seed{policy["seed"]}_dev{step}'
                run(f'dev{step}', [ROOT / 'scripts/predict.py', *common, '--adapter-path', checkpoint,
                     '--manifest', ROOT / policy['dev_manifest'], '--output', predictions,
                     '--run-id', run_id, '--seed', str(policy['seed'])])
                coverage = predictions.parent / 'coverage.json'
                run(f'coverage{step}', [ROOT / 'scripts/validate_run.py', 'predictions',
                     '--manifest', ROOT / policy['dev_manifest'], '--predictions', predictions,
                     '--run-id', run_id, '--checkpoint', checkpoint, '--seed', str(policy['seed']),
                     '--expected-questions', '200', '--expected-recipes', '200', '--output', coverage])
                require(load(coverage)['passed'], 'Incomplete development evaluation')
            rows = read_jsonl(predictions)
            require(len(rows) == len({r['recipe_id'] for r in rows}) == 200
                    and not any(r.get('technical_error') for r in rows), 'Invalid development denominator')
            scores = [score_order(r['raw_response'], r['display_perm'], r['target_order']) for r in rows]
            require(all(s == r['score'] for s, r in zip(scores, rows)), 'Raw development rescore mismatch')
            candidates.append({'step': step, 'checkpoint': str(checkpoint), 'predictions': str(predictions),
                'predictions_sha256': sha256_file(predictions),
                **{key: sum(s[key] for s in scores) / 200 for key in ('exact', 'pair', 'valid')}})
            update(development_candidates=candidates)
        selected = max(candidates, key=lambda r: (r['exact'], r['pair'], -r['step']))
        selection = {'rule': policy['selection_rule'], 'candidates': candidates, 'selected': selected,
                     'selected_identity': model_weight_identity(selected['checkpoint'], adapter=True),
                     'test_access': False, 'confirmation_access': False, 'policy_sha256': policy_sha}
        write_json(destination / 'selection.json', selection)
        update(status='completed_full_data_sft', phase='one_pass_and_fixed_dev_selection_complete',
               active_child_pid=None, selection=selection, finished_utc=now())
    except BaseException as error:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        if state['commands']:
            entry = state['commands'][-1]
            entry.update(returncode=child.returncode if child is not None else None,
                         error=f'{type(error).__name__}: {error}')
            log = destination / f'{entry["stage"]}.log'
            if log.exists():
                entry['log_sha256'] = sha256_file(log)
        update(status='failed_stopped_no_retry', active_child_pid=None,
               error=f'{type(error).__name__}: {error}', finished_utc=now())
        raise
    finally:
        for handle in locks:
            handle.close()


if __name__ == '__main__':
    main()
