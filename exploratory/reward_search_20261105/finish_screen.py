"""Detached CPU-only completion monitor; never starts/restarts training."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import os
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATE = HERE / 'finish_state.json'
SUMMARY_SHA = '625c226d63575a0f2d181af5821a1ce1a829df38ec3e0351c407fecb7caba5f1'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(value):
    temporary = STATE.with_suffix('.json.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATE)


def main():
    if not ROOT.is_relative_to('/root/autodl-tmp') or ROOT.stat().st_dev == Path('/').stat().st_dev:
        raise RuntimeError('Completion artifacts must remain on the mounted data disk')
    if STATE.exists():
        raise RuntimeError('Completion monitor state already exists; inspect instead of restart')
    started = time.monotonic()
    record = {'status': 'waiting_for_screen', 'pid': os.getpid(), 'started_utc': now(),
              'training_restart': False, 'test_access': False, 'automatic_promotion': False}
    try:
        while True:
            state = json.loads((HERE / 'screen/state.json').read_text())
            record.update(observed_utc=now(), screen_status=state['status'], screen_phase=state.get('phase'),
                          selected_arms=len(state.get('selected', {})))
            save(record)
            if state['status'] in ('completed_development_screen', 'failed'):
                break
            live = subprocess.run(['ps', '-p', str(state['pid']), '-o', 'stat='],
                                  capture_output=True, text=True).stdout.strip()
            if not live or live.startswith(('Z', 'X')):
                raise RuntimeError('Training parent exited without a terminal state; no automatic restart')
            if time.monotonic() - started > 21600:
                raise RuntimeError('Six-hour read-only monitoring limit exceeded')
            time.sleep(30)
        summary = HERE / 'summarize_screen.py'
        if hashlib.sha256(summary.read_bytes()).hexdigest() != SUMMARY_SHA:
            raise RuntimeError('Summary source differs from the integration-tested version')
        command = [sys.executable, str(summary), '--output-dir', str(HERE / 'screen_summary')]
        if state['status'] != 'completed_development_screen':
            command.append('--allow-partial')
        record.update(status='summarizing', summary_source_sha256=SUMMARY_SHA)
        save(record)
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
        receipt = {'screen_status': state['status'], 'summary_returncode': result.returncode,
                   'summary_source_sha256': SUMMARY_SHA, 'stdout': result.stdout,
                   'stderr': result.stderr, 'utc': now(), 'test_access': False,
                   'training_restart': False, 'automatic_promotion': False}
        (HERE / 'detached_postprocess_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        archive = HERE / 'screen_terminal_evidence.tar.gz'
        if archive.exists():
            raise RuntimeError('Terminal evidence archive already exists; do not overwrite')
        files = [p for p in HERE.rglob('*') if p.is_file()
                 and p.suffix in ('.py', '.md', '.json', '.jsonl', '.csv', '.log')
                 and '__pycache__' not in p.parts and p != STATE
                 and p.name not in ('finish_parent.log', 'latest_observation.json')]
        receipts = {}
        with tarfile.open(archive, 'w:gz') as stream:
            for path in sorted(files):
                raw = path.read_bytes()
                name = str(path.relative_to(HERE))
                receipts[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
                item = tarfile.TarInfo(name)
                item.size = len(raw)
                stream.addfile(item, io.BytesIO(raw))
            manifest = {'captured_at': now(), 'scope': 'Terminal exploratory development-only screen',
                        'screen_status': state['status'], 'postprocess_returncode': result.returncode,
                        'files': receipts, 'test_access': False, 'binary_weights_included': False}
            raw = (json.dumps(manifest, indent=2) + '\n').encode()
            item = tarfile.TarInfo('LOCAL_SNAPSHOT.json')
            item.size = len(raw)
            stream.addfile(item, io.BytesIO(raw))
        record.update(status='finished', finished_utc=now(), screen_status=state['status'],
                      summary_returncode=result.returncode, archive=str(archive),
                      archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                      archive_bytes=archive.stat().st_size)
        save(record)
    except BaseException as error:
        record.update(status='monitor_failed', error=f'{type(error).__name__}: {error}', finished_utc=now())
        save(record)
        raise


if __name__ == '__main__':
    main()
