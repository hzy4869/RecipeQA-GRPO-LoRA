#!/usr/bin/env python3
"""Prepare/export/verify five frozen inference adapters; no authentication or model code."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tarfile
import zlib
from datetime import datetime, timezone

REMOTE_ROOT = Path('/root/autodl-tmp/rl_recipeqa')
PROTOCOL = 'research/endpoint_bank_20261106/PROTOCOL_V1.json'
PROTOCOL_SHA = '5c417fdca7beb0d4ae77c370070c0996c08b07b33fe2a716a40fd2b734996172'
BASE_REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
EXPORT_SUBDIR = 'exports/endpoint_inference_20261107_v1'
ARCHIVE = 'endpoint_inference_adapters.tar.gz'
MANIFEST = 'endpoint_inference_adapters.manifest.json'
INDEX = 'ADAPTER_INDEX.json'
MAX_SMALL_FILE = 64 * 1024 * 1024
ENDPOINT_NAMES = {'initial_sft768', 'flow44_raw_sft_256', 'flow44_exact_kl_256', 'flow45_raw_sft_256', 'flow45_exact_kl_256'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def packed(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def relative_name(name):
    require(isinstance(name, str) and name and '\\' not in name, 'Invalid member name')
    p = PurePosixPath(name)
    require(not p.is_absolute() and str(p) == name and all(x not in ('', '.', '..') for x in p.parts), 'Unsafe member path')
    require(not any(x in ('.ssh', '.state') or 'PRIVATE' in x or 'confirmation' in x.lower() for x in p.parts), 'Forbidden private/cache path')
    require(p.name not in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'model.safetensors'), 'Forbidden base/resume binary')
    require(p.suffix in ('.json', '.jsonl', '.safetensors', '.jinja', '.txt'), 'Unexpected file type')
    return p


def safe_source(root, name):
    p = relative_name(name)
    current = root
    require(root.is_dir() and not root.is_symlink(), 'Root must be a real directory')
    for part in p.parts:
        current = current / part
        require(not current.is_symlink(), 'Symlink source refused')
    require(current.is_file() and current.resolve().is_relative_to(root.resolve()), 'Missing or escaping source')
    require(stat.S_ISREG(current.stat().st_mode), 'Nonregular source refused')
    return current


def checked_json(path, expected_sha=None):
    if expected_sha is not None:
        require(digest(path) == expected_sha, 'JSON SHA mismatch: ' + str(path))
    return json.loads(Path(path).read_text())


def build_spec(source_root, snapshot_root):
    """Only sealed old metadata and protocol, never current score/greedy rows."""
    source_root, snapshot_root = Path(source_root).resolve(), Path(snapshot_root).resolve()
    protocol = checked_json(source_root / PROTOCOL, PROTOCOL_SHA)
    require(protocol['revision'] == BASE_REVISION and set(protocol['endpoints']) == ENDPOINT_NAMES, 'Wrong frozen endpoints/revision')
    snap = checked_json(snapshot_root / 'LOCAL_SNAPSHOT.json')
    table = {}; roles = {}
    def add(name, sha, size, role):
        relative_name(name)
        require(len(sha) == 64 and all(x in '0123456789abcdef' for x in sha), 'Malformed SHA')
        value = {'sha256': sha, 'bytes': size, 'max_bytes': size if size is not None else MAX_SMALL_FILE}
        require(name not in table or table[name] == value, 'Conflicting member identity')
        table[name] = value; roles.setdefault(role, []).append(name)
    def metadata(name):
        # Source metadata are fixed, completed training records. No endpoint outputs.
        require(not name.startswith(protocol['output_directory'] + '/') and 'predictions' not in name, 'Scores are outside backup scope')
        if name in snap['files']:
            record = snap['files'][name]; p = safe_source(snapshot_root, name)
            require(digest(p) == record['sha256'] and p.stat().st_size == record['bytes'], 'Snapshot metadata differs')
            add(name, record['sha256'], record['bytes'], 'source_metadata')
        else:
            p = safe_source(source_root, name)
            require(name in protocol['artifacts_sha256'], 'Unsealed metadata absent from snapshot')
            require(digest(p) == protocol['artifacts_sha256'][name], 'Protocol artifact differs')
            add(name, digest(p), p.stat().st_size, 'source_metadata')
    base = protocol['endpoints']['initial_sft768']['weight_identity']['base']
    for key, endpoint in protocol['endpoints'].items():
        identity = endpoint['weight_identity']
        require(identity['base'] == base, 'Endpoint base mismatch')
        require(identity['adapter']['directory'] == str(REMOTE_ROOT / endpoint['adapter_path']), 'Adapter directory mismatch')
        weights = identity['adapter']['weight_files']
        require(len(weights) == 1 and weights[0]['name'] == 'adapter_model.safetensors', 'Unexpected adapter weights')
        for item in weights + identity['adapter']['auxiliary_files']:
            name = endpoint['adapter_path'] + '/' + item['name']
            # Some old auxiliary files were not downloaded; SHA is still pinned by the frozen physical identity.
            size = item.get('bytes', snap['files'].get(name, {}).get('bytes'))
            add(name, item['sha256'], size, key)
        if key != 'initial_sft768':
            parent = str(PurePosixPath(endpoint['adapter_path']).parent)
            for name in ['run_signature.json', 'train_outcome.json', 'independent_verification.json']:
                metadata(parent + '/' + name)
            metadata(endpoint['adapter_path'] + '/finite_policy_checkpoint_audit.json')
            arm = key.removesuffix('_256')
            metadata('research/coverage_20261106/locked_v1/child_' + arm + '.json')
    for item in base['auxiliary_files']:
        add(protocol['model_path'] + '/' + item['name'], item['sha256'], item.get('bytes'), 'shared_base_processor')
    initial = 'exploratory/principled_ordering_20261105/full_sft'
    for n in ['state.json', 'train/run_signature.json', 'train/resolved_config.json', 'train/train_outcome.json']:
        metadata(initial + '/' + n)
    metadata('research/coverage_20261106/PROTOCOL_V1.json')
    for n in ['state.json', 'summary.json']:
        metadata('exploratory/coverage_20261106/stage_v1/' + n)
    for flow in ['flow44', 'flow45']:
        metadata('research/coverage_20261106/prepared_v1/formal_' + flow + '.jsonl')
    for bank in protocol['banks'].values():
        metadata(bank['manifest'])
    add(PROTOCOL, PROTOCOL_SHA, (source_root / PROTOCOL).stat().st_size, 'endpoint_protocol')
    return {'schema_version': 1, 'purpose': 'Five fixed inference adapters; no base weights or training resume state; no new score rows.',
            'remote_root': str(REMOTE_ROOT), 'export_subdirectory': EXPORT_SUBDIR,
            'protocol': PROTOCOL, 'protocol_sha256': PROTOCOL_SHA, 'base_revision': BASE_REVISION,
            'base_identity_reference_only': base, 'fresh_base_weight_hashing': False,
            'endpoints': protocol['endpoints'], 'input_policy': protocol['input'], 'required_versions': protocol['required_versions'],
            'files': dict(sorted(table.items())), 'roles': roles,
            'selection': {'initial_sft768': 'historically dev-selected step768; never adapter_final945',
                          'coverage': 'all four predeclared fixed256 endpoints; not score-selected'},
            'tool_sha256': digest(__file__), 'source_snapshot_captured_at': snap['captured_at'],
            'excluded': ['base weights', 'optimizer/scheduler/RNG', 'photos', 'all new endpoint score rows', 'confirmation', 'private human maps']}


def expected_members(spec):
    names = {PROTOCOL}
    for endpoint in spec['endpoints'].values():
        identity = endpoint['weight_identity']
        for item in identity['adapter']['weight_files'] + identity['adapter']['auxiliary_files']:
            names.add(endpoint['adapter_path'] + '/' + item['name'])
    base = spec['base_identity_reference_only']
    for item in base['auxiliary_files']:
        names.add('models/Qwen3-VL-2B-Instruct/' + item['name'])
    initial = 'exploratory/principled_ordering_20261105/full_sft'
    names.update(initial + '/' + n for n in ['state.json', 'train/run_signature.json', 'train/resolved_config.json', 'train/train_outcome.json'])
    names.add('research/coverage_20261106/PROTOCOL_V1.json')
    names.update('exploratory/coverage_20261106/stage_v1/' + n for n in ['state.json', 'summary.json'])
    for flow in ['flow44', 'flow45']:
        names.add('research/coverage_20261106/prepared_v1/formal_' + flow + '.jsonl')
        for method in ['raw_sft', 'exact_kl']:
            arm = flow + '_' + method
            parent = 'exploratory/coverage_20261106/stage_v1/' + arm + '/train/'
            names.update(parent + n for n in ['run_signature.json', 'train_outcome.json', 'independent_verification.json', 'checkpoint-256/finite_policy_checkpoint_audit.json'])
            names.add('research/coverage_20261106/locked_v1/child_' + arm + '.json')
    names.update(['research/endpoint_bank_20261106/prepared_v1/train256.jsonl', 'data/manifests/val.jsonl'])
    return names


def validate_spec_members(spec):
    require(set(spec['files']) == expected_members(spec), 'Missing/extra required backup files')
    for endpoint in spec['endpoints'].values():
        identity = endpoint['weight_identity']
        require(identity['base'] == spec['base_identity_reference_only'], 'Base reference differs')
        require(identity['adapter']['directory'] == str(REMOTE_ROOT / endpoint['adapter_path']), 'Adapter location differs')
        for item in identity['adapter']['weight_files'] + identity['adapter']['auxiliary_files']:
            record = spec['files'][endpoint['adapter_path'] + '/' + item['name']]
            require(record['sha256'] == item['sha256'] and ('bytes' not in item or record['bytes'] == item['bytes']), 'Adapter file identity differs')
    for item in spec['base_identity_reference_only']['auxiliary_files']:
        require(spec['files']['models/Qwen3-VL-2B-Instruct/' + item['name']]['sha256'] == item['sha256'], 'Base processor SHA differs')
    require(spec['files'][PROTOCOL]['sha256'] == PROTOCOL_SHA, 'Protocol archive member differs')


def load_spec(path, expected_sha):
    require(digest(path) == expected_sha, 'Spec SHA mismatch')
    spec = checked_json(path)
    require(spec['schema_version'] == 1 and spec['remote_root'] == str(REMOTE_ROOT), 'Wrong schema/root')
    require(spec['protocol_sha256'] == PROTOCOL_SHA and set(spec['endpoints']) == ENDPOINT_NAMES, 'Wrong protocol/endpoints')
    require(spec['export_subdirectory'] == EXPORT_SUBDIR and spec['base_revision'] == BASE_REVISION, 'Wrong output/revision')
    require(spec['tool_sha256'] == digest(__file__), 'Backup tool differs from reviewed spec')
    validate_spec_members(spec)
    for name, item in spec['files'].items():
        relative_name(name)
        require(type(item['max_bytes']) is int and item['max_bytes'] > 0, 'Missing positive byte bound')
    return spec


def active_group_members(pgid, proc_root=Path('/proc')):
    """Count orphan descendants too; a dead original PID alone is insufficient."""
    result = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit(): continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            continue  # Process exited during this read-only scan.
        if int(fields[2]) == pgid and fields[0] != 'Z': result.append(int(entry.name))
    return sorted(result)


def quiescent(root):
    state_path = root / 'exploratory/endpoint_bank_20261106/stage_v1/state.json'
    state = checked_json(state_path)
    require(state.get('policy_sha256') == PROTOCOL_SHA, 'Terminal state policy SHA differs')
    require(state['status'] in ('completed_endpoint_bank_zero_update', 'failed_preserved_no_retry') and state['active_child_pid'] is None, 'Endpoint queue not terminal')
    for pid in [state.get('pid'), state.get('pgid')]:
        require(type(pid) is int and pid > 1, 'Missing runtime process identity')
        proc = Path('/proc') / str(pid) / 'stat'
        if proc.exists():
            require(proc.read_text().rsplit(')', 1)[1].split()[0] == 'Z', 'Old coordinator/watchdog still alive (or PID reused); defer export')
    require(not active_group_members(state['pgid']), 'Old scientific process group still has live descendants; defer export')
    probe = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
    require(not probe.stdout.strip(), 'GPU compute process active; defer disk-heavy export')
    return {'checked_utc': datetime.now(timezone.utc).isoformat(), 'state_sha256': digest(state_path),
            'status': state['status'], 'policy_sha256': state['policy_sha256'], 'coordinator_and_watchdog_stopped': True, 'owned_group_members_remaining': [], 'gpu_compute_pids': [],
            'gpu_probe': 'nvidia-smi read-only; no CUDA/model initialization'}


def validate_sources(root, spec):
    policy = checked_json(safe_source(root, PROTOCOL), PROTOCOL_SHA)
    require(policy['endpoints'] == spec['endpoints'], 'Source protocol endpoint identity differs')
    records = {}
    for name, item in spec['files'].items():
        p = safe_source(root, name); size = p.stat().st_size
        require(size > 0 and size <= item['max_bytes'] and (item['bytes'] is None or size == item['bytes']), 'Source size mismatch: ' + name)
        require(digest(p) == item['sha256'], 'Source SHA mismatch: ' + name)
        records[name] = {'bytes': size, 'sha256': item['sha256']}
    # Cross-check selected versus fixed-step metadata, without any scoring rows.
    initial = checked_json(root / 'exploratory/principled_ordering_20261105/full_sft/state.json')
    selected = initial['selection']
    require(initial['status'] == 'completed_full_data_sft' and selected['selected']['step'] == 768, 'Initial SFT selection differs')
    require(selected['selected_identity'] == spec['endpoints']['initial_sft768']['weight_identity']['adapter'], 'Initial selected physical identity mismatch')
    for key, endpoint in spec['endpoints'].items():
        if key == 'initial_sft768': continue
        parent = root / str(PurePosixPath(endpoint['adapter_path']).parent)
        audit = checked_json(root / endpoint['adapter_path'] / 'finite_policy_checkpoint_audit.json')
        outcome = checked_json(parent / 'train_outcome.json'); verify = checked_json(parent / 'independent_verification.json')
        require(audit['global_step'] == outcome['global_step'] == verify['optimizer_steps'] == 256, 'Wrong fixed256 metadata')
        require(outcome['status'] == 'completed' and verify['passed'] is True and verify['limit'] == 1024, 'Incomplete training metadata')
        require(audit['adapter_identity'] == endpoint['weight_identity']['adapter'], 'Actual checkpoint physical identity mismatch')
    return records


def validate_single_gzip(archive, max_expanded):
    """Reject concatenated gzip, trailing bytes, corrupt CRC and oversized expansion."""
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    expanded = 0
    try:
        with Path(archive).open('rb') as source:
            for chunk in iter(lambda: source.read(64 * 1024), b''):
                require(not decoder.eof, 'Trailing/concatenated gzip payload')
                pending = chunk
                while pending:
                    value = decoder.decompress(pending, 1024 * 1024)
                    expanded += len(value)
                    require(expanded <= max_expanded, 'Gzip expansion exceeds pinned bounds')
                    require(not decoder.unused_data, 'Trailing/concatenated gzip payload')
                    pending = decoder.unconsumed_tail
        expanded += len(decoder.flush())
    except zlib.error as exc:
        raise ValueError('Invalid gzip/CRC') from exc
    require(decoder.eof and expanded <= max_expanded, 'Truncated/oversized gzip')
    return expanded


def verify_bundle(spec, spec_sha, archive, manifest_path):
    archive, manifest_path = Path(archive), Path(manifest_path)
    require(not archive.is_symlink() and not manifest_path.is_symlink(), 'Symlink backup refused')
    require(manifest_path.stat().st_size <= 1024 * 1024, 'Oversized manifest')
    m = checked_json(manifest_path)
    require(m['schema_version'] == 1 and m['purpose'] == spec['purpose'], 'Manifest schema/purpose differs')
    require(m['base_identity_reference_only'] == spec['base_identity_reference_only'] and m['fresh_base_weight_hashing'] is False and m['excluded'] == spec['excluded'], 'Base/exclusion metadata differs')
    require(m['spec_sha256'] == spec_sha and m['protocol_sha256'] == spec['protocol_sha256'], 'Manifest policy/spec differs')
    require(m['endpoints'] == spec['endpoints'] and m['base_revision'] == spec['base_revision'], 'Manifest model identities differ')
    require(set(m['files']) == set(spec['files']), 'Manifest missing/extra members')
    require(m['archive_name'] == ARCHIVE and archive.stat().st_size == m['archive_bytes'] and digest(archive) == m['archive_sha256'], 'Archive SHA/bytes mismatch')
    core = {k: v for k, v in m.items() if k not in ('archive_name', 'archive_bytes', 'archive_sha256')}
    seen = set(); index_seen = False
    validate_single_gzip(archive, sum(x['max_bytes'] for x in spec['files'].values()) + 2 * 1024 * 1024)
    with gzip.open(archive, 'rb') as gz:
        with tarfile.open(fileobj=gz, mode='r:') as tf:
            for member in tf:
                require(member.name not in seen and member.isfile() and not member.issym() and not member.islnk(), 'Duplicate/nonregular archive member')
                seen.add(member.name)
                if member.name == INDEX:
                    require(member.size <= 1024 * 1024, 'Oversized index')
                    require(tf.extractfile(member).read() == packed(core), 'Inner index differs from external manifest')
                    index_seen = True; continue
                relative_name(member.name)
                require(member.name in spec['files'], 'Extra archive member')
                expected = spec['files'][member.name]; record = m['files'][member.name]
                require(type(record['bytes']) is int and 0 < record['bytes'] <= expected['max_bytes'], 'Invalid member size')
                require(expected['bytes'] is None or record['bytes'] == expected['bytes'], 'Pinned member bytes differ')
                require(member.size == record['bytes'] and record['sha256'] == expected['sha256'], 'Archive member metadata differs')
                h = hashlib.sha256()
                f = tf.extractfile(member)
                for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
                require(h.hexdigest() == expected['sha256'], 'Archive member SHA mismatch')
            tar_end = tf.offset
        # tarfile stops on the first zero header. Inspect everything after the final
        # member rather than silently accepting a second tar or nonzero payload.
        gz.seek(tar_end)
        padding_bytes = 0
        for chunk in iter(lambda: gz.read(1024 * 1024), b''):
            require(not any(chunk), 'Nonzero payload after tar end')
            padding_bytes += len(chunk)
        require(padding_bytes >= 1024 and padding_bytes % 512 == 0, 'Missing/malformed tar end padding')
    require(index_seen and seen == set(spec['files']) | {INDEX}, 'Missing archive members/index')
    return {'status': 'five_endpoint_archive_and_all_members_verified', 'files': len(spec['files']),
            'archive_bytes': m['archive_bytes'], 'archive_sha256': m['archive_sha256'],
            'spec_sha256': spec_sha, 'protocol_sha256': spec['protocol_sha256'],
            'base_weights_included': False, 'optimizer_resume_included': False, 'new_scores_included': False,
            'local_tensor_load_or_crc': False}


def export_bundle(spec, spec_sha):
    root = REMOTE_ROOT
    quiet = quiescent(root)
    out = root / EXPORT_SUBDIR
    # No user-controlled output: every export/partial/temporary stays on the data disk.
    require(root.is_dir() and not root.is_symlink(), 'Invalid remote data root')
    current = root
    for part in PurePosixPath(EXPORT_SUBDIR).parts:
        current = current / part
        require(not current.is_symlink(), 'Symlink export parent refused')
        current.mkdir(exist_ok=True)
    require(out.resolve().is_relative_to(root.resolve()), 'Output escaped data disk')
    archive, mf = out / ARCHIVE, out / MANIFEST
    require(not archive.is_symlink() and not mf.is_symlink(), 'Symlink archive/manifest refused')
    if archive.exists() or mf.exists():
        require(archive.exists() and mf.exists(), 'Incomplete old backup retained; inspect before retrying in a new namespace')
        result = verify_bundle(spec, spec_sha, archive, mf)
        result['existing_complete_backup_reused'] = True
        result['source_current_rechecked'] = False
        return result
    records = validate_sources(root, spec)
    need = 2 * sum(x['bytes'] for x in records.values()) + 16 * 1024 * 1024
    require(shutil.disk_usage(out).free >= need, 'Insufficient export storage')
    core = {'schema_version': 1, 'spec_sha256': spec_sha, 'protocol_sha256': spec['protocol_sha256'],
            'captured_utc': datetime.now(timezone.utc).isoformat(), 'purpose': spec['purpose'],
            'endpoints': spec['endpoints'], 'base_revision': spec['base_revision'],
            'base_identity_reference_only': spec['base_identity_reference_only'], 'fresh_base_weight_hashing': False,
            'files': records, 'quiescence': quiet, 'excluded': spec['excluded']}
    # The spec itself is an external, root-pinned prerequisite. Its digest binds this archive.
    partial = out / (ARCHIVE + '.partial')
    require(not partial.exists(), 'Old partial retained; do not overwrite')
    with partial.open('xb') as f, gzip.GzipFile(filename='', mode='wb', compresslevel=1, fileobj=f, mtime=0) as gz, tarfile.open(fileobj=gz, mode='w') as tf:
        for name in sorted(records):
            p = safe_source(root, name)
            info = tarfile.TarInfo(name); info.size = records[name]['bytes']; info.mode = 0o600; info.mtime = 0
            with p.open('rb') as src: tf.addfile(info, src)
        b = packed(core); info = tarfile.TarInfo(INDEX); info.size = len(b); info.mode = 0o600; tf.addfile(info, io.BytesIO(b))
    manifest = dict(core, archive_name=ARCHIVE, archive_bytes=partial.stat().st_size, archive_sha256=digest(partial))
    mpartial = out / (MANIFEST + '.partial')
    with mpartial.open('xb') as f: f.write(packed(manifest))
    result = verify_bundle(spec, spec_sha, partial, mpartial)  # Re-read every archive byte before publication.
    # If a producer file changed while packing, the pinned per-member SHA above refuses it.
    os.replace(partial, archive); os.replace(mpartial, mf)
    result['existing_complete_backup_reused'] = False
    result['source_current_rechecked'] = True
    result['remote_archive'] = str(archive); result['remote_manifest'] = str(mf)
    return result


def download_with_sftp(sftp, spec, spec_sha, local_directory):
    """Caller owns already-authenticated SFTP; this module never handles credentials/SSH."""
    out = Path(local_directory).resolve(); out.mkdir(parents=True, exist_ok=True)
    archive, mf = out / ARCHIVE, out / MANIFEST
    require(not archive.is_symlink() and not mf.is_symlink(), 'Symlink archive/manifest refused')
    if archive.exists() or mf.exists():
        require(archive.exists() and mf.exists(), 'Local incomplete backup retained; use a fresh destination after inspection')
        result = verify_bundle(spec, spec_sha, archive, mf); result['download_skipped'] = True; return result
    remote = str(REMOTE_ROOT / EXPORT_SUBDIR)
    mp, ap = out / (MANIFEST + '.partial'), out / (ARCHIVE + '.partial')
    require(not mp.exists() and not ap.exists() and not mp.is_symlink() and not ap.is_symlink(), 'Old local partial retained')
    # Manifest size is bounded before reading; no authentication material is stored.
    require(sftp.stat(remote + '/' + MANIFEST).st_size <= 1024 * 1024, 'Oversized remote manifest')
    with mp.open('xb'): pass
    sftp.get(remote + '/' + MANIFEST, str(mp))
    m = checked_json(mp)
    require(m['spec_sha256'] == spec_sha and set(m['files']) == set(spec['files']), 'Remote manifest spec/file set differs')
    require(sftp.stat(remote + '/' + ARCHIVE).st_size == m['archive_bytes'], 'Remote archive size differs')
    require(0 < m['archive_bytes'] <= sum(x['max_bytes'] for x in spec['files'].values()) + 16 * 1024 * 1024, 'Remote archive exceeds fixed size bounds')
    with ap.open('xb'): pass
    sftp.get(remote + '/' + ARCHIVE, str(ap))
    result = verify_bundle(spec, spec_sha, ap, mp)
    os.replace(ap, archive); os.replace(mp, mf)
    result['download_skipped'] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build-spec'); b.add_argument('--source-root', type=Path, default=Path.cwd()); b.add_argument('--snapshot-root', type=Path, required=True); b.add_argument('--output', type=Path, required=True)
    for name in ('export', 'verify'):
        p = sub.add_parser(name); p.add_argument('--spec', type=Path, required=True); p.add_argument('--spec-sha256', required=True)
        if name == 'verify':
            p.add_argument('--archive', type=Path, required=True); p.add_argument('--manifest', type=Path, required=True); p.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'build-spec':
        value = build_spec(args.source_root, args.snapshot_root); raw = packed(value)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as f: f.write(raw)
        print(json.dumps({'status': 'backup_spec_prepared_no_export', 'path': str(args.output), 'sha256': sha_bytes(raw), 'members': len(value['files'])})); return
    spec = load_spec(args.spec, args.spec_sha256)
    result = export_bundle(spec, args.spec_sha256) if args.command == 'export' else verify_bundle(spec, args.spec_sha256, args.archive, args.manifest)
    if args.command == 'verify' and args.output:
        with args.output.open('xb') as f: f.write(packed(result))
    print(json.dumps(result, sort_keys=True))

if __name__ == '__main__':
    main()
