#!/usr/bin/env python3
"""Freeze measured, verified inputs after pilots and before formal training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/rl_recipeqa'))
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to('/root/autodl-tmp'):
        raise ValueError('Use the data disk')
    os.chdir(root)
    sys.path.insert(0, str(root / 'src'))
    from recipeqa.modeling import model_weight_identity, package_versions, sha256_file, write_json
    destination = root / 'configs/protocol_lock.json'
    if destination.exists():
        raise RuntimeError('A protocol lock already exists; do not silently replace it')
    assert json.loads((root / 'metadata/complete_pilots.json').read_text())['status'] == 'completed'
    pilots = json.loads((root / 'results/same_w_pilot_summary.json').read_text())['runs']
    assert len(pilots) == 3 and {r['method'] for r in pilots} == {'sft', 'grpo_exact', 'grpo_pair'}
    assert all(r['status'] == 'completed_evidence_summary' and not r['issues'] for r in pilots)
    outcomes = [json.loads((Path(r['run_dir']) / 'train_outcome.json').read_text()) for r in pilots]
    assert len({r['initial_adapter_fingerprint'] for r in outcomes}) == 1
    assert len({r['initial_visual_fingerprint'] for r in outcomes}) == 1
    assert all(r['global_step'] == 20 and r['visual_weights_unchanged'] for r in outcomes)
    blank = root / 'runs/w_blank_control/predictions.jsonl'
    rows = [json.loads(line) for line in blank.read_text().splitlines() if line.strip()]
    assert len(rows) == 20 and not any(row.get('technical_error') for row in rows)
    if sum(row['score']['exact'] for row in rows) >= 4:
        raise RuntimeError('Blank-image control >=4/20; inspect identity leakage/chance before freezing')
    source_files = sorted([*Path('src/recipeqa').glob('*.py'), *Path('scripts').glob('*.py'),
                           *Path('ops').glob('*.py')])
    data_files = [Path('data/manifests') / name for name in [
        'train.jsonl', 'train_visits.jsonl', 'train_visits_4pass.jsonl', 'val.jsonl', 'test.jsonl',
        'audit.jsonl', 'orbit.jsonl', 'process.jsonl', 'bank.jsonl', 'manifest_report.json',
        'image_report.json', 'sft4pass_prefix_audit.json']]
    paths = [*source_files, *data_files, Path('configs/candidate_protocol.json'), Path('RESEARCH_PLAN.md')]
    candidate = json.loads((root / 'configs/candidate_protocol.json').read_text())
    w = json.loads((root / 'metadata/w_selection.json').read_text())
    identity = model_weight_identity(w['selected']['checkpoint'], adapter=True)
    assert identity == w['selected_identity']
    protocol = {'status': 'frozen', 'created_at': datetime.now(timezone.utc).isoformat(),
        'source_commit': args.source_commit, 'file_sha256': {str(p): sha256_file(p) for p in paths},
        'W': identity, 'base': model_weight_identity('models/Qwen3-VL-2B-Instruct'),
        'versions': package_versions(), 'candidate': candidate,
        'sft_sufficiency_policy': candidate['sft_sufficiency_policy'],
        'pilot_summary': 'results/same_w_pilot_summary.json',
        'blank_control': {'n': 20, 'exact': sum(r['score']['exact'] for r in rows) / 20},
        'rental': json.loads((root / 'metadata/budget_policy.json').read_text()),
        'human_input_audit': 'pending: team temporarily unavailable; fixed samples retained, no semantic filtering',
        'resume_limitation': 'State, positions, RNG and LR checked; cross-process BF16 numeric trajectories not bitwise equivalent',
        'formal_test_model_predictions_used_for_selection': False}
    write_json(destination, protocol)
    print(json.dumps({'status': 'frozen', 'path': str(destination), 'sha256': sha256_file(destination)}))


if __name__ == '__main__':
    main()
