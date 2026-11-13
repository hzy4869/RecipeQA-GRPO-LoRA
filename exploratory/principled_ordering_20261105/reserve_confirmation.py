"""Reserve recipe-disjoint, previously unevaluated official-validation recipes.

This prepares inputs only. It never loads a model or evaluates a prediction.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from recipeqa.data import load_source, balanced_presentations, inspect_images, make_presentation
from recipeqa.modeling import read_jsonl, sha256_file, write_json


def priority(namespace, value):
    return hashlib.sha256((namespace + ':' + str(value)).encode()).hexdigest()


def main():
    if not ROOT.is_relative_to('/root/autodl-tmp') or ROOT.stat().st_dev == Path('/').stat().st_dev:
        raise RuntimeError('Reservation must remain on mounted data disk')
    output = ROOT / 'exploratory/principled_ordering_20261105/data'
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'CONFIRMATION_LOCK.json'
    if lock.exists():
        raise RuntimeError('Reservation already exists; do not replace a heldout cohort')
    old_path = ROOT / 'data/manifests/val.jsonl'
    if sha256_file(old_path) != 'c998796fbc07a24c7a24062d6d0aba07500c75b34568fd7ba67f8c38daae03c9':
        raise RuntimeError('Original development manifest changed')
    used = {row['recipe_id'] for row in read_jsonl(old_path)}
    source, report = load_source(ROOT / 'data/raw/val.json', 'val')
    if report['sha256'] != 'cca6c2c7a3f5516251a8eb86e6c26548a56cebff871a8fba330b51bc788c4e04':
        raise RuntimeError('Official validation source changed')
    pool = defaultdict(list)
    for row in source:
        if row['recipe_id'] not in used:
            pool[row['recipe_id']].append(row)
    recipes = sorted(pool, key=lambda value: priority('confirmation-recipe-20261112', value))[:200]
    if len(recipes) != 200:
        raise RuntimeError('Fewer than 200 unused independent validation recipes')
    records = [min(pool[recipe], key=lambda row: priority('confirmation-question-20261112', row['qid']))
               for recipe in recipes]
    presentations = balanced_presentations(records, 20261113, 'confirmation-unseen-val200')
    names = [name for row in records for name in row['images_chronological']]
    image_info, image_report = inspect_images(names, ROOT / 'data/images', verify=True)
    if image_report['issues']:
        write_json(output / 'confirmation_image_issues.json', image_report)
        raise RuntimeError('Reserved images have technical issues; retained, not replaced')
    manifest = [make_presentation(row, presentations[row['qid']],
                manifest_set='confirmation_val200_unused', image_info=image_info) for row in records]
    path = output / 'confirmation200.jsonl'
    with path.open('x') as stream:
        for row in manifest:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    write_json(output / 'confirmation_image_report.json', image_report)
    metadata = {
        'status': 'reserved_before_new_diagnostic_or_training_results',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'questions': 200, 'recipes': 200, 'available_unused_recipes': len(pool),
        'source': report, 'excluded_original_dev_recipes': len(used),
        'manifest': str(path), 'manifest_sha256': sha256_file(path),
        'selection_source_sha256': sha256_file(__file__),
        'recipe_seed': 20261112, 'presentation_seed': 20261113,
        'recipe_disjoint_from_original_development': not (set(recipes) & used),
        'prediction_status': 'not_run',
        'use_policy': 'Do not infer on this cohort or inspect model-specific performance until final methods, seeds and selection rules are separately frozen. This is independent confirmation, not an official test leaderboard.',
        'original_test_seen_before_this_extension': True,
        'human_ambiguity_labels': 'pending; no semantic filtering or AI replacement',
    }
    write_json(lock, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
