#!/usr/bin/env python3
"""Build a CPU-only, offline browser demo from verified, complete E3 predictions.

Use --check-only locally without photographs; generation requires the original
image-root. This is playback of recorded experiments, never live inference.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
from itertools import permutations
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from recipeqa.metrics import score_order, orbit_metrics
from recipeqa.modeling import hash_json, build_messages
from validate_run import validate_prediction_coverage, load_jsonl

METHODS = ('sft', 'exact', 'pair')
NAMES = {'sft': 'SFT-cont', 'exact': 'Exact-GRPO', 'pair': 'Pair-GRPO'}
LAYOUTS = list(permutations(range(4)))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    manifest = load_jsonl(args.manifest)
    summary = json.loads(args.summary.read_text())
    selections = json.loads(args.selections.read_text())['selected']
    views = {row['qid']: row for row in load_jsonl(args.review_view)}
    cases = {}
    for row in manifest:
        key = row['qid']
        case = cases.setdefault(key, {'recipe': row['recipe_id'], 'rows': {}, 'images': {}})
        layout = tuple(row['display_perm'])
        if layout in case['rows'] or case['recipe'] != row['recipe_id']:
            raise ValueError('Duplicate layout or conflicting recipe identity')
        case['rows'][layout] = row
        for semantic, path in zip(layout, row['image_paths_display']):
            if semantic in case['images'] and case['images'][semantic] != path:
                raise ValueError('One image changes identity across layouts')
            case['images'][semantic] = path
        view = views[key]
        without_body = lambda steps: [{k: v for k, v in step.items() if k != 'body'} for step in steps]
        if view['recipe_id'] != row['recipe_id'] or without_body(view['steps']) != without_body(row['steps']) or not view['review_text_view']['roundtrip_text_equal']:
            raise ValueError('Model-visible recipe view does not match this question')
    if len(cases) != 60 or len({c['recipe'] for c in cases.values()}) != 60 or any(set(c['rows']) != set(LAYOUTS) or set(c['images']) != set(range(4)) for c in cases.values()):
        raise ValueError('Exactly 60 independent recipes with all 24 layouts are required')
    expected_runs = {f'eval_{method}_seed42': method for method in METHODS}
    groups = {g['identity']['run_id']: g for g in summary['groups']}
    if set(groups) != set(expected_runs) or len(args.predictions) != 3:
        raise ValueError('The three primary seed42 models are required; no missing/extra groups')
    results, aggregates, configs = {}, {}, {}
    receipts = {key: digest(p) for key, p in zip(('input_manifest', 'result_summary', 'selected_checkpoints', 'model_visible_recipe'), (args.manifest, args.summary, args.selections, args.review_view))}
    for path in args.predictions:
        rows = load_jsonl(path)
        run = rows[0]['run_id'] if rows else None
        if run not in expected_runs or expected_runs[run] in results:
            raise ValueError('Unexpected/duplicate prediction model')
        method = expected_runs[run]
        checkpoint = selections[f'{method}_seed42']['checkpoint']
        config_path = path.with_suffix(path.suffix + '.config.json')
        config = json.loads(config_path.read_text())
        if config['manifest_sha256'] != digest(args.manifest):
            raise ValueError('Actual display/photo manifest differs from the recorded model input')
        configs[method] = config
        weights = hash_json(config['weight_identity'])
        if config['model_weights_sha256'] != weights:
            raise ValueError('Prediction config weight receipt is contradictory')
        gate = validate_prediction_coverage(manifest, rows, expected_run_id=run,
            expected_checkpoint=checkpoint, expected_seed=42, expected_questions=60,
            expected_recipes=60, require_orbit=True)
        if not gate['passed']:
            raise ValueError('Incomplete/contradictory E3 coverage: ' + json.dumps(gate['issues'][:3]))
        for row in rows:
            if row['model_id'] != 'Qwen/Qwen3-VL-2B-Instruct' or row['model_weights_sha256'] != weights or row['decoding']['do_sample'] is not False:
                raise ValueError('Model, actual-weight receipt or greedy decoding differs')
            if row['score'] != score_order(row['raw_response'], row['display_perm'], row['target_order']):
                raise ValueError('Stored score does not reproduce from the original response')
        computed = orbit_metrics([{**row, 'metrics': row['score']} for row in rows])
        stored = groups[run]
        if stored['identity']['checkpoint'] != checkpoint or stored['identity']['seed'] != 42 or computed != stored['orbit']:
            raise ValueError('Orbit summary or selected checkpoint differs from raw evidence')
        results[method] = {(row['qid'], tuple(row['display_perm'])): row for row in rows}
        aggregates[method] = computed['recipe_macro']
        receipts.update({f'raw_{method}': digest(path), f'config_{method}': digest(config_path)})
    questions = []
    counts = {method: {'high_pair_wrong_responses': 0, 'semantic_change_questions': 0} for method in METHODS}
    for index, (qid, case) in enumerate(sorted(cases.items()), 1):
        displayed = []
        flags = {'high_pair_wrong': False, 'semantic_change': False}
        for method in METHODS:
            sequence = [results[method][qid, layout]['score'] for layout in LAYOUTS]
            changed = len({tuple(s['semantic_order']) for s in sequence if s['valid']}) > 1
            high = sum(s['valid'] and not s['exact'] and s['pair'] >= 5 / 6 for s in sequence)
            counts[method]['high_pair_wrong_responses'] += high
            counts[method]['semantic_change_questions'] += int(changed)
            flags['high_pair_wrong'] |= high > 0
            flags['semantic_change'] |= changed
        for layout in LAYOUTS:
            predictions = {}
            for method in METHODS:
                row = results[method][qid, layout]
                score = row['score']
                predictions[method] = {'raw': row['raw_response'], 'valid': score['valid'],
                    'exact': score['exact'], 'pair': score['pair'], 'semantic': score['semantic_order']}
            displayed.append({'layout': layout, 'predictions': predictions,
                              'reference': case['rows'][layout]['target_labels']})
        steps = views[qid]['steps']
        text = '\n\n'.join(f"Step {i}: {s.get('title', '').strip()}" + ('\n' + s['body'] if s.get('body') else '') for i, s in enumerate(steps, 1))
        if hashlib.sha256(text.encode()).hexdigest() != views[qid]['review_text_view']['rendered_recipe_sha256']:
            raise ValueError('Displayed recipe text differs from its model-visible rendering receipt')
        messages = build_messages(text)
        for method in METHODS:
            config = configs[method]
            for layout in LAYOUTS:
                observed = results[method][qid, layout]['input_audit']
                expected_input = hash_json({'messages': messages, 'display_perm': list(layout),
                    'image_paths_display': case['rows'][layout]['image_paths_display'], 'preprocessing': config['input']})
                if observed['prompt_text_hash'] != hash_json(messages) or observed['input_hash'] != expected_input or observed['uniform_body_cap'] != views[qid]['review_text_view']['uniform_body_cap']:
                    raise ValueError('Displayed text/photos do not match the recorded model-visible input')
        questions.append({'id': f'Q{index:02d}', 'recipe': text, 'flags': flags,
            'images': [f'assets/Q{index:02d}-I{k + 1}.jpg' for k in range(4)],
            'source_images': [case['images'][k] for k in range(4)], 'displays': displayed,
            'K': {method: sum(item['predictions'][method]['exact'] for item in displayed) for method in METHODS}})
    return {'methods': METHODS, 'names': NAMES, 'aggregates': aggregates, 'counts': counts,
            'questions': questions, 'receipts': receipts}


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Same photos, different presentation</title>
<style>body{font:16px system-ui;margin:28px auto;max-width:1180px;padding:0 18px;color:#142230;background:#f7f9fc}h1{font-size:29px;margin-bottom:8px}.note{color:#526276;line-height:1.5}select,button{font:inherit;padding:8px;margin:4px}.photos{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.photos figure{margin:8px 0;background:white;padding:12px;border:1px solid #dce3ec;border-radius:10px}.photos img{width:100%;height:190px;object-fit:contain}.photos figcaption{font-weight:bold}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{padding:16px;background:white;border:1px solid #dce3ec;border-radius:10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:17px ui-monospace}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:3px}.grid button{font-size:11px;padding:3px;margin:0;border:1px solid #b9c5d4}.ok{background:#c6eadf}.bad{background:#f9d7cc}.invalid{background:#e2e5ea}.selected{outline:2px solid #1d4671}table{border-collapse:collapse;width:100%;background:white}th,td{padding:9px;text-align:left;border-bottom:1px solid #dce3ec}summary{cursor:pointer;padding:10px 0}#recipe{font:14px system-ui;line-height:1.5;max-height:350px;overflow:auto}footer{margin-top:25px;border-top:1px solid #ccd5e0;padding-top:14px;font-size:13px}@media(max-width:750px){.photos{grid-template-columns:repeat(2,1fr)}.cards{grid-template-columns:1fr}}</style>
<h1>Same photos, different presentation</h1><p class="note">Recorded outputs from Qwen3-VL-2B on 60 held-out RecipeQA questions, each with all 24 image presentations. This is offline playback of real experiments, not live inference. Training seed: 42.</p>
<table><thead><tr><th>Model</th><th>A: average correct</th><th>B: all 24 correct</th><th>C: semantic consistency</th><th>Mean K / 24</th></tr></thead><tbody id="overall"></tbody></table>
<p class="note">A averages correctness across presentations. B requires all 24 to be correct. C is the frequency of the most common valid image order; a model can be consistently wrong. Each independent sample is a recipe, not a presentation.</p>
<label>Filter <select id="filter"><option value="all">All questions</option><option value="high_pair_wrong">High Pair, wrong full order</option><option value="semantic_change">Changes between valid semantic orders</option></select></label><label>Question <select id="question"></select></label><label>Presentation <select id="display"></select></label><span id="coverage"></span>
<details><summary>Ordered recipe text visible to the model</summary><pre id="recipe"></pre></details><section class="photos" id="photos"></section><section class="cards" id="cards"></section>
<details><summary>Reference answer and image identities</summary><p id="reference"></p><p class="note">I1-I4 are fixed source-chronological image identities used here to compare semantic orders. The model received only photographs and A-D display labels. Changing labels changes the correct label string.</p></details>
<footer id="footer"></footer><script type="application/json" id="data">PAYLOAD</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),pct=x=>(100*x).toFixed(2)+'%';
function el(tag,text,parent){const e=document.createElement(tag);if(text!==null)e.textContent=text;if(parent)parent.appendChild(e);return e;}
for(const m of D.methods){const tr=el('tr',null,$('overall')),a=D.aggregates[m];for(const t of [D.names[m],pct(a.mean_exact),pct(a.all24_exact),pct(a.modal_consistency),a.success_count.toFixed(2)+'/24'])el('td',t,tr);}
for(let i=0;i<24;i++){const o=el('option',''+(i+1),$('display'));o.value=i;}
function filter(){const old=$('question').value,kind=$('filter').value;$('question').replaceChildren();for(let i=0;i<D.questions.length;i++){const q=D.questions[i];if(kind!=='all'&&!q.flags[kind])continue;const o=el('option',q.id,$('question'));o.value=i;}if([...$('question').options].some(o=>o.value===old))$('question').value=old;$('coverage').textContent=$('question').options.length+'/60 recipes';render();}
function render(){if(!$('question').options.length)return;const q=D.questions[+$('question').value],i=+$('display').value,s=q.displays[i];$('recipe').textContent=q.recipe;$('photos').replaceChildren();s.layout.forEach((id,p)=>{const f=el('figure',null,$('photos'));el('figcaption','Image '+String.fromCharCode(65+p),f);const im=el('img',null,f);im.src=q.images[id];im.alt='Displayed image '+String.fromCharCode(65+p);});$('cards').replaceChildren();for(const m of D.methods){const p=s.predictions[m],c=el('article',null,$('cards'));c.className='card';el('h2',D.names[m],c);el('pre',p.raw,c);el('p',(p.valid?(p.exact?'Correct full order':'Wrong full order'):'Invalid answer')+' | Pair '+pct(p.pair),c);el('p','Semantic order: '+(p.semantic?p.semantic.map(x=>'I'+(x+1)).join(' → '):'invalid'),c);el('p','K = '+q.K[m]+'/24 presentations correct',c);const g=el('div',null,c);g.className='grid';q.displays.forEach((x,j)=>{const z=x.predictions[m],b=el('button',''+(j+1),g);b.className=(z.valid?(z.exact?'ok':'bad'):'invalid')+(j===i?' selected':'');b.title='Presentation '+(j+1);b.onclick=()=>{$('display').value=j;render();};});} $('reference').textContent='Recorded gold label order: '+s.reference.join(' ');}
$('filter').onchange=filter;$('question').onchange=render;$('display').onchange=render;filter();
$('footer').textContent='Green: correct. Coral: wrong. Grey: invalid. All 60 questions and 24 presentations are available; default question is the first stored qid in lexicographic order. High-Pair filter uses valid Pair ≥ 5/6 with Exact = 0. Counts (out of 1440 responses / 60 recipes per model): '+D.methods.map(m=>D.names[m]+' '+D.counts[m].high_pair_wrong_responses+' high-Pair wrong responses; '+D.counts[m].semantic_change_questions+' recipes with multiple valid semantic orders').join(' | ')+'. Source hashes are in source_receipts.json. E4 human evidence assessment remains pending.';
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest', 'summary', 'selections', 'review-view', 'image-root', 'output-dir'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--predictions', type=Path, nargs=3, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    data = prepare(args)
    if args.check_only:
        print(json.dumps({'status': 'verified_complete_recorded_E3', 'recipes': 60, 'presentations_per_recipe': 24, 'models': 3, 'image_bytes_not_checked': True}))
        return
    out = args.output_dir.resolve()
    if Path('/root/autodl-tmp').exists() and not out.is_relative_to('/root/autodl-tmp'):
        raise ValueError('Remote output must stay on the data disk')
    assets = out / 'assets';assets.mkdir(parents=True, exist_ok=True)
    image_root = args.image_root.resolve()
    from PIL import Image
    photo_hashes = {}
    for q in data['questions']:
        for source, target in zip(q.pop('source_images'), q['images']):
            path = (image_root / source).resolve()
            if not path.is_relative_to(image_root) or not path.is_file():
                raise ValueError('Required source photo missing or outside image-root')
            destination = (out / target).resolve()
            if not destination.is_relative_to(out):
                raise ValueError('Demo asset destination escapes output directory')
            with Image.open(path) as image:
                image.verify()
            shutil.copyfile(path, destination)
            photo_hashes[target] = digest(destination)
    receipts = data.pop('receipts')
    payload = json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')
    (out / 'index.html').write_text(PAGE.replace('PAYLOAD', payload), encoding='utf-8')
    (out / 'source_receipts.json').write_text(json.dumps({'status': 'complete_recorded_E3_playback', 'source_sha256': receipts,
        'photo_sha256': photo_hashes, 'script_sha256': digest(__file__), 'human_evidence_audit': 'pending', 'new_model_inference': False}, indent=2) + '\n')
    print(json.dumps({'output': str(out / 'index.html'), 'photos': 240, 'recipes': 60, 'models': 3, 'new_model_inference': False}))


if __name__ == '__main__':
    main()
