#!/usr/bin/env python3
"""Build an unexecutable policy draft from existing development inputs only.

No SSH, model/processor load, tensor allocation, training or inference. Image
bytes must be present under --data-root or separately supplied in an audited
SHA map. Only root's later review/budget/freeze can produce an executable policy.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from recipeqa.modeling import InputConfig,read_jsonl,sha256_file,write_json
from decomposition import CYCLE,prepare_metadata,require
from run_decomposition import MODEL_ORDER,CONDITIONS,DEV_SHA,SOURCE_PATHS,EXPECTED_CPU_TESTS

REMOTE_ROOT=Path('/root/autodl-tmp/rl_recipeqa')
HISTORICAL_PREFIX='exploratory/principled_ordering_20261105/finite_trials_v2/flow44'


def build(data_root,source_root,image_hashes=None):
    data_root=Path(data_root).resolve();source_root=Path(source_root).resolve()
    manifest='data/manifests/val.jsonl'
    require(sha256_file(data_root/manifest)==DEV_SHA,'Only unchanged known development200 allowed')
    histories={name:f'{HISTORICAL_PREFIX}/{method}/dev64/predictions.jsonl'
               for name,method in [('sft44','raw_sft'),('exactkl44','exact_kl')]}
    matched=prepare_metadata(read_jsonl(data_root/manifest),{name:read_jsonl(data_root/path) for name,path in histories.items()})
    models={};base=None;model_path=None;revision=None
    artifacts={manifest:DEV_SHA}
    for name,path in histories.items():
        config=json.loads((data_root/(path+'.config.json')).read_text())
        require(config['input']==asdict(InputConfig()) and config['cli']['seed']==44,'Historical input/seed differs')
        identity=config['weight_identity'];cli=config['cli']
        current_base=identity['base']
        if base is None:base=current_base;model_path=cli['model_path'];revision=cli['revision']
        require(current_base==base and cli['model_path']==model_path and cli['revision']==revision,'Paired base/revision differ')
        checkpoint=str(Path(cli['adapter_path']).relative_to(REMOTE_ROOT))
        expected='raw_sft' if name=='sft44' else 'exact_kl'
        require(checkpoint==f'{HISTORICAL_PREFIX}/{expected}/checkpoint-64','Only prechosen fixed64 flow44 endpoints')
        models[name]={'checkpoint':checkpoint,'identity':identity['adapter'],'historical_A':path}
        artifacts.update({item:sha256_file(data_root/item) for item in (path,path+'.config.json')})
    image_paths={p for item in matched['sft44'] for p in item['record']['image_paths_display']}
    if image_hashes is None:
        image_hashes={path:sha256_file(data_root/'data/images'/path) for path in sorted(image_paths)}
    require(set(image_hashes)==image_paths,'Exactly fixed32 cohort images must have SHA receipts')
    require(all(isinstance(d,str) and len(d)==64 and all(c in '0123456789abcdef' for c in d) for d in image_hashes.values()),'Malformed image digest')
    # This is intentionally not executable. No current clock is created here;
    # first-preflight time must be set once by the later reviewed launch wrapper.
    return {'schema_version':1,'status':'draft_not_authorized_for_execution',
        'research_question':'Does fixed64 Exact-KL change observable source-step matching, relative order, and self-consistency versus paired SFT44?',
        'model_order':list(MODEL_ORDER),'conditions':list(CONDITIONS),'n_recipes':32,
        'manifest':manifest,'manifest_sha256':DEV_SHA,'image_root':'data/images',
        'cohort_qids':[item['record']['qid'] for item in matched['sft44']],
        'input':asdict(InputConfig()),'new_A_is_primary':True,
        'oracle_cycle':list(CYCLE),'max_new_tokens':{'A':32,'B':64,'C':32,'Cpi':32},
        'new_calls':256,'pilot_recipes':2,'pilot_calls_included_in_full256':16,
        'remaining_calls_after_pilot':240,'seed':44,'updates':0,'optimizer_constructed':False,
        'expected_cpu_tests':EXPECTED_CPU_TESTS,
        'wall_limit_seconds':1800,'first_preflight_started_utc':None,'absolute_deadline_utc':None,
        'budget_cny_cap':0.8,'reference_cny_per_hour':1.6,'budget_authorization':None,
        'output_directory':'exploratory/grounding_decomposition_20261106/stage_v1',
        'model_path':str(Path(model_path).relative_to(REMOTE_ROOT)),'revision':revision,
        'base_identity':base,'models':models,'image_sha256':dict(sorted(image_hashes.items())),
        'artifacts_sha256':artifacts,
        'sources_sha256':{path:sha256_file(source_root/path) for path in SOURCE_PATHS},
        'fixed_scoring':{'out_of_range_or_missing_pairs':'wrong in denominator6',
            'ties':'wrong pair; no complete derived order',
            'final_order':'strict field parsing independent of mapping format validity',
            'denominators':{'questions_per_model':32,'photos_per_model':128,'pairs_per_model':192},
            'technical_failures':'incomplete stage; preserve all raw evidence; no retries'},
        'scope_limits':['Known development, one optimization flow, no independent-seed confirmation.',
            'Historical pixel bytes unavailable; contemporaneous A/B pixels and grid are actually compared.',
            'C/Cpi remove recipe/images and change context; not causal grounding attribution or deployment scores.',
            'Source-step labels are data provenance, with human semantic uniqueness audit pending.',
            'No test/confirmation inference or selection; no weight updates, reward changes or new trainer.'],
        'freeze_requirements':['Root independent source/scoring review and separate30min/0.8CNY announcement.',
            'Set clock once at first CPU preflight and absolute at most1800seconds later; preserve across fixes.',
            'Verify source/image/artifact SHA, data-disk paths, idle GPU and exclusive locks.',
            'Require complete zero-skip CPU suite, real processor pixel/text preflight, then all16 fixed pilot calls.',
            'Full remaining240 gate must pass; no score-based stopping/replacement or partial overall result.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--image-sha256-json',type=Path,help='Audited complete path->sha256 JSON map; missing images cannot be guessed')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();require(not args.output.exists(),'Fresh draft output required; never overwrite a frozen policy')
    hashes=json.loads(args.image_sha256_json.read_text()) if args.image_sha256_json else None
    policy=build(args.data_root,args.source_root,hashes);write_json(args.output,policy)
    print(json.dumps({'status':policy['status'],'new_calls':256,'model_or_processor_loaded':False,
                      'output':str(args.output),'sha256':sha256_file(args.output)}))


if __name__=='__main__':main()
