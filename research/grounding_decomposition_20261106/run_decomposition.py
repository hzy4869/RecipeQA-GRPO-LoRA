#!/usr/bin/env python3
"""DRAFT paired32 A/B/C/Cpi diagnostic: metadata preparation is CPU-only.

The run subcommand refuses draft policies. No source is edited, model downloaded,
reward changed or training performed. Root must separately review/freeze budget.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
from datetime import datetime,timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'research/grounding_equivariance_20261106'))
from recipeqa.modeling import (InputConfig,encode_messages,ensure_data_disk,hash_json,load_images,load_model,load_processor,
   model_weight_identity,package_versions,read_jsonl,sha256_file,write_json)
from recipeqa.metrics import parse_order,score_order
from orbit_diagnostic import (remaining_budget,install_budget_alarm,cancel_budget_alarm,
   install_termination_handlers,restore_termination_handlers)
from decomposition import (LABELS,CYCLE,prepare_metadata,rebuild_visible_recipe,validate_legal_b_format,
   oracle_table,score_b,score_oracle,require)

MODEL_ORDER=('sft44','exactkl44');CONDITIONS=('A','B','C','Cpi')
DEV_SHA='c998796fbc07a24c7a24062d6d0aba07500c75b34568fd7ba67f8c38daae03c9'
LOCK_NAMES=('formal_training_pipeline','evaluation_pipeline','evaluation_handoff','seed43_pipeline',
            'reward_search_20261105','finite_trials_20261105','orbit_diagnostic_20261106','projection_20261106','grounding_decomposition_20261106')
THREADS=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
EXPECTED_CPU_TESTS=25
SOURCE_PATHS=('research/grounding_decomposition_20261106/run_decomposition.py',
 'research/grounding_decomposition_20261106/decomposition.py',
 'research/grounding_decomposition_20261106/build_draft_policy.py',
 'research/grounding_decomposition_20261106/launch_stage.py',
 'research/grounding_decomposition_20261106/test_decomposition.py',
 'research/grounding_equivariance_20261106/orbit_diagnostic.py',
 'src/recipeqa/modeling.py','src/recipeqa/metrics.py','src/recipeqa/process_metrics.py')


def now(): return datetime.now(timezone.utc).isoformat()
def load(path): return json.loads(Path(path).read_text())


def cpu_test_receipt(completed,expected_tests):
    """Bind a non-vacuous unittest count and all terminal outcome counts.

    A zero-exit test process alone is insufficient: empty collection, missing
    final result, duplicate run summaries, skips or an unexpected suite size
    all make the preflight fail.
    """
    report=completed.stderr
    counts=re.findall(r'^Ran (\d+) tests? in [^\n]+$',report,flags=re.M)
    tests_run=int(counts[0]) if len(counts)==1 else None
    final=report.strip().splitlines()[-1] if report.strip() else ''
    outcome=re.fullmatch(r'(OK|FAILED)(?: \(([^)]*)\))?',final)
    totals={'skipped':0,'failures':0,'errors':0}
    valid_totals=True
    if outcome and outcome.group(2):
        for part in outcome.group(2).split(', '):
            match=re.fullmatch(r'(skipped|failures|errors)=(\d+)',part)
            if match is None:valid_totals=False
            else:totals[match.group(1)]=int(match.group(2))
    passed=bool(completed.returncode==0 and tests_run==expected_tests and tests_run>0
        and outcome and outcome.group(1)=='OK' and valid_totals and not any(totals.values()))
    return {'returncode':completed.returncode,'stdout':completed.stdout,'stderr':completed.stderr,
        'expected_tests':expected_tests,'tests_run':tests_run,**totals,'passed':passed}

def metadata_receipt(matched):
    return {'status':'cpu_metadata_prepared_not_model_inference','n_recipes':32,'model_order':list(MODEL_ORDER),
      'conditions':list(CONDITIONS),'new_multimodal_calls':128,'new_text_only_calls':128,'total_new_calls':256,
      'selection':'unchanged original dev200 file first32, before model score inspection',
      'cohort':[{'qid':item['record']['qid'],'recipe_id':item['record']['recipe_id'],
        'presentation_id':item['record']['presentation_id'],'gold_visible_steps':item['gold_visible_steps'],
        'context_source_ids':[s['source_step_id'] for s in item['record']['steps']],
        'uniform_body_cap':item['historical_A']['input_audit']['uniform_body_cap'],
        'body_clipped':bool(item['historical_A']['input_audit']['body_clipped_step_ids'])} for item in matched['sft44']],
      'limits':['Source-step agreement is not independently human-verified visual uniqueness.',
        'C/Cpi change the context and are behavioral oracle-table controls, not deployment scores or internal causal attribution.',
        'Historical A input hashes do not include full pixel bytes; new contemporaneous A is the primary paired input control.']}


def tensor_digest(tensor):
    """Exact current CPU tensor bytes; no GPU allocation."""
    require(tensor.device.type=='cpu','Input proof must be computed before moving tensors to GPU')
    import torch
    value=tensor.detach().contiguous();digest=hashlib.sha256()
    digest.update(json.dumps([str(value.dtype),list(value.shape)]).encode())
    raw=value.view(-1).view(torch.uint8)
    for start in range(0,raw.numel(),8*1024*1024):digest.update(raw[start:start+8*1024*1024].numpy().tobytes())
    return digest.hexdigest()


def recipe_and_pixel_preflight(record,original_a,processor,image_root):
    """Reconstruct recorded A bytes and prove contemporaneous A/B visual equality."""
    import torch
    recipe,a_messages,b_messages=rebuild_visible_recipe(record,original_a['input_audit'],processor.tokenizer)
    images=load_images(record,image_root)
    a=encode_messages(processor,a_messages,images);b=encode_messages(processor,b_messages,images)
    audit=original_a['input_audit']
    require(a['input_ids'].shape[-1]==audit['input_tokens'],'Reconstructed A actual input length changed')
    require(a['image_grid_thw'].tolist()==audit['image_grid_thw'],'Reconstructed A actual image grid changed')
    if audit.get('image_token_count') is not None:
        require(int((a['input_ids']==processor.image_token_id).sum())==audit['image_token_count'],'Original image token count changed')
    for key in ('pixel_values','image_grid_thw'):
        require(key in a and key in b and torch.equal(a[key],b[key]),'A/B actual visual tensor differs: '+key)
    require(b['input_ids'].shape[-1]<=4096,'B exceeds frozen input budget; no silent text or image truncation allowed')
    require(len(recipe.encode())>0,'Visible recipe must not be empty')
    # B's first text block differs only in output-format instructions; the recipe
    # substring is literally the same UTF8 bytes, rather than only same metadata.
    a_recipe=a_messages[0]['content'][0]['text'].split('Ordered recipe:\n',1)[1]
    b_recipe=b_messages[0]['content'][0]['text'].split('Ordered recipe:\n',1)[1]
    require(a_recipe.encode()==b_recipe.encode()==recipe.encode(),'A/B visible recipe bytes differ')
    legal=validate_legal_b_format(processor.tokenizer,len(record['steps']),64)
    # Verify the assistant answer template, EOS and generation prefix boundary
    # for the legal B format without generating or consulting model outcomes.
    tokenizer=processor.tokenizer;eos=tokenizer.eos_token_id
    prefix=processor.apply_chat_template(b_messages,tokenize=False,add_generation_prompt=True)
    full=processor.apply_chat_template(b_messages+[{'role':'assistant','content':legal['sample_completion']}],tokenize=False,add_generation_prompt=False)
    prefix_ids=tokenizer.encode(prefix,add_special_tokens=False);full_ids=tokenizer.encode(full,add_special_tokens=False)
    require(full.startswith(prefix) and full_ids[:len(prefix_ids)]==prefix_ids,'B completion boundary is not prefix-stable')
    suffix=full_ids[len(prefix_ids):]
    require(eos in suffix,'B canonical assistant format has no EOS')
    through_eos=suffix[:suffix.index(eos)+1]
    require(len(through_eos)<=64,'Valid B assistant format including EOS exceeds64 tokens')
    proof={'qid':record['qid'],'presentation_id':record['presentation_id'],'recipe_utf8_sha256':hashlib.sha256(recipe.encode()).hexdigest(),
      'A_prompt_sha256':hash_json(a_messages),'B_prompt_sha256':hash_json(b_messages),'uniform_body_cap':audit['uniform_body_cap'],
      'A_input_tokens':int(a['input_ids'].shape[-1]),'B_input_tokens':int(b['input_ids'].shape[-1]),
      'pixel_values_sha256':tensor_digest(a['pixel_values']),'image_grid_thw':a['image_grid_thw'].tolist(),
      'same_current_A_B_pixel_values':True,'same_current_A_B_image_grid':True,'same_visible_recipe_utf8_bytes':True,
      'valid_B_assistant_tokens_including_eos':len(through_eos),'historical_A_recipe_and_grid_verified':True,
      'historical_full_pixel_bytes_not_retroactively_proven':True}
    return {'A':a_messages,'B':b_messages},proof


def estimate_remaining(pilot_rows,load_seconds):
    require(len(pilot_rows)==16 and len(load_seconds)==2,'Timing gate requires first2 recipes x4conditions x2models and both actual loads')
    keys={(row['model'],row['condition']) for row in pilot_rows}
    require(keys=={(model,c) for model in MODEL_ORDER for c in CONDITIONS},'Both models and every condition must be timed')
    cells={key:[r for r in pilot_rows if (r['model'],r['condition'])==key] for key in keys}
    require(all(len(values)==2 for values in cells.values()),'Each timing cell must contain2 fixed recipes')
    require({(row['model'],row['condition'],row['cohort_index']) for row in pilot_rows}
        =={(m,c,i) for m in MODEL_ORDER for c in CONDITIONS for i in range(2)},'Pilot must be exactly first2 fixed recipes, zero replacements')
    require(all(value>0 and math.isfinite(value) for value in load_seconds),'Actual model load durations must be positive and finite')
    for row in pilot_rows:require(row['total_seconds']>0 and math.isfinite(row['total_seconds']),'Invalid measured prediction duration')
    # Pilot rows are already canonical observations and are never sampled again.
    # Thirty recipes remain in every model/condition; two full model reloads remain.
    prediction=30*sum(max(r['total_seconds'] for r in values) for values in cells.values())
    return {'estimated_remaining_seconds':1.5*(prediction+sum(load_seconds))+120,
      'safety_factor':1.5,'cpu_io_allowance_seconds':120,'remaining_predictions':240,'remaining_model_loads':2,
      'components_seconds':{'predictions':prediction,'model_reloads':sum(load_seconds)},
      'scope':'Measured complete prepare+encode+generate+write time, not generation-only. Fixed hard deadline independently enforced; no score-based replacement.'}


def expected_keys():return {(model,index,condition) for model in MODEL_ORDER for index in range(32) for condition in CONDITIONS}


def summarize(rows):
    keys=[(r['model'],r['cohort_index'],r['condition']) for r in rows]
    require(len(keys)==len(set(keys))==256 and set(keys)==expected_keys(),'Complete256 distinct paired observations required; no partial summaries')
    require(all(not r.get('technical_error') for r in rows),'Technical failures make the diagnostic incomplete')
    result={}
    for model in MODEL_ORDER:
        cells={c:sorted([r for r in rows if r['model']==model and r['condition']==c],key=lambda r:r['cohort_index']) for c in CONDITIONS}
        A=cells['A'];B=cells['B'];b=[r['metrics'] for r in B]
        result[model]={'A':{k:sum(r['metrics'][k] for r in A)/32 for k in ('exact','pair','valid')},
          'B':{'source_step_match':sum(r['step_match_count'] for r in b)/128,'all_steps_match':sum(r['all_steps_match'] for r in b)/32,
             'mapping_pair':sum(r['mapping_pair_correct_count'] for r in b)/192,
             **{k:sum(r[k] for r in b)/32 for k in ('complete_in_range_mapping','unique_mapping','derived_exact','derived_pair','final_exact','final_pair','final_valid','format_valid','mapping_order_comparable','mapping_order_consistent')},
             'mapping_order_comparable_n':sum(r['mapping_order_comparable'] for r in b)},
          'C_oracle':{k:sum(r['metrics'][k] for r in cells['C'])/32 for k in ('exact','valid')},
          'Cpi_oracle':{k:sum(r['metrics'][k] for r in cells['Cpi'])/32 for k in ('exact','valid')},
          'B_issue_counts':{k:sum(r[k] for r in b) for k in ('unclear_or_missing_count','out_of_range_count','in_range_tied_pair_count')},
          'B_minus_A_exact':sum(bb['final_exact']-aa['metrics']['exact'] for aa,bb in zip(A,b))/32,
          'C_to_Cpi':{'both_exact_n':sum(c['metrics']['exact'] and cp['metrics']['exact'] for c,cp in zip(cells['C'],cells['Cpi'])),
                     'both_valid_and_answer_changed_n':sum(c['metrics']['valid'] and cp['metrics']['valid'] and c['metrics']['parsed_order']!=cp['metrics']['parsed_order'] for c,cp in zip(cells['C'],cells['Cpi']))},
          'A_historical_repeat':{'raw_equal_n':sum(r['historical_A_raw_equal'] for r in A),
                               'parsed_equal_n':sum(r['historical_A_parsed_equal'] for r in A),'n':32},
          'runtime_by_condition':{c:{'calls':len(cell),'completion_tokens':sum(r['completion_tokens'] for r in cell),
              'token_limit_reached_n':sum(r['reached_token_limit'] for r in cell),
              'prediction_seconds':sum(r['total_seconds'] for r in cell)} for c,cell in cells.items()},
          'body_clipping_strata':{stratum:{'n':len(indices),
             'A_exact':sum(A[i]['metrics']['exact'] for i in indices)/len(indices) if indices else None,
             'B_final_exact':sum(b[i]['final_exact'] for i in indices)/len(indices) if indices else None,
             'B_source_step_match':sum(b[i]['step_match_count'] for i in indices)/(4*len(indices)) if indices else None}
             for stratum,indices in [('unclipped',[i for i,r in enumerate(A) if not r['body_clipped']]),
                                     ('clipped',[i for i,r in enumerate(A) if r['body_clipped']])]},
          'per_recipe':[{'qid':a['qid'],'A':a['metrics'],'B':bb['metrics'],'C':c['metrics'],'Cpi':cp['metrics']}
             for a,bb,c,cp in zip(A,B,cells['C'],cells['Cpi'])]}
    return {'status':'completed_paired32_decomposition_development_only','results':result,
      'limits':['SFT/EK paired fixed64 flow44 only,32 known-development recipes; no population, independent-seed or hidden-CoT causal claim.',
                'C/Cpi are context-changing oracle-table behavioral tasks, not deployment image accuracy.',
                'B uses a longer response format and output budget than A; report actual tokens and timing.',
                'Source-step labels may be visually ambiguous; human semantic review remains pending.']}


def prepare_command(args):
    output=Path(args.output);require(not output.exists(),'Fresh metadata receipt required')
    matched=prepare_metadata(read_jsonl(args.manifest),{'sft44':read_jsonl(args.a_sft),'exactkl44':read_jsonl(args.a_ek)})
    write_json(output,metadata_receipt(matched))
    print(json.dumps({'status':'cpu_metadata_prepared','n_recipes':32,'model_inference':False,'output':str(output)}))


def run_command(args):
    policy_path=Path(args.policy).resolve();policy=load(policy_path)
    require(policy['status']=='locked_before_paired_grounding_decomposition','DRAFT ONLY: root must separately announce/review/freeze budget and policy')
    require(os.environ.get('RECIPEQA_GROUNDING_OUTER_WATCHDOG_POLICY_SHA256')==sha256_file(policy_path),
        'Formal run must be launched through the policy-bound native timeout watchdog')
    require(ROOT==Path('/root/autodl-tmp/rl_recipeqa') and ROOT.stat().st_dev!=Path('/').stat().st_dev,'Separate remote data disk required')
    require(policy['model_order']==list(MODEL_ORDER) and policy['conditions']==list(CONDITIONS) and policy['n_recipes']==32
            and policy['oracle_cycle']==list(CYCLE) and policy['new_A_is_primary'] is True,'Fixed paired protocol changed')
    require(policy['max_new_tokens']=={'A':32,'B':64,'C':32,'Cpi':32} and policy['wall_limit_seconds']==1800,'Fixed output/time budget changed')
    require(set(policy['sources_sha256'])==set(SOURCE_PATHS),'All and only reviewed scientific/helper/test sources must be bound')
    require(policy['updates']==0 and policy['optimizer_constructed'] is False and policy['new_calls']==256
            and policy['pilot_recipes']==2 and policy['seed']==44,'Read-only fixed paired protocol required')
    require(policy['input']==asdict(InputConfig()),'Frozen original input budget changed')
    require(policy['expected_cpu_tests']==EXPECTED_CPU_TESTS,'Frozen complete CPU suite size changed')
    require(policy['manifest']=='data/manifests/val.jsonl' and policy['manifest_sha256']==DEV_SHA,'Known dev200 only')
    required_artifacts={policy['manifest']}
    for spec in policy['models'].values():
        required_artifacts.update((spec['historical_A'],spec['historical_A']+'.config.json'))
    require(set(policy['artifacts_sha256'])==required_artifacts,'Bind exactly known dev and the two historical development prediction/config pairs')
    start=datetime.fromisoformat(policy['first_preflight_started_utc'].replace('Z','+00:00'))
    absolute=datetime.fromisoformat(policy['absolute_deadline_utc'].replace('Z','+00:00'))
    require(start.tzinfo is not None and absolute.tzinfo is not None and start<=datetime.now(timezone.utc)<absolute
          and 0<(absolute-start).total_seconds()<=1800,'New30min absolute deadline must begin at first preflight, never reset')
    for name in ('HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TMPDIR','TORCH_HOME'):
        require(os.environ.get(name) and Path(os.environ[name]).resolve().is_relative_to(ROOT),'Required cache/temp escaped data disk: '+name)
    for name in ('XDG_CACHE_HOME','PIP_CACHE_DIR','TRITON_CACHE_DIR','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','HF_MODULES_CACHE','TRANSFORMERS_CACHE'):
        require(not os.environ.get(name) or Path(os.environ[name]).resolve().is_relative_to(ROOT),'Active cache escaped data disk: '+name)
    require(Path(sys.prefix).resolve().is_relative_to(ROOT) and Path(sys.executable).absolute().is_relative_to(ROOT)
            and all(os.environ.get(n)=='4' for n in THREADS),'Reviewed data-disk Python and4threads required')
    def confined(value):
        path=(ROOT/value).resolve();require(path.is_relative_to(ROOT),'Path escaped data disk');return path
    destination=confined(policy['output_directory']);require(destination.is_relative_to(ROOT/'exploratory/grounding_decomposition_20261106') and not destination.exists(),'Fresh output required; no resume/retry')
    destination.mkdir(parents=True);state={'status':'preflight','pid':os.getpid(),'started_utc':now(),'active_model':None,'completed_predictions':0,
       'policy_sha256':sha256_file(policy_path),'automatic_retry':False,'resume':False,'updates':0,'optimizer_constructed':False,'test_access':False,'confirmation_access':False}
    phase_deadline=time.monotonic()+1800-(datetime.now(timezone.utc)-start).total_seconds()
    locks=[];alarm_previous=None;rows=[];seen=set();model=None;processor=None
    def update(**values):state.update(values,updated_utc=now());write_json(destination/'state.json',state)
    def remaining():return remaining_budget(phase_deadline,absolute)
    def immutable():
        require(sha256_file(policy_path)==state['policy_sha256'],'Policy changed')
        for group in ('sources_sha256','artifacts_sha256'):
            for path,digest in policy[group].items():require(sha256_file(confined(path))==digest,'Source/input artifact changed: '+path)
        for path,digest in policy['image_sha256'].items():
            image=(confined(policy['image_root'])/path).resolve()
            require(image.is_relative_to(confined(policy['image_root'])) and sha256_file(image)==digest,'Image escaped data disk or bytes changed')
    previous_handlers=install_termination_handlers()
    try:
        alarm_previous=install_budget_alarm(remaining());update();immutable()
        for name in LOCK_NAMES:
            handle=(ROOT/'metadata'/(name+'.lock')).open('a+')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:handle.close();raise
            locks.append(handle)
        import subprocess
        check=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=min(15,remaining()))
        require(check.returncode==0 and not check.stdout.strip(),'GPU idle state not established')
        cpu=subprocess.run([sys.executable,str(HERE/'test_decomposition.py')],capture_output=True,text=True,
            timeout=min(60,remaining()),env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        cpu_receipt=cpu_test_receipt(cpu,policy['expected_cpu_tests'])
        write_json(destination/'cpu_tests.json',{**cpu_receipt,
            'suite_sha256':sha256_file(HERE/'test_decomposition.py'),'cuda_visible_devices':'','updates':0})
        require(cpu_receipt['passed'],'Exact complete CPU suite count must pass with zero skips/failures/errors')
        matched=prepare_metadata(read_jsonl(confined(policy['manifest'])),{name:read_jsonl(confined(policy['models'][name]['historical_A'])) for name in MODEL_ORDER})
        require(policy['cohort_qids']==[item['record']['qid'] for item in matched['sft44']],'Fixed first32 cohort identity changed')
        write_json(destination/'cohort_metadata.json',metadata_receipt(matched))
        required_images={path for item in matched['sft44'] for path in item['record']['image_paths_display']}
        require(set(policy['image_sha256'])==required_images,'Every fixed cohort image byte hash must be frozen before prediction')
        import torch
        processor=load_processor(str(confined(policy['model_path'])),policy['revision'],InputConfig())
        prompts=[];proofs=[]
        for item in matched['sft44']:
            remaining();messages,proof=recipe_and_pixel_preflight(item['record'],item['historical_A'],processor,str(confined(policy['image_root'])))
            for condition in ('C','Cpi'):
                values,expected,msg=oracle_table(item['gold_visible_steps'],condition);messages[condition]=msg
            prompts.append(messages);proofs.append(proof)
        write_json(destination/'rendered_and_pixel_preflight.json',{'proofs':proofs,'cuda_initialized':torch.cuda.is_initialized(),'versions':package_versions()})
        require(torch.cuda.is_initialized() is False,'Input preflight unexpectedly initialized CUDA')
        write_json(destination/'visible_prompts.json',prompts)
        base_identity=model_weight_identity(confined(policy['model_path']));require(base_identity==policy['base_identity'],'Base weights changed')
        for name in MODEL_ORDER:
            spec=policy['models'][name];expected_method='raw_sft' if name=='sft44' else 'exact_kl'
            require(spec['checkpoint'].endswith('/flow44/'+expected_method+'/checkpoint-64'),'Fixed64 paired endpoints only')
            identity=model_weight_identity(confined(spec['checkpoint']),adapter=True);require(identity==spec['identity'],'Adapter bytes changed')
            config=load(confined(spec['historical_A']+'.config.json'))
            require(config['weight_identity']=={'base':base_identity,'adapter':identity} and config['manifest_sha256']==DEV_SHA
                    and config['input']==asdict(InputConfig()) and config['cli']['seed']==44,'Historical A model/input identity differs')
        raw=destination/'raw_predictions.jsonl';raw.open('x').close()
        timings=destination/'timing_events.jsonl';timings.open('x').close()
        def encode(condition,index):
            if condition in ('A','B'):
                record=matched['sft44'][index]['record'];images=load_images(record,str(confined(policy['image_root'])))
                encoded=encode_messages(processor,prompts[index][condition],images)
                require(encoded['image_grid_thw'].tolist()==proofs[index]['image_grid_thw']
                      and tensor_digest(encoded['pixel_values'])==proofs[index]['pixel_values_sha256'],'Actual inference pixels/grid differ from CPU proof')
            else:
                template=processor.apply_chat_template(prompts[index][condition],tokenize=False,add_generation_prompt=True)
                encoded=processor(text=[template],return_tensors='pt',padding=False,truncation=False,add_special_tokens=False)
                encoded.pop('token_type_ids',None)
            require(encoded['input_ids'].shape[-1]<=4096,'Actual input exceeds4096; no truncation')
            return encoded
        def predict(name,index,condition):
            require((name,index,condition) not in seen,'Duplicate prediction; no retry')
            remaining();started=time.monotonic();item=matched[name][index];record=item['record'];encoded=encode(condition,index).to(model.device)
            prefix=encoded['input_ids'].shape[-1];generation=copy.deepcopy(model.generation_config)
            generation.do_sample=False;generation.temperature=None;generation.top_p=None;generation.top_k=None
            generation.max_new_tokens=policy['max_new_tokens'][condition];generation.use_cache=True;generation.pad_token_id=processor.tokenizer.pad_token_id
            torch.manual_seed(44);torch.cuda.manual_seed_all(44);torch.cuda.synchronize();generation_started=time.monotonic()
            with torch.inference_mode():sequence=model.generate(**encoded,generation_config=generation)
            torch.cuda.synchronize();ids=sequence[0,prefix:]
            text=processor.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            row={'model':name,'checkpoint':policy['models'][name]['checkpoint'],'condition':condition,'cohort_index':index,
                 'qid':record['qid'],'recipe_id':record['recipe_id'],'presentation_id':record['presentation_id'],
                 'body_clipped':bool(item['historical_A']['input_audit']['body_clipped_step_ids']),
                 'raw_response':text,'technical_error':None,'input_tokens':int(prefix),'completion_tokens':int(ids.numel()),
                 'reached_token_limit':ids.numel()==generation.max_new_tokens,'generation_seconds':time.monotonic()-generation_started,
                 'prompt_sha256':hash_json(prompts[index][condition]),'input_proof_sha256':hash_json(proofs[index]),
                 'model_identity_sha256':hash_json({'base':base_identity,'adapter':policy['models'][name]['identity']}),
                 'decoding':{'do_sample':False,'max_new_tokens':generation.max_new_tokens}}
            if condition=='A':
                row['metrics']=score_order(text,record['display_perm'],record['target_order']);old=item['historical_A']['raw_response']
                row.update(historical_A_raw_equal=text==old,historical_A_parsed_equal=parse_order(text)==parse_order(old))
            elif condition=='B':row['metrics']=score_b(text,record)
            else:
                values,expected,_=oracle_table(item['gold_visible_steps'],condition);row['oracle_table']=values;row['metrics']=score_oracle(text,expected)
            row['prewrite_seconds']=time.monotonic()-started
            with raw.open('a') as handle:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
            seen.add((name,index,condition));rows.append(row);update(completed_predictions=len(rows))
            row['total_seconds']=time.monotonic()-started
            # Timing is recorded after the canonical raw row and state writes;
            # avoids pretending a row can contain the duration of writing itself.
            with timings.open('a') as handle:handle.write(json.dumps({'model':name,'cohort_index':index,
                'condition':condition,'total_seconds':row['total_seconds'],'utc':now()})+'\n')
            print(json.dumps({'model':name,'index':index,'condition':condition,'seconds':row['total_seconds']}),flush=True)
            del encoded,sequence,ids
            return row
        def load_one(name):
            nonlocal model
            remaining();immutable();started=time.monotonic()
            model=load_model(str(confined(policy['model_path'])),policy['revision'],adapter_path=str(confined(policy['models'][name]['checkpoint'])),trainable=False,seed=44)
            require(not model.training,'Inference model unexpectedly in training mode');update(active_model=name)
            return time.monotonic()-started
        def unload():
            nonlocal model
            model=None;gc.collect();torch.cuda.empty_cache();update(active_model=None)
        pilot=[];loads=[]
        for name in MODEL_ORDER:
            loads.append(load_one(name))
            for index in range(2):
                for condition in CONDITIONS:pilot.append(predict(name,index,condition))
            unload()
        gate=estimate_remaining(pilot,loads);gate['remaining_seconds']=remaining();gate['passed']=gate['estimated_remaining_seconds']<gate['remaining_seconds']
        write_json(destination/'whole_paired_timing_gate.json',gate)
        require(gate['passed'],'Complete remaining paired batch does not fit; stop without resampling or dropping conditions')
        for name in MODEL_ORDER:
            load_one(name)
            for index in range(2,32):
                for condition in CONDITIONS:predict(name,index,condition)
            unload()
        remaining();immutable();summary=summarize(rows)
        require(model_weight_identity(confined(policy['model_path']))==base_identity,'Base bytes changed')
        for name in MODEL_ORDER:require(model_weight_identity(confined(policy['models'][name]['checkpoint']),adapter=True)==policy['models'][name]['identity'],'Adapter bytes changed')
        remaining();write_json(destination/'summary.json',summary)
        write_json(destination/'artifacts_receipt.json',{'files':[{'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha256_file(path)}
          for path in sorted(destination.rglob('*')) if path.is_file() and path not in (destination/'state.json',destination/'artifacts_receipt.json')]})
        remaining();update(status=summary['status'],summary_sha256=sha256_file(destination/'summary.json'))
    except BaseException as error:
        if alarm_previous is not None:cancel_budget_alarm(alarm_previous);alarm_previous=None
        update(status='failed_or_budget_stopped_no_retry',error=type(error).__name__+': '+str(error))
        raise
    finally:
        if alarm_previous is not None:cancel_budget_alarm(alarm_previous)
        for handle in reversed(locks):fcntl.flock(handle,fcntl.LOCK_UN);handle.close()
        restore_termination_handlers(previous_handlers)


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    cpu=commands.add_parser('prepare',help='CPU metadata only; no processor/model/GPU')
    for name in ('manifest','a-sft','a-ek','output'):cpu.add_argument('--'+name,required=True)
    run=commands.add_parser('run',help='Future separately locked budget only; draft rejected');run.add_argument('--policy',required=True)
    args=parser.parse_args()
    prepare_command(args) if args.command=='prepare' else run_command(args)


if __name__=='__main__':main()
