"""Lean CPU definitions for a DRAFT paired photo-to-step decomposition probe.

Backend source IDs and oracle tables never enter the B visual prompt. Existing
model/E4 implementations are imported read-only; this module adds a new parser.
"""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations
import re
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from recipeqa.modeling import InputConfig,_render_recipe,build_messages,hash_json
from recipeqa.metrics import parse_order,score_order
from recipeqa.process_metrics import gold_display_steps

LABELS=('A','B','C','D')
CYCLE=(1,2,3,0)
B_INSTRUCTION=("Match each photograph to the numbered Step shown in this recipe. "
 "Use a displayed step number, or 'unclear' if you cannot identify one. "
 "Then return the labels in recipe order. Return exactly these five lines, with no observations or other text:\n"
 "A: step <number or unclear>\nB: step <number or unclear>\nC: step <number or unclear>\nD: step <number or unclear>\nOrder: <four image labels>")
ORACLE_INSTRUCTION=("The table gives authoritative numerical step assignments. Some tables may be synthetic. "
 "Return the image labels in increasing numerical step order. Use each label exactly once. "
 "Output only four labels separated by spaces.\n\n")


def require(condition,message):
    if not condition: raise ValueError(message)


def source_mapping(record):
    require(record.get('source_split')=='val' and record.get('manifest_set')=='val','Only the existing development manifest is allowed')
    ids=[s['source_step_id'] for s in record['steps']]
    require(all(type(s) is int for s in ids) and len(set(ids))==len(ids),'Context source IDs must be distinct integers')
    gold,chronological=gold_display_steps(record)
    require(len(gold)==4 and len(set(gold))==4 and all(type(s) is int and 1<=s<=len(ids) for s in gold),
            'Four distinct in-context visible ordinals required')
    require(tuple(sorted(range(4),key=lambda i:gold[i]))==parse_order(' '.join(record['target_labels'])),
            'Source-step order and original order gold disagree')
    return gold,chronological


def prepare_metadata(dev_rows,a_rows_by_model):
    require(len(dev_rows)==200 and len({r['recipe_id'] for r in dev_rows})==200,'Original development200 required')
    selected=dev_rows[:32]
    require(len({r['qid'] for r in selected})==len({r['recipe_id'] for r in selected})==32,'Fixed first32 must be distinct recipes/questions')
    require(set(a_rows_by_model)=={'sft44','exactkl44'},'Exactly the two fixed64 flow44 models are required')
    matched={}
    for model,rows in a_rows_by_model.items():
        bank={}
        for r in rows:
            key=(r['qid'],r['presentation_id'])
            require(key not in bank,'Duplicate historical A prediction')
            bank[key]=r
        require(len(bank)==200,'Historical A must cover complete development200')
        output=[]
        for record in selected:
            original=bank.get((record['qid'],record['presentation_id']))
            require(original is not None and not original.get('technical_error'),'Missing or technically failed historical A')
            require(original.get('prompt_kind')=='answer' and original.get('sample_index')==0
                and original['decoding']['do_sample'] is False and original['decoding']['max_new_tokens']==32,'A must remain original greedy answer-only32')
            for field in ('recipe_id','display_perm','target_order','target_labels'):
                require(original[field]==record[field],'A mapping differs: '+field)
            require(original['score']==score_order(original['raw_response'],record['display_perm'],record['target_order']),'A raw rescore differs')
            audit=original['input_audit']
            require(type(audit['uniform_body_cap']) is int and audit['uniform_body_cap']>=0
                    and audit['steps']==audit['titles_retained']==len(record['steps']),'Invalid original A input audit')
            gold,_=source_mapping(record)
            output.append({'record':record,'historical_A':original,'gold_visible_steps':gold})
        matched[model]=output
    for a,b in zip(matched['sft44'],matched['exactkl44']):
        require(a['historical_A']['input_audit']==b['historical_A']['input_audit'],'Paired models did not receive identical original inputs')
    return matched


def b_messages(recipe_text):
    result=build_messages(recipe_text,process=False)
    first=result[0]['content'][0]['text']
    old='Use each label exactly once. Output only four labels separated by spaces.'
    require(first.count(old)==1,'Frozen answer-only instruction changed')
    result[0]['content'][0]['text']=first.replace(old,'Use each label exactly once in the final order.')
    require(result[0]['content'][-1]=={'type':'text','text':'\nAnswer:'},'Frozen answer marker changed')
    result[0]['content'][-1]={'type':'text','text':B_INSTRUCTION}
    return result


def rebuild_visible_recipe(record,audit,tokenizer):
    """No fresh clipping search: use exactly the recorded A body cap."""
    recipe=_render_recipe(record['steps'],tokenizer,audit['uniform_body_cap'])
    original=build_messages(recipe,process=False)
    require(hash_json(original)==audit['prompt_text_hash'],'Original A recipe/prompt hash did not reconstruct')
    require(len(tokenizer.encode(recipe,add_special_tokens=False))==audit['recipe_tokens_used'],'Original recipe token count changed')
    original_hash=hash_json({'messages':original,'display_perm':record['display_perm'],
      'image_paths_display':record['image_paths_display'],'preprocessing':asdict(InputConfig())})
    require(original_hash==audit['input_hash'],'Original A input coordinate/hash differs')
    return recipe,original,b_messages(recipe)


def validate_legal_b_format(tokenizer,max_step,budget=64):
    require(type(max_step) is int and max_step>=4 and budget==64,'Predeclared B max64 and at least four steps required')
    candidates=['\n'.join(f'{label}: step {value}' for label in LABELS)+'\nOrder: D C B A'
                for value in [*range(1,max_step+1),'unclear']]
    longest=max(candidates,key=lambda text:len(tokenizer.encode(text,add_special_tokens=False)))
    # Plain completion plus EOS length; the actual assistant template boundary
    # is verified separately before generation, without adapting to model scores.
    ids=tokenizer.encode(longest,add_special_tokens=False)
    count=len(ids)+1
    require(count<=budget,'A legal five-line answer itself exceeds the frozen64-token budget')
    return {'sample_completion':longest,'plain_answer_plus_eos_tokens':count,'budget':budget}


def oracle_table(gold_steps,condition):
    require(condition in ('C','Cpi') and len(gold_steps)==len(set(gold_steps))==4
       and all(type(s) is int and s>=1 for s in gold_steps),'Four distinct visible ordinals required')
    values=list(gold_steps) if condition=='C' else [gold_steps[i] for i in CYCLE]
    expected=tuple(sorted(range(4),key=lambda i:values[i]))
    table='\n'.join(f'{label}: {number}' for label,number in zip(LABELS,values))
    messages=[{'role':'user','content':[{'type':'text','text':ORACLE_INSTRUCTION+table}]}]
    return values,expected,messages


def parse_b(text):
    lines=[s.strip() for s in text.strip().splitlines() if s.strip()] if isinstance(text,str) else []
    fields=defaultdict(list);orders=[];observed=[];unknown=[]
    for line in lines:
        mapping=re.fullmatch(r'([A-D]):\s*step\s+([1-9]\d*|unclear)',line,flags=re.I)
        order=re.fullmatch(r'Order:\s*(.*)',line,flags=re.I)
        if mapping:
            label,value=mapping.groups();label=label.upper();observed.append(label)
            fields[label].append(None if value.lower()=='unclear' else int(value))
        elif order:
            orders.append(order.group(1));observed.append('Order')
        else: unknown.append(line)
    mappings={label:{'present':len(fields[label])==1,'duplicate':len(fields[label])>1,
               'step':fields[label][0] if len(fields[label])==1 else None} for label in LABELS}
    final=parse_order(orders[0]) if len(orders)==1 else None
    valid=(observed==[*LABELS,'Order'] and not unknown and all(m['present'] for m in mappings.values()) and final is not None)
    return {'mapping':mappings,'order':list(final) if final is not None else None,'order_valid':final is not None,
            'format_valid':valid,'unknown_lines':unknown,'order_field_count':len(orders)}


def score_b(text,record):
    gold,_=source_mapping(record); parsed=parse_b(text)
    predicted=[parsed['mapping'][label]['step'] for label in LABELS]
    present=[parsed['mapping'][label]['present'] for label in LABELS]
    in_range=[present[i] and type(s) is int and 1<=s<=len(record['steps']) for i,s in enumerate(predicted)]
    matches=[int(in_range[i] and predicted[i]==gold[i]) for i in range(4)]
    unique=all(in_range) and len(set(predicted))==4
    derived=tuple(sorted(range(4),key=lambda i:predicted[i])) if unique else None
    pair_correct=sum(bool(in_range[i] and in_range[j] and predicted[i]!=predicted[j]
                    and (predicted[i]<predicted[j])==(gold[i]<gold[j])) for i,j in combinations(range(4),2))
    order_score=score_order(parsed['order'],record['display_perm'],record['target_order'])
    derived_score=score_order(derived,record['display_perm'],record['target_order'])
    comparable=unique and parsed['order_valid']
    return {'parsed':parsed,'predicted_visible_steps':predicted,'gold_visible_steps':gold,'in_range':in_range,
      'unclear_or_missing_count':sum(s is None for s in predicted),
      'out_of_range_count':sum(present[i] and type(s) is int and not 1<=s<=len(record['steps']) for i,s in enumerate(predicted)),
      'in_range_tied_pair_count':sum(bool(in_range[i] and in_range[j] and predicted[i]==predicted[j]) for i,j in combinations(range(4),2)),
      'source_step_matches':matches,'step_match_count':sum(matches),'all_steps_match':int(all(matches)),
      'mapping_pair_correct_count':pair_correct,'mapping_pair':pair_correct/6,
      'complete_in_range_mapping':int(all(in_range)),'unique_mapping':int(unique),'derived_order':list(derived) if derived is not None else None,
      'derived_exact':derived_score['exact'],'derived_pair':derived_score['pair'],
      'mapping_order_comparable':int(comparable),'mapping_order_consistent':int(comparable and tuple(parsed['order'])==derived),
      'final_exact':order_score['exact'],'final_pair':order_score['pair'],'final_valid':order_score['valid'],
      'format_valid':int(parsed['format_valid'])}


def score_oracle(text,expected):
    parsed=parse_order(text)
    return {'valid':int(parsed is not None),'exact':int(parsed==tuple(expected)),
            'parsed_order':list(parsed) if parsed is not None else None,'expected_order':list(expected),
            'scope':'oracle text-table composition only; not image-task deployment accuracy'}
