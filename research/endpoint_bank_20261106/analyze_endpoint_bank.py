#!/usr/bin/env python3
"""Complete-only, fixed-theta/fixed-input endpoint mathematics; no inference."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import bank_math as m
import verify_endpoint_scores as v

HERE = Path(__file__).resolve().parent
STRATA = ('initial_MAP_correct', 'initial_wrong_qstar_MAP_correct', 'initial_wrong_qstar_MAP_wrong')
METRICS = ('qgold', 'qref_gold', 'KL_current_reference', 'common_beta_J', 'Jstar',
           'Jref', 'available_objective_gain', 'achieved_objective_gain',
           'Jstar_minus_J', 'KL_current_qstar', 'raw_sequence_NLL', 'raw_sequence_CE_div_C',
           'conditional_NLL', 'canonical_mass', 'log_canonical_mass', 'entropy_nats')


def required_commands():
    return {'cpu_preflight', 'cached_initial_sft768__train256_verify'} | {
        phase + '_' + task['id'] + suffix for task in v.fixed_tasks()
        for phase in ('pilot', 'formal') for suffix in ('', '_verify')}


def complete_gate(policy, state):
    # Must precede opening ANY new candidate score or greedy response file.
    m.require(state['status'] == 'completed_endpoint_bank_zero_update'
              and state['active_child_pid'] is None, 'Incomplete endpoint bank: refuse all new score reads')
    commands = state['commands']; expected = required_commands()
    m.require(len(commands) == len(expected) == 38
              and {x['stage'] for x in commands} == expected, 'Missing/duplicate/unplanned stage command')
    start = v.utc(policy['first_preflight_started_utc']); end = v.utc(policy['absolute_deadline_utc'])
    committed = v.utc(state['updated_utc']); begun = v.utc(state['started_utc'])
    m.require(start <= begun <= committed < end
              and state['first_preflight_started_utc'] == policy['first_preflight_started_utc']
              and state['absolute_deadline_utc'] == policy['absolute_deadline_utc'], 'Original stage clock reset')
    for command in commands:
        m.require(command['returncode'] == 0 and command['pgid'] == state['pgid']
                  and begun <= v.utc(command['started_utc']) <= v.utc(command['finished_utc']) <= committed,
                  'Failed/detached/after-commit command')
    m.require(state['formal_tasks'] == state['pilot_tasks'] == 9
              and state['formal_rows'] == 2024 and state['pilot_rows'] == 18
              and state['cached_initial_train_rows'] == 256
              and state['candidate_forwards'] == 51050 and state['model_loads'] == 18
              and state['optimizer_steps'] == state['backward_calls'] == state['generation_calls'] == 0
              and all(state[k] is False for k in ('automatic_retry', 'resume', 'test_access', 'confirmation_access')),
              'Complete stage counts/update/heldout scope differs')


def bind_remote_verification(files, policy, directory, checked, policy_path):
    path = Path(directory) / 'independent_verification.json'; receipt = files.json(path)
    m.require(receipt['passed'] is True and receipt['policy_sha256'] == checked['policy_sha256']
              and receipt['verifier_sha256'] == policy['sources_sha256']['research/endpoint_bank_20261106/verify_endpoint_scores.py']
              and receipt['physical_pixels_and_model_files_verified'] is True,
              'Actual remote independent physical/math audit missing')
    for key in ('phase', 'task_id', 'rows', 'signature_sha256', 'output_sha256',
                'candidate_forwards', 'optimizer_steps', 'backward_calls', 'generation_calls'):
        m.require(receipt[key] == checked[key], 'Remote independent receipt scope differs: ' + key)
    signed = files.json(Path(directory) / 'signature.json'); payload = signed['payload']
    required = {str(files.relative(policy_path))} | {
        str(Path(directory) / n) for n in ('signature.json', 'completion.json', 'scores.jsonl')}
    physical = {}
    for name, digest in payload['pixel_sha256'].items():
        physical[str(files.relative(Path(policy['image_root']) / name))] = {'sha256': digest}
    for kind in ('base', 'adapter'):
        item = payload['weight_identity'][kind]
        for spec in item['weight_files'] + item.get('auxiliary_files', []):
            physical[str(files.relative(Path(item['directory']) / spec['name']))] = spec
    required.update(physical)
    if checked['phase'] == 'cached':
        required.add(str(Path(directory) / 'lineage.json'))
        required.update(policy['cached_initial_train']['source_receipt_sha256'])
    remote_inputs = receipt['input_receipts']
    m.require(required.issubset(remote_inputs)
              and receipt['input_files_verified'] == len(remote_inputs), 'Remote physical/math proof omits core input receipts')
    for name, expected in physical.items():
        actual = remote_inputs[name]
        m.require(actual['sha256'] == expected['sha256'] and actual['bytes'] > 0
                  and ('bytes' not in expected or actual['bytes'] == expected['bytes']), 'Remote physical file receipt differs from locked identity')
    # The remote verifier hashed real image/model files. Missing local binaries
    # are not implicitly reread or relabelled as local physical/tensor evidence.
    bound = 0; remote_only = []
    for name, item in receipt['input_receipts'].items():
        if files.path(name).is_file():
            m.require(files.sha(name) == item['sha256'] and len(files.read(name)) == item['bytes'],
                      'Exported remote input receipt differs')
            bound += 1
        else:
            allowed = (Path(name).suffix in ('.safetensors', '.bin', '.jpg', '.jpeg', '.png', '.webp')
                       or Path(name).is_relative_to(policy['model_path'])
                       or any(Path(name).is_relative_to(e['adapter_path']) for e in policy['endpoints'].values()))
            m.require(allowed, 'Missing nonphysical remote evidence: ' + name)
            remote_only.append(name)
    return {'path': str(path), 'sha256': files.sha(path), 'exported_inputs_rehashed': bound,
            'remote_physical_files_not_locally_exported': remote_only}


def describe(rows):
    result = {'n_questions': len(rows), 'n_recipes': len({x['recipe_id'] for x in rows})}
    for key in METRICS:
        values = [x[key] for x in rows]
        result[key] = ({'mean': statistics.mean(values), 'min': min(values), 'max': max(values)}
                       if values else None)
    result.update(MAP_correct=sum(x['map']['map_exact'] for x in rows),
                  exact_tie_questions=sum(x['map']['tied_max_count'] > 1 for x in rows),
                  initial_exact_tie_questions=sum(x['initial_map']['tied_max_count'] > 1 for x in rows),
                  qstar_exact_tie_questions=sum(x['qstar_map']['tied_max_count'] > 1 for x in rows))
    available = math.fsum(x['available_objective_gain'] for x in rows)
    achieved = math.fsum(x['achieved_objective_gain'] for x in rows)
    result['objective_gain_realization'] = {
        'sum_available': available, 'sum_achieved': achieved,
        'ratio_of_sums_not_mean_per_question_ratio': achieved / available if available != 0 else None,
        'negative_values_not_clipped': True, 'zero_denominator_is_NA': True}
    return result


def existing_greedy(files, policy, endpoint, records, parsed):
    spec = policy['dev_greedy'][endpoint]
    m.require(spec['predictions'] in policy['artifacts_sha256'] and spec['config'] in policy['artifacts_sha256'],
              'Existing greedy raw/config must be frozen before this experiment')
    config = files.json(spec['config']); raw = files.rows(spec['predictions'])
    initial_config = files.json(policy['dev_greedy']['initial_sft768']['config'])
    identity = policy['endpoints'][endpoint]['weight_identity']
    m.require(config['weight_identity'] == identity and config['model_weights_sha256'] == v.obj_sha(identity)
              and config['input'] == policy['input'] and config['generation']['do_sample'] is False
              and config['manifest_sha256'] == policy['banks']['dev200']['source_manifest_sha256']
              and config['cli']['max_new_tokens'] == 32 and config['cli']['num_samples'] == 1
              and config['cli']['temperature'] == 0 and config['cli']['process'] is False
              and config['cli']['limit'] is None and config['generation'] == initial_config['generation'],
              'Existing open-greedy model/input/decoder differs')
    indexed = {(r['qid'], r['presentation_id']): r for r in records}
    scored = {(x['record']['qid'], x['record']['presentation_id']): x for x in parsed}
    m.require(len(raw) == len(indexed) == 200, 'Existing greedy200 denominator differs')
    result = {}; counts = Counter()
    for row in raw:
        key = (row['qid'], row['presentation_id'])
        m.require(key in indexed and key not in result, 'Existing greedy duplicate/unknown input')
        record = indexed[key]; candidate = scored[key]
        for field in ('qid', 'recipe_id', 'presentation_id', 'source_split', 'display_perm', 'target_order', 'target_labels'):
            m.require(row[field] == record[field], 'Existing greedy coordinate identity changed')
        m.require(row['technical_error'] is None and row['sample_index'] == 0 and row['prompt_kind'] == 'answer'
                  and row['seed'] == config['cli']['seed'] and row['run_id'] == config['cli']['run_id']
                  and row['decoding'] == {'do_sample': False, 'temperature': 0.0, 'max_new_tokens': 32, 'top_p': 1.0, 'top_k': 0}
                  and row['checkpoint'] == identity['adapter']['directory']
                  and row['model_weights_sha256'] == config['model_weights_sha256']
                  and row['input_audit'] == candidate['row']['input_audit'], 'Existing greedy input/weight/error differs')
        own = m.score_greedy(row['raw_response'], record)
        m.require(row['score']['valid'] == own['valid'] and row['score']['exact'] == own['exact']
                  and row['score']['semantic_order'] == own['semantic_order'], 'Existing greedy independent rescore differs')
        answer_index = (m.ACTIONS.index(' '.join(chr(65 + record['display_perm'].index(s))
                                               for s in own['semantic_order'])) if own['valid'] else None)
        own.update(candidate_index=answer_index, raw_response=row['raw_response'])
        result[key] = own
    return result


def analyze_all(files, policy_path):
    policy = files.json(policy_path); v.check_policy(policy); digest = files.sha(policy_path)
    root = Path(policy['output_directory']); state = files.json(root / 'state.json')
    complete_gate(policy, state)  # first gate; no partial scalar-score analysis.
    m.require(state['policy_sha256'] == digest, 'Stage policy/source lock differs')
    for module in ('bank_math.py', 'verify_endpoint_scores.py', 'analyze_endpoint_bank.py'):
        name = 'research/endpoint_bank_20261106/' + module
        m.require(hashlib.sha256((HERE / module).read_bytes()).hexdigest() == policy['sources_sha256'][name],
                  'Actually executing independent analysis source differs from frozen source')
    by_stage = {c['stage']: c for c in state['commands']}
    gate = files.json(root / 'whole_nine_task_gate.json')
    resource_check = v.verify_resource_gate(files, policy, gate)
    gate_done = v.utc(gate['completed_utc'])
    cpu_done = v.utc(by_stage['cpu_preflight']['finished_utc'])
    for task in v.fixed_tasks():
        timing = gate['pilot_timing_inputs'][task['id']]
        m.close(timing['command_wall_seconds'], by_stage['pilot_' + task['id']]['wall_seconds'], 'actual outer pilot command wall')
        m.require(cpu_done <= v.utc(by_stage['pilot_' + task['id']]['started_utc'])
                  and v.utc(by_stage['pilot_' + task['id'] + '_verify']['finished_utc']) <= gate_done
                  and gate_done <= v.utc(by_stage['formal_' + task['id']]['started_utc']),
                  'Formal/pilot work preceded actual full prerequisite completion')
    parsed = {}; receipts = {}
    for phase, tasks in (('cached', [{'id': 'initial_sft768__train256', 'endpoint': 'initial_sft768', 'bank': 'train256'}]),
                         ('pilot', v.fixed_tasks()), ('formal', v.fixed_tasks())):
        for task in tasks:
            directory = root / ('cached_initial_train' if phase == 'cached' else phase + '/' + task['id'])
            checked = v.verify_task(files, policy_path, phase, task['id'], directory, physical=False)
            receipts[phase + '__' + task['id']] = bind_remote_verification(files, policy, directory, checked, policy_path)
            if phase != 'pilot': parsed[(task['endpoint'], task['bank'])] = checked['_parsed']
    m.require(len(parsed) == 10 and sum(len(x) for x in parsed.values()) == 2280,
              'All five endpoint columns in both banks required')
    for bank in v.BANK_COUNTS:
        initial = parsed[('initial_sft768', bank)]
        for endpoint in v.ENDPOINTS:
            current = parsed[(endpoint, bank)]
            for left, right in zip(initial, current):
                m.require(left['record'] == right['record'] and left['row']['input_audit'] == right['row']['input_audit']
                          and left['row']['pixel_sha256'] == right['row']['pixel_sha256']
                          and left['row']['candidates'] == right['row']['candidates'],
                          'Fixed endpoint columns differ in inputs/pixels/encoding/candidate IDs')
    all_rows = []; summaries = {}; stratum_summaries = []; greedy_summary = {}
    for bank in v.BANK_COUNTS:
        initial = parsed[('initial_sft768', bank)]
        for endpoint in v.ENDPOINTS:
            source = parsed[(endpoint, bank)]; output = []
            greedy = (existing_greedy(files, policy, endpoint, [x['record'] for x in source], source)
                      if bank == 'dev200' else None)
            for cur, ref in zip(source, initial):
                record = cur['record']; result = m.metrics(cur['policy'], ref['policy'], cur['gold_index'])
                result.update(endpoint=endpoint, bank=bank, qid=record['qid'], recipe_id=record['recipe_id'],
                              presentation_id=record['presentation_id'],
                              common_J_is_counterfactual_for_rawSFT=endpoint in ('initial_sft768', 'flow44_raw_sft_256', 'flow45_raw_sft_256'),
                              greedy_available=greedy is not None)
                if greedy is None:
                    result.update(greedy_exact=None, greedy_valid=None, greedy_candidate_index=None,
                                  greedy_equals_conditional_MAP=None)
                else:
                    answer = greedy[(record['qid'], record['presentation_id'])]
                    result.update(greedy_exact=answer['exact'], greedy_valid=answer['valid'],
                                  greedy_candidate_index=answer['candidate_index'],
                                  greedy_equals_conditional_MAP=answer['candidate_index'] == result['map']['chosen_index'])
                output.append(result)
            all_rows.extend(output); summaries[endpoint + '__' + bank] = describe(output)
            for group in STRATA:
                subset = [r for r in output if r['stratum'] == group]
                stratum_summaries.append({'endpoint': endpoint, 'bank': bank, 'stratum': group, **describe(subset)})
            greedy_summary[endpoint + '__' + bank] = (
                {'n': 200, 'open_greedy_correct': sum(r['greedy_exact'] for r in output),
                 'conditional_MAP_correct': sum(r['map']['map_exact'] for r in output),
                 'same_answer': sum(r['greedy_equals_conditional_MAP'] for r in output),
                 'different_answer': sum(not r['greedy_equals_conditional_MAP'] for r in output),
                 'both_correct': sum(r['greedy_exact'] == r['map']['map_exact'] == 1 for r in output),
                 'only_conditional_MAP_correct': sum(r['map']['map_exact'] == 1 and r['greedy_exact'] == 0 for r in output),
                 'only_open_greedy_correct': sum(r['map']['map_exact'] == 0 and r['greedy_exact'] == 1 for r in output),
                 'both_wrong': sum(r['greedy_exact'] == r['map']['map_exact'] == 0 for r in output)}
                if greedy is not None else {'available': False, 'reason': 'No existing matched train greedy; no new generation.'})
    # Same reference and stratum membership in all endpoint columns, not five
    # separately data-selected partitions. Descriptive question means only.
    contrasts = []; indexed = {(r['endpoint'], r['bank'], r['qid'], r['presentation_id']): r for r in all_rows}
    for bank in v.BANK_COUNTS:
        for flow in ('flow44', 'flow45'):
            sft, rl = flow + '_raw_sft_256', flow + '_exact_kl_256'
            for ref in parsed[('initial_sft768', bank)]:
                rec = ref['record']; a = indexed[(sft, bank, rec['qid'], rec['presentation_id'])]
                b = indexed[(rl, bank, rec['qid'], rec['presentation_id'])]
                m.require(a['stratum'] == b['stratum'], 'Reference strata changed across methods')
                dq = b['qgold'] - a['qgold']; dkl = b['KL_current_reference'] - a['KL_current_reference']
                dj = b['common_beta_J'] - a['common_beta_J']
                m.close(dj, dq - m.BETA * dkl, 'Fixed-input endpoint J decomposition')
                contrasts.append({'flow': flow, 'bank': bank, 'qid': rec['qid'], 'recipe_id': rec['recipe_id'],
                                  'presentation_id': rec['presentation_id'], 'stratum': a['stratum'],
                                  'RL_minus_SFT_qgold': dq, 'RL_minus_SFT_KL': dkl,
                                  'RL_minus_SFT_J': dj, 'RL_minus_SFT_gap': b['Jstar_minus_J'] - a['Jstar_minus_J'],
                                  'RL_minus_SFT_MAP_exact': b['map']['map_exact'] - a['map']['map_exact']})
    m.require(len(all_rows) == 2280 and len(stratum_summaries) == 30 and len(contrasts) == 912,
              'Whole-bank output denominator differs')
    files.stable()
    return {'status': 'complete_fixed_endpoint_bank_analysis', 'policy_sha256': digest,
            'endpoints': list(v.ENDPOINTS), 'bank_counts': v.BANK_COUNTS, 'questions_per_endpoint': 456,
            'train_recipe_clusters': 254, 'dev_recipe_clusters': 200, 'cell_rows': 2280,
            'new_formal_input_scores': 2024, 'new_pilot_input_scores_separate': 18,
            'new_formal_candidate_forwards': 25 * 2024, 'new_pilot_candidate_forwards': 25 * 18,
            'all_new_stage_candidate_forwards': 51050, 'new_model_loads': 18,
            'cached_train_cells': 256, 'cached_train_new_candidate_forwards': 0,
            'historical_whole1024_reference_cost_not_free_or_recharged': 25600,
            'beta': m.BETA, 'odds_multiplier': math.exp(1 / m.BETA), 'tie_policy': m.TIE_POLICY,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'summaries': summaries, 'stratum_summaries': stratum_summaries,
            'existing_dev_greedy_comparison': greedy_summary, 'remote_task_verification_bindings': receipts,
            'input_receipts': files.inputs, 'new_model_GPU_network_calls': 0,
            'independently_recomputed_resource_gate': resource_check,
            'local_physical_model_pixel_reaudit': False,
            'limits': ['Fixed same-input endpoint observations, not online changing-theta means.',
                       'qstar uses gold and is not deployable, not a heldout accuracy upper bound and not a prediction used for the task.',
                       'qstar MAP wrong does not prohibit another lower-J policy from obtaining correct MAP. Small gap does not certify shared-parameter global optimality.',
                       'No large/small-gap threshold, checkpoint/flow selection or automatic new-method choice is made.',
                       'Train256 has254 recipe clusters, already seen by initialSFT. Dev200 is repeatedly reused; no freshconfirmation/test reads.',
                       'Conditional24MAP and old opengreedy use different decoding; their differences are not automatically RL gains.',
                       'Commonpositive-betaJ is counterfactual for SFT; rawCE/C and conditionalNLL remain separate.',
                       'Absent local image/model files rely on bound actual remote independent physical audits, not local tensor/CRC replay.',
                       'Questionwise descriptive summaries have no invented independent-recipe CI or fulltrainingseed uncertainty.'],
            '_cells': all_rows, '_strata': stratum_summaries, '_contrasts': contrasts}


def write_outputs(report, destination):
    destination = Path(destination); temporary = destination.with_name(destination.name + '.partial')
    m.require(not destination.exists() and not temporary.exists(), 'Fresh output only; preserve previous/partial evidence')
    temporary.mkdir(parents=True)
    for name, key in (('cells.csv', '_cells'), ('strata.csv', '_strata'), ('paired_endpoint_differences.csv', '_contrasts')):
        data = report.pop(key)
        with (temporary / name).open('w', newline='') as handle:
            flat = [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in data]
            writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    (temporary / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.rename(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=v.PROJECT / '.state/remote_principled_20261105')
    parser.add_argument('--output', type=Path, default=HERE / 'complete_analysis_v1')
    parser.add_argument('--require-all', action='store_true', help='All38commands/2280cells are always required.')
    args = parser.parse_args(argv); files = v.Files(args.root); report = analyze_all(files, args.policy)
    write_outputs(report, args.output); print(json.dumps({'status': report['status'], 'cell_rows': report['cell_rows']}))


if __name__ == '__main__':
    try: main()
    except (ValueError, KeyError, FileNotFoundError, AssertionError) as exc:
        print(json.dumps({'status': 'endpoint_bank_not_analyzed', 'error': type(exc).__name__ + ': ' + str(exc)}), file=sys.stderr)
        raise SystemExit(1)
