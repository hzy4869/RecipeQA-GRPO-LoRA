from pathlib import Path
s=Path('research/coverage_20261106/coverage_train.py').read_text()
s=s.replace('"""Explicit1024 coverage derivative of frozen finite_policy_train.py.\n\nTraining coefficients,24candidate scoring, per-candidate autograd, clipping,\nAdamW and update loop are unchanged. New contracts permit4smoke/1024formal,\nbind separately frozen coverage policies/sources, and enforce shared16h clock.\nNo monkeypatch of old code, no new reward, no sample filtering or resume.\n"""', '"""Explicit two-pass reference-refresh derivative of frozen coverage trainer.\n\nKeeps EOS24 objective/autograd/Adam math; adds actual midpoint reference\nscoring with complete optimizer/grad/mode/RNG continuity, then final fixed\ntraining-bank scoring. New policy, namespace and original20h commonclock.\nNo heuristic reward, score filtering, checkpoint selection, resume or old edits.\n"""')
s=s.replace('import json\n','import json\nimport hashlib\n')
s=s.replace('from coverage_contract import','from refresh_contract import')
s=s.replace('"Coverage policy escaped data disk"','"Refresh policy escaped data disk"').replace('RECIPEQA_COVERAGE_OUTER_WATCHDOG','RECIPEQA_REFRESH_OUTER_WATCHDOG')
s=s.replace('"Coverage policy must be loaded before source binding"','"Refresh policy must be loaded before source binding"').replace('"Frozen coverage source changed"','"Frozen refresh source changed"')
a=s.index('def validate_budget(');z=s.index('\n\ndef append(',a)
s=s[:a]+'''def validate_budget(records, limit, max_steps, unique_count, passes):
    require((unique_count, passes, limit, max_steps) in ((4,1,4,1),(4,2,8,2),(256,2,512,128)),
            "Only fixed ordinary/path smoke or two-pass formal budgets")
    require(len(records) == unique_count and len({(str(r['qid']),str(r['presentation_id'])) for r in records}) == unique_count,
            "Unique input manifest differs; repeated visits are separate from question identity")
    for row in records:
        require(row.get('source_split') == 'train' and row.get('technical_status') == 'ok'
                and row.get('manifest_set') not in ('val','test','confirmation','orbit','process'), "Only original undamaged train input")
        require(len(row.get('target_labels',[])) == 4 and set(row['target_labels']) == set('ABCD'), "Bad canonical gold")
    return records
''' + s[z:]
s=s.replace('return argparse.Namespace(**cli, policy=str(path))','validate_authorization(_LOCKED_POLICY, path)\n    return argparse.Namespace(**cli, policy=str(path), arm=_LOCKED_POLICY["arm"], mode=_LOCKED_POLICY["mode"])')
s=s.replace('records = validate_budget(read_jsonl(args.manifest), args.limit, args.max_steps)','records = validate_budget(read_jsonl(args.manifest), args.limit, args.max_steps, args.unique_count, args.passes)')
s=s.replace('"Coverage child/common wall budget exhausted; preserve partial evidence, no resume"','"Refresh child/common wall budget exhausted; preserve partial evidence, no resume"')
s=s.replace('state = {"status": "preparing", "method": args.method,','state = {"status": "preparing", "method": args.method, "arm": args.arm, "mode": args.mode,')
a=s.index('        if args.reference_cache:\n            require(reference_config_path.is_file()')
z=s.index('        require(len(reference) == args.limit,',a)
s=s[:a]+'''        require(bool(args.reference_cache) and reference_config_path.is_file() and reference_completion_path.is_file(), "Sealed q0 cache and sidecars required")
        cached_spec = json.loads(reference_config_path.read_text())
        require(cached_spec == json.loads(json.dumps(reference_spec)), "q0 cache differs in actual weights/input/pixels/versions/scoring")
        completion = json.loads(reference_completion_path.read_text())
        require(completion.get('status') == 'complete_fixed_initial_reference' and completion.get('rows') == args.unique_count
                and completion.get('cache_sha256') == sha256_file(reference_path), "q0 cache incomplete or changed")
        reference = read_jsonl(reference_path)
''' +s[z:]
s=s.replace('require(len(reference) == args.limit, "Incomplete reference policy")','require(len(reference) == args.unique_count, "Incomplete unique q0 policy")')
s=s.replace('"Coverage fixed canonical C=5 includingEOS required"','"Refresh fixed canonical C=5 includingEOS required"')
s=s.replace('signature_payload = {"schema_version": 1, "method": args.method,','signature_payload = {"schema_version": 2, "method": args.method, "arm": args.arm, "mode": args.mode,\n                             "unique_count": args.unique_count, "passes": args.passes, "refresh_boundary_step": args.refresh_boundary_step,\n                             "visit_sequence_sha256": hash_json([{"visit_index": i, "pass_index": i // args.unique_count, "index_within_pass": i % args.unique_count, "record_sha256": hash_json(records[i % args.unique_count])} for i in range(args.limit)]),\n                             "authorization_sha256": sha256_file(ROOT / _LOCKED_POLICY["authorization_path"]),')
s=s.replace('"coverage_mode": _LOCKED_POLICY["mode"]','"refresh_mode": _LOCKED_POLICY["mode"]')
s=s.replace('"reference_lineage_sha256": sha256_file(ROOT / _LOCKED_POLICY["reference_lineage"]) if _LOCKED_POLICY["reference_lineage"] else None,','"reference_lineage_sha256": sha256_file(ROOT / _LOCKED_POLICY["q0_reference_lineage"]),')
s=s.replace('sha256_file(ROOT / _LOCKED_POLICY["generated_reference_receipt"]) if _LOCKED_POLICY["mode"] != "reference" else None','sha256_file(ROOT / _LOCKED_POLICY["generated_reference_receipt"])')
a=s.index('        if args.prepare_reference_only:\n');z=s.index('        optimizer = torch.optim.AdamW',a)
s=s[:a]+s[z:]
s=s.replace('        torch.cuda.reset_peak_memory_stats()','        active_reference = reference\n        active_reference_hash = reference_hash\n        reference_generation = 0\n        refresh_forward_candidates = 0\n        final_bank_forward_candidates = 0\n        refresh_event = None\n        torch.cuda.reset_peak_memory_stats()')
s=s.replace('"reference_policy_sha256": reference_hash,\n                       "state_sha256"', '"reference_policy_sha256": reference_hash, "active_reference_policy_sha256": active_reference_hash, "reference_generation": reference_generation,\n                       "state_sha256"')
s=s.replace('record, ref = records[visit_index], reference[visit_index]','unique_index = visit_index % args.unique_count\n                record, ref = records[unique_index], active_reference[unique_index]')
s=s.replace('visit = {"visit_index": visit_index, "visit_id": record.get("visit_id"),','visit = {"visit_index": visit_index, "pass_index": visit_index // args.unique_count, "index_within_pass": unique_index,\n                         "reference_row_index": unique_index, "reference_generation": reference_generation, "arm": args.arm,\n                         "q0_reference_policy_sha256": reference_hash, "active_reference_policy_sha256": active_reference_hash,\n                         "visit_id": record.get("visit_id"),')
s=s.replace('"exact_rewards": rewards, "reference_logq": ref["logq"], "reference_policy_sha256": reference_hash,','"exact_rewards": rewards, "reference_logq": ref["logq"], "reference_policy_sha256": active_reference_hash,')
s=s.replace('entry = {"global_step_before_update": step, "global_step": step + 1,','entry = {"global_step_before_update": step, "global_step": step + 1, "arm": args.arm, "reference_generation": reference_generation,\n                     "active_reference_policy_sha256": active_reference_hash,')
s=s.replace('"cumulative_candidate_backwards": total_backward_candidates}', '"cumulative_candidate_backwards": total_backward_candidates,\n                     "cumulative_refreshed_reference_candidate_forwards": refresh_forward_candidates}')
s=s.replace('                save_checkpoint(step + 1, output / f"checkpoint-{step + 1}")','                save_checkpoint(step + 1, output / f"checkpoint-{step + 1}")\n            if args.refresh_boundary_step == step + 1:\n                refresh_event, active_reference = build_boundary_reference(step + 1)\n                active_reference_hash = refresh_event["reference_policy_sha256"]\n                refresh_forward_candidates += refresh_event["candidate_forwards"]\n                reference_generation = 1\n                update(status="training", reference_generation=1, active_reference_policy_sha256=active_reference_hash, refresh_event=refresh_event)')
s=s.replace('require(actual_visits == [(record["qid"], record["presentation_id"]) for record in records], "Actual fixed-prefix exposure differs")','require(actual_visits == [(records[i % args.unique_count]["qid"], records[i % args.unique_count]["presentation_id"]) for i in range(args.limit)], "Actual two-pass fixed flow differs")\n        final_bank = build_final_bank(max_steps)\n        final_bank_forward_candidates = final_bank["candidate_forwards"]')
s=s.replace('        model.save_pretrained(output / "adapter_final", safe_serialization=True)\n        processor.save_pretrained(output / "adapter_final")\n','')
s=s.replace('outcome = {"status": "completed", "method": args.method, "global_step": max_steps,','outcome = {"status": "completed", "method": args.method, "arm": args.arm, "mode": args.mode, "global_step": max_steps,\n                   "unique_count": args.unique_count, "passes": args.passes, "refresh_event": refresh_event, "final_train_bank": final_bank,')
s=s.replace('"candidate_forwards": reference_forward_candidates + current_policy_forward_candidates + gradient_forward_candidates,','"candidate_forwards": reference_forward_candidates + current_policy_forward_candidates + gradient_forward_candidates + refresh_forward_candidates + final_bank_forward_candidates,\n                   "refreshed_reference_candidate_forwards": refresh_forward_candidates, "final_bank_candidate_forwards": final_bank_forward_candidates,')
s=s.replace('"final_adapter_identity": model_weight_identity(output / "adapter_final", adapter=True),','"final_adapter_identity": model_weight_identity(output / f"checkpoint-{max_steps}", adapter=True),')
Path('research/reference_refresh_20261107/refresh_train.py').write_text(s)
