        def continuity_payload(step):
            return {"schema_version": 1, "global_step": step,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "gradients": {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                                  for name, parameter in model.named_parameters() if parameter.requires_grad},
                    "model_modes": {name: module.training for name, module in model.named_modules()},
                    "rng": {"python_global": random.getstate(), "sampling_rng": rng.getstate(),
                            "numpy": np.random.get_state(), "cpu": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all()}}

        def score_current_unique(directory, step, role):
            require(not directory.exists(), "Never overwrite fresh-reference/final-bank output")
            directory.mkdir()
            checkpoint = output / f"checkpoint-{step}"
            audit_path = checkpoint / "finite_policy_checkpoint_audit.json"
            checkpoint_audit = json.loads(audit_path.read_text())
            require(checkpoint_audit["global_step"] == step, "Current checkpoint step differs")
            scored_identity = {"base": initial_identity["base"], "adapter": model_weight_identity(checkpoint, adapter=True)}
            require(scored_identity["adapter"] == checkpoint_audit["adapter_identity"], "Current checkpoint physical identity differs")
            before_adapter = adapter_fingerprint(model)
            before_vision = frozen_vision_fingerprint(model)
            before_payload = continuity_payload(step)
            before_summary = continuity_summary(before_payload)
            if role == "fresh_reference":
                torch.save(before_payload, directory / "continuity_before.pt")
            del before_payload
            cache = directory / ("reference_policy.jsonl" if role == "fresh_reference" else "scores.jsonl")
            current_spec = {"schema_version": 2, "role": role, "policy": "canonical24_full_sequence_plusEOS_temperature1",
                            "scored_identity": scored_identity, "initial_identity": initial_identity,
                            "checkpoint": str(checkpoint), "checkpoint_audit_sha256": sha256_file(audit_path),
                            "global_step": step, "arm": args.arm, "flow": _LOCKED_POLICY["flow"],
                            "input": asdict(config), "revision": args.revision,
                            "manifest_sha256": args.expected_manifest_sha256, "selected_records_sha256": hash_json(records),
                            "candidate_texts": CANDIDATES, "pixel_sha256": pixels,
                            "q0_reference_policy_sha256": reference_hash,
                            "q0_reference_lineage_sha256": signature_payload["reference_lineage_sha256"],
                            "scoring_source_sha256": {key: bound_sources[key] for key in SCORING_SOURCE_PATHS},
                            "versions": package_versions(), "run_signature_sha256": signature,
                            "numerical_reference_atol": args.logp_atol}
            write_json(str(cache) + ".config.json", current_spec)
            rows = []; forwards = 0; tick = time.monotonic(); stage_started = now()
            for index, record in enumerate(records):
                immutable()
                audit, candidates, encoded = prepare(record)
                require(audit["input_hash"] == reference[index]["input_audit"]["input_hash"]
                        and candidates == reference[index]["candidates"], "Fresh/final scoring input differs from actual q0")
                token_logps, checks = score_candidates(model, encoded, candidates, score_args, literal_check=index == 0)
                forwards += 24 + len(checks)
                current = log_policy([sum(values) for values in token_logps])
                require(current["logZ"] <= 1e-4, "Scored canonical mass exceeds one")
                exact = [score_order(candidate["text"], record["display_perm"], record["target_order"])["exact"] for candidate in candidates]
                require(exact == reference[index]["exact_rewards"] and sum(exact) == 1, "Fresh/final source gold differs")
                row = {"index": index, "qid": record["qid"], "recipe_id": record["recipe_id"],
                       "presentation_id": record["presentation_id"], "source_split": "train",
                       "display_perm": record["display_perm"], "target_order": record["target_order"], "target_labels": record["target_labels"],
                       "exact_rewards": exact, "gold_candidate_index": exact.index(1), "input_audit": audit,
                       "candidates": candidates, **current, "token_logps": token_logps, "numerical_checks": checks,
                       "reference_spec_sha256": hash_json(current_spec), "scored_identity_sha256": hash_json(scored_identity),
                       "q0_source_row_sha256": hash_json(reference[index]), "q0_reference_row_index": index,
                       "role": role, "global_step": step, "run_signature_sha256": signature}
                append(cache, row); rows.append(row); del encoded
            after_payload = continuity_payload(step)
            after_summary = continuity_summary(after_payload)
            if role == "fresh_reference":
                torch.save(after_payload, directory / "continuity_after.pt")
            del after_payload
            require(before_summary == after_summary, "Read-only scoring changed optimizer/gradient/mode/RNG/step state")
            after_adapter = adapter_fingerprint(model); after_vision = frozen_vision_fingerprint(model)
            require(before_adapter == after_adapter and before_vision == after_vision == initial_vision,
                    "Read-only scoring changed adapter/frozen vision")
            require(len(rows) == args.unique_count == args.final_bank_count, "Fresh/final unique bank incomplete")
            completion = {"status": "complete_current_checkpoint_reference" if role == "fresh_reference" else "complete_fixed_final_train_bank",
                          "passed": True, "role": role, "rows": len(rows), "cache_sha256": sha256_file(cache),
                          "reference_spec_sha256": hash_json(current_spec), "run_signature_sha256": signature,
                          "checkpoint_audit_sha256": sha256_file(audit_path), "scored_identity": scored_identity,
                          "global_step": step, "candidate_forwards": forwards, "optimizer_steps": 0, "backward_calls": 0, "generation_calls": 0,
                          "before_state": before_summary, "after_state": after_summary,
                          "adapter_fingerprint_before": before_adapter, "adapter_fingerprint_after": after_adapter,
                          "visual_fingerprint_before": before_vision, "visual_fingerprint_after": after_vision,
                          "started_utc": stage_started, "finished_utc": now(), "wall_seconds": time.monotonic() - tick}
            write_json(str(cache) + ".completion.json", completion)
            return cache, current_spec, rows, completion

        def build_boundary_reference(step):
            require(args.arm == "refresh_beta" and args.refresh_boundary_step == step and refresh_event is None,
                    "Only the one predeclared real refresh boundary")
            directory = output / f"refresh_at_step{step}"
            cache, current_spec, rows, completion = score_current_unique(directory, step, "fresh_reference")
            event = {"schema_version": 1, "passed": True, "reference_generation": 1, "boundary_step": step,
                     "rows": len(rows), "run_signature_sha256": signature, "child_policy_sha256": policy_sha,
                     "checkpoint": str(output / f"checkpoint-{step}"), "checkpoint_audit_sha256": completion["checkpoint_audit_sha256"],
                     "scored_identity": completion["scored_identity"],
                     "q0_reference_policy_sha256": reference_hash, "reference_policy_path": str(cache),
                     "reference_policy_sha256": completion["cache_sha256"], "reference_spec_sha256": hash_json(current_spec),
                     "config_sha256": sha256_file(str(cache) + ".config.json"),
                     "completion_sha256": sha256_file(str(cache) + ".completion.json"),
                     "continuity_files": {side: {"path": str(directory / ("continuity_" + side + ".pt")),
                                      "sha256": sha256_file(directory / ("continuity_" + side + ".pt")),
                                      "bytes": (directory / ("continuity_" + side + ".pt")).stat().st_size}
                                          for side in ("before", "after")},
                     "before_state": completion["before_state"], "after_state": completion["after_state"],
                     "candidate_forwards": completion["candidate_forwards"], "optimizer_steps": 0, "backward_calls": 0, "generation_calls": 0,
                     "started_utc": completion["started_utc"], "finished_utc": completion["finished_utc"], "wall_seconds": completion["wall_seconds"]}
            write_json(directory / "refresh_event.json", event)
            return event, rows

        def build_final_bank(step):
            cache, spec, rows, completion = score_current_unique(output / "final_train_bank", step, "final_train_bank")
            return {"passed": True, "path": str(cache), "rows": len(rows), "raw_sha256": sha256_file(cache),
                    "config_sha256": sha256_file(str(cache) + ".config.json"),
                    "completion_sha256": sha256_file(str(cache) + ".completion.json"),
                    "candidate_forwards": completion["candidate_forwards"], "global_step": step,
                    "run_signature_sha256": signature, "checkpoint_identity": completion["scored_identity"]["adapter"],
                    "optimizer_steps": 0, "backward_calls": 0, "generation_calls": 0}

