def logical_content(value):
    """Portable exact-state representation; framework imports only for actual arrays."""
    if value is None or type(value) in (bool, int, str):
        return {"kind": type(value).__name__, "value": value}
    if type(value) is float:
        require(math.isfinite(value), "Nonfinite continuity scalar")
        return {"kind": "float", "value": value}
    if isinstance(value, dict):
        items = [[logical_content(k), logical_content(v)] for k, v in value.items()]
        items.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, ensure_ascii=False))
        return {"kind": "dict", "items": items}
    if isinstance(value, (tuple, list)):
        return {"kind": type(value).__name__, "items": [logical_content(x) for x in value]}
    module = type(value).__module__
    if module.startswith("torch"):
        import torch
        require(isinstance(value, torch.Tensor), "Unsupported torch continuity object")
        tensor = value.detach().cpu().contiguous()
        require(bool(torch.isfinite(tensor).all().item()), "Nonfinite continuity tensor")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"kind": "torch_tensor", "dtype": str(tensor.dtype), "shape": list(tensor.shape),
                "sha256": hashlib.sha256(raw).hexdigest()}
    if module.startswith("numpy"):
        import numpy as np
        if isinstance(value, np.ndarray):
            require(not value.dtype.hasobject, "Object-array continuity refused")
            array = np.ascontiguousarray(value)
            return {"kind": "numpy_array", "dtype": str(array.dtype), "shape": list(array.shape),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        if isinstance(value, np.generic):
            return {"kind": "numpy_scalar", "dtype": str(value.dtype), "value": logical_content(value.item())}
    raise ValueError("Unsupported exact continuity type: " + str(type(value)))


def continuity_summary(payload):
    keys = {"schema_version", "global_step", "optimizer", "scheduler", "gradients", "model_modes", "rng"}
    require(set(payload) == keys and payload["schema_version"] == 1, "Incomplete continuity payload")
    components = {key: hash_json(logical_content(payload[key])) for key in sorted(keys)}
    steps = []
    for key in sorted(payload["optimizer"]["state"]):
        value = payload["optimizer"]["state"][key]["step"]
        step = float(value.item()) if hasattr(value, "item") else float(value)
        require(math.isfinite(step) and int(step) == step, "Invalid actual Adam step")
        steps.append(int(step))
    require(len(steps) == 392 and set(steps) == {payload["global_step"]}, "Adam moments/step continuity differs")
    require(payload["scheduler"]["last_epoch"] == payload["global_step"], "Scheduler step continuity differs")
    return {"logical_sha256": hash_json(components), "component_sha256": components,
            "optimizer_steps": steps, "scheduler_last_epoch": payload["scheduler"]["last_epoch"],
            "global_step": payload["global_step"]}


def validate_authorization(child, child_path):
    """Persisted CPU/q0/whole-stage gate must precede every model load."""
    path = (ROOT / child["authorization_path"]).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), "Missing confined pre-model authorization")
    auth = json.loads(path.read_text())
    mode, arm, flow = child["mode"], child["arm"], child["flow"]
    key = flow + "_" + arm
    if mode != "formal": key = mode + "_" + key
    require(auth["schema_version"] == 1 and auth["phase"] == mode and auth["child_key"] == key
            and auth["child_policy_sha256"] == sha256_file(child_path), "Authorization child/role differs")
    parent_path = (ROOT / auth["stage_policy_path"]).resolve()
    require(parent_path.is_relative_to(ROOT) and sha256_file(parent_path) == auth["stage_policy_sha256"], "Authorization parent policy differs")
    parent = json.loads(parent_path.read_text())
    require(parent["children"][key]["sha256"] == sha256_file(child_path)
            and (ROOT / parent["children"][key]["path"]).resolve() == child_path
            and parent["sources_sha256"] == child["sources_sha256"], "Parent child/source binding differs")
    require(os.environ.get("RECIPEQA_REFRESH_OUTER_WATCHDOG") == auth["stage_policy_sha256"], "Actual outerwatchdog must bind parent")
    for field in ("first_preflight_started_utc", "absolute_deadline_utc"):
        require(auth[field] == parent[field] == child[field], "Authorization changed original clock")
    stamp = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp(auth["first_preflight_started_utc"]) <= stamp(auth["issued_utc"]) <= datetime.now(timezone.utc)
            < stamp(auth["absolute_deadline_utc"]), "Authorization issued outside original clock")
    def receipt(item):
        file = (ROOT / item["path"]).resolve()
        require(file.is_relative_to(ROOT) and file.is_file() and sha256_file(file) == item["sha256"], "Authorization receipt SHA differs")
        data = json.loads(file.read_text()); require(data["passed"] is True, "Referenced technical gate failed")
        return data
    cpu = receipt(auth["cpu_receipt"])
    require(cpu["tests_run"] == parent["expected_cpu_tests"] and cpu["skipped"] == cpu["failures"] == cpu["errors"] == 0
            and cpu["cuda_initialized"] is False, "Nonvacuous zero-skip CPU gate required")
    require(set(auth["q0_receipts"]) == {"formal_flow44", "formal_flow45", "smoke_flow44", "smoke_flow45"}, "All four q0 audits required")
    for item in auth["q0_receipts"].values(): receipt(item)
    if mode == "formal":
        from refresh_contract import ARMS, PATH_SMOKES
        require(set(auth["technical_receipts"]) == {*("smoke_" + name for name in ARMS), *PATH_SMOKES}, "All ten technical runs required")
        for item in auth["technical_receipts"].values(): receipt(item)
        gate = receipt(auth["whole_gate_receipt"])
        require(gate["all_eight_formal_arms_required"] is True, "No reduced resource gate")
    return auth


