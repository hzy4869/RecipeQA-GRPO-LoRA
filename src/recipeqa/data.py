"""Recipe-grouped RecipeQA manifests with explicit, audit-only source metadata.

No network access is performed here. Source IDs, target fields, and image paths
are BACKEND METADATA: a prompt builder must use only step text, label A--D, and
opened image pixels. Display permutations map display position to semantic ID.
"""

from __future__ import annotations

import hashlib
import html
from html.parser import HTMLParser
from itertools import permutations
import json
from pathlib import Path
import random
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "recipeqa-ordering-v1"
LABELS = ("A", "B", "C", "D")
PERMUTATIONS = tuple(permutations(range(4)))
SPLITS = ("train", "val", "test")
DEFAULT_SIZES = {
    "train": 1024, "warmup": 256, "debug": 32, "val": 200,
    "monitor": 50, "bank": 100, "audit": 100, "orbit": 60, "process": 40,
}
IMAGE_RE = re.compile(r"(.+)_(\d+)_(\d+)\.[^.]+$")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)


class DataValidationError(ValueError):
    """Source metadata violates the declared task construction."""


class _PlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fragments: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag.lower() in {"script", "style"}:
            self.hidden_depth += 1
        elif tag.lower() in {"p", "br", "div", "li", "hr"}:
            self.fragments.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"}:
            self.hidden_depth = max(0, self.hidden_depth - 1)
        elif tag.lower() in {"p", "div", "li"}:
            self.fragments.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.fragments.append(data)


def clean_text(value: Any) -> str:
    """Remove HTML/URLs without using image source steps to select text."""
    if value is None:
        return ""
    parser = _PlainTextParser()
    parser.feed(str(value))
    parser.close()
    text = html.unescape("".join(parser.fragments))
    return " ".join(URL_RE.sub(" ", text).split())


def source_image_step(filename: str) -> int:
    match = IMAGE_RE.fullmatch(Path(filename).name)
    if not match:
        raise DataValidationError(f"Unrecognized image filename: {filename!r}")
    return int(match.group(2))


def canonicalize_row(row: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Validate all original choices and extract the four chronological images."""
    if row.get("task") != "visual_ordering":
        raise DataValidationError("Only visual_ordering rows are accepted")
    if row.get("split", split) != split:
        raise DataValidationError(f"Split mismatch in {row.get('qid')}")
    choices = row["choice_list"]
    answer = row["answer"]
    if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(choices):
        raise DataValidationError(f"Invalid answer in {row.get('qid')}")
    images = list(choices[answer])
    if len(choices) != 4 or len(images) != 4 or len(set(images)) != 4:
        raise DataValidationError(f"Expected four choices/four distinct images: {row.get('qid')}")
    if any(not isinstance(name, str) or Path(name).name != name or "\\" in name for name in images):
        raise DataValidationError(f"Image names must be basenames, not paths: {row.get('qid')}")
    if any(sorted(choice) != sorted(images) for choice in choices):
        raise DataValidationError(f"Choices contain different image sets: {row.get('qid')}")
    image_steps = [source_image_step(name) for name in images]
    if len(set(image_steps)) != 4 or image_steps != sorted(image_steps):
        raise DataValidationError(f"Correct choice is not four ascending source steps: {row.get('qid')}")
    steps = [
        {"source_step_id": int(step["id"]), "title": clean_text(step.get("title")),
         "body": clean_text(step.get("body"))}
        for step in row["context"]
    ]
    step_ids = [step["source_step_id"] for step in steps]
    if len(set(step_ids)) != len(step_ids) or step_ids != sorted(step_ids):
        raise DataValidationError(f"Context has duplicate/unordered step IDs: {row.get('qid')}")
    if not set(image_steps).issubset(step_ids):
        raise DataValidationError(f"Image source steps absent from full text: {row.get('qid')}")
    return {
        "schema_version": SCHEMA_VERSION,
        "qid": str(row["qid"]), "recipe_id": str(row["recipe_id"]),
        "source_split": split, "steps": steps,
        "images_chronological": images,
        "source_step_ids_chronological": image_steps,
        "target_order": list(range(4)),
    }


def load_source(path: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    rows = payload["data"] if isinstance(payload, dict) else payload
    records = [canonicalize_row(row, split) for row in rows if row.get("task") == "visual_ordering"]
    qids = [record["qid"] for record in records]
    if len(qids) != len(set(qids)):
        raise DataValidationError(f"Duplicate qid in {path}")
    return sorted(records, key=lambda r: (r["recipe_id"], r["qid"])), {
        "filename": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "all_questions": len(rows), "ordering_questions": len(records),
        "ordering_recipes": len({r["recipe_id"] for r in records}),
    }


def _rng(seed: int, namespace: str) -> random.Random:
    # Hashing the namespace explicitly avoids dependence on Python's salted hash.
    value = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return random.Random(int.from_bytes(value, "big"))


def _one_per_recipe(records: Sequence[dict[str, Any]], seed: int, namespace: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["recipe_id"]].append(record)
    recipe_ids = sorted(grouped)
    _rng(seed, f"{namespace}:recipe_order").shuffle(recipe_ids)
    result = []
    for recipe_id in recipe_ids:
        candidates = sorted(grouped[recipe_id], key=lambda row: row["qid"])
        result.append(_rng(seed, f"{namespace}:question:{recipe_id}").choice(candidates))
    return result


def _take(records: Sequence[dict[str, Any]], count: int, label: str) -> list[dict[str, Any]]:
    if count < 0 or count > len(records):
        raise DataValidationError(f"Cannot take {count} {label} from {len(records)} recipes")
    return list(records[:count])


def select_sets(
    sources: Mapping[str, Sequence[dict[str, Any]]], *, data_seed: int = 20261104,
    audit_seed: int = 20261106, sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Select by recipe, preserving official splits and nested diagnostics."""
    sizes = {**DEFAULT_SIZES, **(sizes or {})}
    recipe_sets = {split: {r["recipe_id"] for r in sources[split]} for split in SPLITS}
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            if recipe_sets[first] & recipe_sets[second]:
                raise DataValidationError(f"Official recipe overlap: {first}/{second}")
    pool = _one_per_recipe(sources["train"], data_seed, "train")
    selected: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for name in ("train", "warmup", "debug"):
        selected[name] = _take(pool[offset:], sizes[name], name)
        offset += sizes[name]
    val_pool = _one_per_recipe(sources["val"], data_seed, "val")
    selected["val"] = _take(val_pool, sizes["val"], "val")
    for name in ("monitor", "bank"):
        subset = list(selected["val"])
        _rng(data_seed, name).shuffle(subset)
        selected[name] = _take(subset, sizes[name], name)
    selected["test"] = sorted(sources["test"], key=lambda r: (r["recipe_id"], r["qid"]))
    audit_pool = _one_per_recipe(sources["test"], audit_seed, "audit")
    selected["audit"] = _take(audit_pool, sizes["audit"], "audit")
    selected["audit_reserve"] = list(audit_pool[sizes["audit"]:])
    for parent, child in (("audit", "orbit"), ("orbit", "process")):
        subset = list(selected[parent])
        _rng(audit_seed, child).shuffle(subset)
        selected[child] = _take(subset, sizes[child], child)
        selected[f"{child}_reserve"] = subset[sizes[child]:]
    return selected


def balanced_presentations(
    records: Sequence[Mapping[str, Any]], seed: int, namespace: str,
) -> dict[str, tuple[int, ...]]:
    """Pre-randomize IDs, then cycle all 24 display permutations."""
    qids = sorted(record["qid"] for record in records)
    _rng(seed, namespace).shuffle(qids)
    return {qid: PERMUTATIONS[i % 24] for i, qid in enumerate(qids)}


def index_image_files(image_root: Path) -> dict[str, list[Path]]:
    """Index basenames recursively; no assumptions about ZIP extraction layout."""
    result: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            result[path.name].append(path)
    return dict(result)


def _image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_images(
    filenames: Iterable[str], image_root: Path | None, *, verify: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    filenames = sorted(set(filenames))
    index = index_image_files(image_root) if image_root is not None else {}
    if image_root is not None and not image_root.is_dir():
        raise DataValidationError(f"Image root is not a directory: {image_root}")
    Image = None
    if image_root is not None and verify:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for image decoding; install it or explicitly use --skip-image-verification") from exc
    result: dict[str, dict[str, Any]] = {}
    for name in filenames:
        if image_root is None:
            result[name] = {"path": name, "status": "unresolved", "decoded": False}
            continue
        matches = index.get(Path(name).name, [])
        if not matches:
            result[name] = {"path": name, "status": "missing", "decoded": False}
        else:
            duplicate_info: dict[str, Any] = {}
            if len(matches) > 1:
                # The official ZIP contains some identical photos copied into
                # multiple split folders. Never guess a photo from its name or
                # prefer a split directory: only byte-identical copies can be
                # resolved automatically. JSON recipe splits remain unchanged.
                candidate_paths = [p.relative_to(image_root).as_posix() for p in matches]
                candidate_hashes = {_path: _image_sha256(path) for _path, path in zip(candidate_paths, matches)}
                if len(set(candidate_hashes.values())) != 1:
                    result[name] = {
                        "path": None, "status": "ambiguous_basename", "decoded": False,
                        "candidates": candidate_paths, "candidate_sha256": candidate_hashes,
                        "resolution": "unresolved_different_content",
                    }
                    continue
                duplicate_info = {
                    "candidates": candidate_paths, "candidate_sha256": candidate_hashes,
                    "content_sha256": next(iter(candidate_hashes.values())),
                    "resolution": "identical_sha256_copies_lexicographic_path",
                }
            path = matches[0]
            info: dict[str, Any] = {
                "path": path.relative_to(image_root).as_posix(),
                "status": "ok" if verify else "not_decoded", "decoded": False,
                "bytes": path.stat().st_size, **duplicate_info,
            }
            if verify:
                try:
                    with Image.open(path) as image:
                        info["source_mode"] = image.mode
                        rgb = image.convert("RGB")
                        rgb.load()
                        info["width"], info["height"] = rgb.size
                        info["decoded"] = True
                except Exception as exc:
                    info["status"] = "corrupt"
                    info["error"] = f"{type(exc).__name__}: {exc}"
            result[name] = info
    report = {
        "image_root": str(image_root.resolve()) if image_root is not None else None,
        "scope": "Selected question images, including diagnostic reserve candidates",
        "requested_unique_images": len(filenames),
        "indexed_image_files": sum(len(paths) for paths in index.values()),
        "duplicate_basenames_resolved_by_identical_sha256": sum(
            item.get("resolution") == "identical_sha256_copies_lexicographic_path" for item in result.values()
        ),
        "status_counts": dict(sorted(Counter(item["status"] for item in result.values()).items())),
        "issues": [{"filename": name, **info} for name, info in result.items() if info["status"] != "ok"],
        "images": result,
    }
    return result, report


def make_presentation(
    record: Mapping[str, Any], display_perm: Sequence[int], *, manifest_set: str,
    image_info: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a SELF-CONTAINED record; never use its metadata as prompt text."""
    display_perm = tuple(display_perm)
    if len(display_perm) != 4 or sorted(display_perm) != list(range(4)):
        raise DataValidationError(f"Invalid display permutation: {display_perm}")
    images = record["images_chronological"]
    if image_info is not None:
        info = image_info
    elif "image_paths_chronological" in record:
        info = {
            name: {"path": path, "status": status}
            for name, path, status in zip(
                images, record["image_paths_chronological"], record["image_status_chronological"]
            )
        }
    else:
        info = {name: {"path": name, "status": "unresolved"} for name in images}
    statuses = [info[name]["status"] for name in images]
    technical_status = "ok" if all(status == "ok" for status in statuses) else "unverified" if all(status in {"ok", "not_decoded", "unresolved"} for status in statuses) else "image_issue"
    result = dict(record)
    result.update({
        "manifest_set": manifest_set,
        "presentation_id": f"{record['qid']}::p{''.join(str(i) for i in display_perm)}",
        "display_perm": list(display_perm),
        "target_labels": [LABELS[display_perm.index(i)] for i in range(4)],
        "image_paths_chronological": [info[name]["path"] for name in images],
        "image_status_chronological": statuses,
        "image_paths_display": [info[images[i]]["path"] for i in display_perm],
        "image_status_display": [info[images[i]]["status"] for i in display_perm],
        "technical_status": technical_status,
    })
    return result


def make_training_visits(
    records: Sequence[Mapping[str, Any]], *, training_seed: int = 42, passes: int = 2,
    image_info: Mapping[str, Mapping[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Fixed per-seed stream: each pool question once per pass, with augmentation.

    Train all methods over this file ONCE. A trainer must not shuffle/resample it
    independently or interpret ``passes`` again as its own number of epochs.
    """
    if passes < 1:
        raise DataValidationError("Training passes must be positive")
    for pass_index in range(passes):
        ordered = sorted(records, key=lambda row: row["qid"])
        rng = _rng(training_seed, f"training_visits:pass:{pass_index}")
        rng.shuffle(ordered)
        perms = list(PERMUTATIONS)
        rng.shuffle(perms)
        for index, record in enumerate(ordered):
            visit = make_presentation(record, perms[index % 24], manifest_set="train_visits", image_info=image_info)
            visit.update({
                "visit_index": pass_index * len(records) + index,
                "pass_index": pass_index,
                "visit_id": f"seed{training_seed}:pass{pass_index}:index{index}",
                "training_seed": training_seed,
            })
            yield visit


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    sha = hashlib.sha256()
    count = 0
    with path.open("wb") as handle:
        for record in records:
            line = (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
            handle.write(line)
            sha.update(line)
            count += 1
    return {"records": count, "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def load_manifest_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream self-contained manifest rows; image paths remain relative."""
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prepare_manifests(
    data_dir: Path, output_dir: Path, *, image_root: Path | None = None,
    verify_images: bool = True, data_seed: int = 20261104,
    presentation_seed: int = 20261105, audit_seed: int = 20261106,
    training_seed: int = 42, training_passes: int = 2,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Create fixed study manifests and checksums without removing bad images."""
    sources, source_reports = {}, {}
    for split in SPLITS:
        sources[split], source_reports[split] = load_source(data_dir / f"{split}.json", split)
    selected = select_sets(sources, data_seed=data_seed, audit_seed=audit_seed, sizes=sizes)
    all_records = {record["qid"]: record for rows in selected.values() for record in rows}
    # Explicit cross-split qid validation, even if recipe IDs are disjoint.
    source_qids = [row["qid"] for rows in sources.values() for row in rows]
    if len(source_qids) != len(set(source_qids)):
        raise DataValidationError("Question IDs are not globally unique")
    image_info, image_report = inspect_images(
        (name for record in all_records.values() for name in record["images_chronological"]),
        image_root, verify=verify_images,
    )
    # Every diagnostic shares its question's primary test presentation. Training
    # can call make_presentation at runtime for in-split augmentation.
    maps = {}
    for split in SPLITS:
        split_rows = [record for record in all_records.values() if record["source_split"] == split]
        maps.update(balanced_presentations(split_rows, presentation_seed, split))
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "seeds": {"data": data_seed, "presentation": presentation_seed, "audit": audit_seed, "training": training_seed},
        "sizes": {**DEFAULT_SIZES, **(sizes or {})},
        "sources": source_reports,
        "selection_algorithm": "sorted recipe IDs then SHA256-namespaced Python Random shuffle; one random question per recipe; official splits never mixed",
        "task": "RecipeQA-derived text-guided four-image FREE ordering, not original four-option benchmark",
        "source_metadata_prompt_warning": "Only cleaned full recipe text, A-D display labels, and decoded pixels enter prompts; filenames/paths/qid/recipe_id/source_step_ids/targets never enter prompts",
        "technical_image_policy": "No rows automatically excluded/replaced. Technical issues remain in main test and are reported separately; replacements require an explicit logged policy.",
        "image_report": "image_report.json",
        "manifests": {},
    }
    for name, rows in selected.items():
        presentations = (
            make_presentation(record, perm, manifest_set=name, image_info=image_info)
            for record in rows
            for perm in (PERMUTATIONS if name == "orbit" else (maps[record["qid"]],))
        )
        file_report = write_jsonl(output_dir / f"{name}.jsonl", presentations)
        file_report.update({
            "questions": len(rows), "recipes": len({r["recipe_id"] for r in rows}),
            "source_splits": sorted({r["source_split"] for r in rows}),
            "technical_issue_questions": sum(any(image_info[n]["status"] not in {"ok", "not_decoded", "unresolved"} for n in r["images_chronological"]) for r in rows),
            "unverified_questions": sum(any(image_info[n]["status"] in {"not_decoded", "unresolved"} for n in r["images_chronological"]) for r in rows),
            "presentation_counts": dict(sorted(Counter("".join(map(str, maps[r["qid"]])) for r in rows).items())) if name != "orbit" else {"".join(map(str, perm)): len(rows) for perm in PERMUTATIONS},
        })
        report["manifests"][name] = file_report
    report["manifests"]["train_visits"] = {
        **write_jsonl(output_dir / "train_visits.jsonl", make_training_visits(
            selected["train"], training_seed=training_seed, passes=training_passes, image_info=image_info,
        )),
        "questions": len(selected["train"]), "recipes": len(selected["train"]),
        "source_splits": ["train"], "passes": training_passes,
        "trainer_policy": "Read file once in stored order; no additional epoch multiplier or reshuffling",
    }
    image_payload = json.dumps(image_report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    (output_dir / "image_report.json").write_text(image_payload)
    report["image_report_sha256"] = hashlib.sha256(image_payload.encode()).hexdigest()
    (output_dir / "manifest_report.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return report
