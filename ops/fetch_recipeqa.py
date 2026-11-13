#!/usr/bin/env python3
"""Resume, verify, and safely extract the official RecipeQA assets on DATA DISK.

Examples (stop any external curl writer before using the downloader):
  python ops/fetch_recipeqa.py --download-only
  python ops/fetch_recipeqa.py --extract-only
  python ops/fetch_recipeqa.py --download-only --files images.zip

Only /root/autodl-tmp and its descendants are accepted as --root. All partial
files, extraction files, locks, logs, and reports stay under that root. No
credentials are used or saved. Network transfer resumes raw/NAME.part; completed
assets are atomically renamed only after size/hash checks.
"""

from __future__ import annotations

import argparse
import binascii
from datetime import datetime, timezone
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import time
import urllib.error
import urllib.request
import zipfile


ALLOWED_ROOT = Path("/root/autodl-tmp")
DEFAULT_ROOT = ALLOWED_ROOT / "rl_recipeqa"
BASE_URL = "https://vision.cs.hacettepe.edu.tr/files/recipeqa/"
ASSETS = {
    # ZIP hash was not published/audited: compute and freeze the observed SHA256.
    "images.zip": {"bytes": 2271690314, "sha256": None},
    "train.json": {"bytes": 158656297, "sha256": "331f5fd153a01fad0c834976f2ff79665ffdffbc7cbe44a7a1c36c25ae1476f2"},
    "val.json": {"bytes": 19501603, "sha256": "cca6c2c7a3f5516251a8eb86e6c26548a56cebff871a8fba330b51bc788c4e04"},
    "test.json": {"bytes": 19052395, "sha256": "03092eca22d7a4015a4bfe9b9c133224d7386ddf663349e259337f000c1b5c9b"},
}


class IntegrityError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    allowed = ALLOWED_ROOT.resolve()
    if root != allowed and allowed not in root.parents:
        raise ValueError(f"--root must resolve within {ALLOWED_ROOT}; got {root}")
    return root


def safe_path(root: Path, *parts: str) -> Path:
    root = root.resolve()
    candidate = root.joinpath(*parts)
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise IntegrityError(f"Path escapes data root (including symlinks): {candidate}")
    return candidate


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class Progress:
    def __init__(self, root: Path, interval: float = 3.0):
        self.path = safe_path(root, "ops", "fetch_recipeqa_progress.json")
        self.events = safe_path(root, "ops", "fetch_recipeqa_events.jsonl")
        self.interval = interval
        self.last = 0.0
        self.state = {"started_at": now(), "pid": os.getpid(), "root": str(root)}

    def update(self, *, force: bool = False, **changes: object) -> None:
        self.state.update(changes)
        monotonic = time.monotonic()
        if force or monotonic - self.last >= self.interval:
            self.state["updated_at"] = now()
            atomic_json(self.path, self.state)
            line = json.dumps(self.state, sort_keys=True)
            with self.events.open("a") as handle:
                handle.write(line + "\n")
            print(line, flush=True)
            self.last = monotonic


def file_sha256(path: Path, progress: Progress | None = None) -> str:
    digest = hashlib.sha256()
    seen = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            seen += len(chunk)
            if progress:
                progress.update(phase="verify_sha256", asset=path.name, verified_bytes=seen, total_bytes=path.stat().st_size)
    return digest.hexdigest()


def verify_asset(path: Path, name: str, progress: Progress) -> dict:
    expected = ASSETS[name]
    actual_size = path.stat().st_size
    if actual_size != expected["bytes"]:
        raise IntegrityError(f"{name}: size {actual_size}, expected {expected['bytes']}")
    progress.update(force=True, phase="verify_sha256", asset=name, verified_bytes=0, total_bytes=actual_size)
    actual_sha = file_sha256(path, progress)
    if expected["sha256"] and actual_sha != expected["sha256"]:
        raise IntegrityError(f"{name}: SHA256 {actual_sha}, expected {expected['sha256']}")
    if name.endswith(".json"):
        # Hash matching already establishes the exact audited file. This check
        # also catches accidental HTML/error responses when expected hashes change.
        with path.open() as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise IntegrityError(f"{name}: expected official JSON object with a data list")
    return {
        "url": BASE_URL + name, "bytes": actual_size, "sha256": actual_sha,
        "expected_sha256": expected["sha256"],
        "sha256_provenance": "audited official metadata" if expected["sha256"] else "observed and recorded; no pre-audited ZIP digest",
        "verified_at": now(), "path": str(path),
    }


def download_asset(root: Path, name: str, progress: Progress, *, retries: int, timeout: float) -> dict:
    directory = safe_path(root, "data", "raw")
    directory.mkdir(parents=True, exist_ok=True)
    destination = safe_path(root, "data", "raw", name)
    partial = safe_path(root, "data", "raw", name + ".part")
    expected = ASSETS[name]["bytes"]
    if destination.exists():
        return verify_asset(destination, name, progress)
    if partial.exists() and partial.stat().st_size > expected:
        raise IntegrityError(f"Oversized partial {partial}: inspect before discarding it")
    if partial.exists() and partial.stat().st_size == expected:
        result = verify_asset(partial, name, progress)
        os.replace(partial, destination)
        result["path"] = str(destination)
        return result
    needed = expected - (partial.stat().st_size if partial.exists() else 0)
    if shutil.disk_usage(directory).free < needed + 128 * 1024 * 1024:
        raise IntegrityError(f"Insufficient DATA DISK space to download {name}: need {needed} additional bytes plus 128MiB reserve")
    start = time.monotonic()
    starting_size = partial.stat().st_size if partial.exists() else 0
    for attempt in range(retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected:
            break
        headers = {"User-Agent": "RecipeQA-reproducible-course-study/1.0", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(BASE_URL + name, headers=headers)
        progress.update(force=True, phase="download", asset=name, status="running", attempt=attempt + 1, downloaded_bytes=offset, total_bytes=expected)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status_code = response.getcode()
                if status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    prefix = f"bytes {offset}-"
                    if not content_range.startswith(prefix) or not content_range.endswith(f"/{expected}"):
                        raise IntegrityError(f"Unexpected Content-Range for {name}: {content_range!r}")
                    mode = "ab"
                elif status_code == 200:
                    length = response.headers.get("Content-Length")
                    if length and int(length) != expected:
                        raise IntegrityError(f"Official source size changed for {name}: Content-Length={length}, audited={expected}")
                    # A server may ignore Range. Restart this SAME .part safely;
                    # never append a full response to an existing partial.
                    mode, offset = "wb", 0
                else:
                    raise IntegrityError(f"Unexpected HTTP {status_code} for {name}")
                with partial.open(mode) as handle:
                    position = offset
                    while chunk := response.read(1024 * 1024):
                        if position + len(chunk) > expected:
                            raise IntegrityError(f"{name}: response exceeds audited size")
                        handle.write(chunk)
                        position += len(chunk)
                        elapsed = max(0.001, time.monotonic() - start)
                        progress.update(phase="download", asset=name, downloaded_bytes=position, total_bytes=expected,
                                        bytes_per_second=max(0, position - starting_size) / elapsed)
                    handle.flush()
                    os.fsync(handle.fileno())
                if partial.stat().st_size != expected:
                    raise OSError(f"Premature EOF: {partial.stat().st_size}/{expected} bytes")
            break
        except IntegrityError:
            raise
        except (OSError, urllib.error.URLError, TimeoutError, http.client.IncompleteRead) as exc:
            if attempt >= retries:
                raise RuntimeError(f"{name}: exhausted {retries + 1} attempts; partial retained at {partial}: {exc}") from exc
            delay = min(30, 2 ** min(attempt, 5))
            progress.update(force=True, phase="retry_wait", asset=name, status="retrying",
                            last_error=f"{type(exc).__name__}: {exc}", retry_in_seconds=delay)
            time.sleep(delay)
    result = verify_asset(partial, name, progress)
    os.replace(partial, destination)
    result["path"] = str(destination)
    progress.update(force=True, phase="download_complete", asset=name, status="running", downloaded_bytes=expected, total_bytes=expected)
    return result


def safe_zip_target(root: Path, member: zipfile.ZipInfo) -> Path:
    """Reject traversal, platform separators, links, and nonregular special files."""
    name = member.filename
    if not name or "\x00" in name or "\\" in name:
        raise IntegrityError(f"Unsafe ZIP member name: {name!r}")
    pure = PurePosixPath(name)
    if not pure.parts or pure.is_absolute() or ".." in pure.parts or (pure.parts and ":" in pure.parts[0]):
        raise IntegrityError(f"ZIP traversal/absolute member rejected: {name!r}")
    mode = member.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise IntegrityError(f"ZIP link/special file rejected: {name!r}")
    target = root.joinpath(*pure.parts)
    resolved = target.resolve()
    if resolved != root.resolve() and root.resolve() not in resolved.parents:
        raise IntegrityError(f"ZIP member escapes extraction directory: {name!r}")
    return target


def crc_matches(path: Path, member: zipfile.ZipInfo) -> bool:
    if not path.is_file() or path.stat().st_size != member.file_size:
        return False
    crc = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            crc = binascii.crc32(chunk, crc)
    return crc & 0xFFFFFFFF == member.CRC


def extract_images(root: Path, progress: Progress) -> dict:
    archive = safe_path(root, "data", "raw", "images.zip")
    if not archive.is_file():
        raise IntegrityError(f"Complete archive missing: {archive}; download first (a .part is not complete)")
    archive_report = verify_asset(archive, "images.zip", progress)
    destination = safe_path(root, "data", "images")
    destination.mkdir(parents=True, exist_ok=True)
    report_path = safe_path(root, "ops", "recipeqa_extraction_report.json")
    report = {"archive": archive_report, "image_root": str(destination), "started_at": now(),
              "status": "running", "files_extracted": 0, "files_crc_verified_existing": 0,
              "errors": [], "crc_checked_uncompressed_bytes": 0}
    atomic_json(report_path, report)
    try:
        with zipfile.ZipFile(archive) as zip_handle:
            members = zip_handle.infolist()
            # Validate ALL paths/types before writing ANY archive member.
            targets = [safe_zip_target(destination, item) for item in members]
            target_names = [str(path) for path in targets]
            if len(target_names) != len(set(target_names)):
                raise IntegrityError("ZIP contains duplicate normalized destination paths")
            regular = [(item, target) for item, target in zip(members, targets) if not item.is_dir()]
            total_bytes = sum(item.file_size for item, _ in regular)
            already_present = sum(target.stat().st_size for _, target in regular if target.is_file())
            need_bytes = max(0, total_bytes - already_present)
            if shutil.disk_usage(destination).free < need_bytes + 128 * 1024 * 1024:
                raise IntegrityError(f"Insufficient DATA DISK space to extract: need at least {need_bytes} additional bytes plus reserve")
            report.update({"archive_entries": len(members), "image_file_entries": len(regular), "total_uncompressed_bytes": total_bytes})
            for index, (member, target) in enumerate(regular):
                try:
                    if crc_matches(target, member):
                        report["files_crc_verified_existing"] += 1
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        partial = target.with_name(target.name + ".extracting.part")
                        # Existing filesystem symlinks must not redirect even the
                        # temporary file outside the extraction root.
                        if destination.resolve() not in partial.resolve().parents:
                            raise IntegrityError(f"Extraction temporary path escapes image root: {partial}")
                        with zip_handle.open(member) as source, partial.open("wb") as output:
                            shutil.copyfileobj(source, output, 1024 * 1024)
                            output.flush()
                            os.fsync(output.fileno())
                        # ZipExtFile checks the stored CRC when its stream ends.
                        if partial.stat().st_size != member.file_size:
                            raise IntegrityError(f"Extracted size mismatch: {member.filename}")
                        os.replace(partial, target)
                        report["files_extracted"] += 1
                    report["crc_checked_uncompressed_bytes"] += member.file_size
                except Exception as exc:
                    report["errors"].append({"member": member.filename, "error": f"{type(exc).__name__}: {exc}"})
                    atomic_json(report_path, report)
                progress.update(phase="extract_crc", status="running", asset="images.zip",
                                entries_done=index + 1, entries_total=len(regular),
                                extracted_files=report["files_extracted"], reused_crc_verified_files=report["files_crc_verified_existing"],
                                uncompressed_bytes_checked=report["crc_checked_uncompressed_bytes"],
                                total_uncompressed_bytes=total_bytes, crc_or_extraction_errors=len(report["errors"]))
            report["status"] = "failed" if report["errors"] else "completed"
            report["completed_at"] = now()
            atomic_json(report_path, report)
    except Exception as exc:
        report.update({"status": "failed", "completed_at": now(), "fatal_error": f"{type(exc).__name__}: {exc}"})
        atomic_json(report_path, report)
        raise
    if report["errors"]:
        raise IntegrityError(f"Extraction/CRC failed for {len(report['errors'])} members; see {report_path}")
    progress.update(force=True, phase="extract_complete", status="running", extracted_files=report["files_extracted"],
                    reused_crc_verified_files=report["files_crc_verified_existing"], crc_or_extraction_errors=0)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download-only", action="store_true")
    mode.add_argument("--extract-only", action="store_true")
    parser.add_argument("--files", nargs="+", choices=list(ASSETS), default=list(ASSETS), help="Download order defaults to images.zip first, then three JSON files")
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--progress-seconds", type=float, default=3)
    args = parser.parse_args()
    if args.retries < 0 or args.timeout <= 0 or args.progress_seconds <= 0:
        parser.error("retries must be nonnegative; timeout/progress-seconds must be positive")
    try:
        root = validate_root(args.root)
    except ValueError as exc:
        parser.error(str(exc))
    root.mkdir(parents=True, exist_ok=True)
    safe_path(root, "ops").mkdir(parents=True, exist_ok=True)
    lock_path = safe_path(root, "ops", "fetch_recipeqa.lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another fetch_recipeqa process holds the lock; do not run concurrent writers.", file=sys.stderr)
            return 2
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + "\n")
        lock.flush()
        progress = Progress(root, args.progress_seconds)
        progress.update(force=True, phase="start", status="running", mode="extract_only" if args.extract_only else "download_only" if args.download_only else "download_and_extract")
        inventory_path = safe_path(root, "ops", "recipeqa_assets.json")
        try:
            inventory = json.loads(inventory_path.read_text()) if inventory_path.exists() else {"assets": {}}
            if not args.extract_only:
                for name in args.files:
                    inventory["assets"][name] = download_asset(root, name, progress, retries=args.retries, timeout=args.timeout)
                    inventory["updated_at"] = now()
                    atomic_json(inventory_path, inventory)
            if not args.download_only:
                extracted = extract_images(root, progress)
                inventory["assets"]["images.zip"] = extracted["archive"]
                inventory["updated_at"] = now()
                atomic_json(inventory_path, inventory)
            progress.update(force=True, phase="done", status="completed", completed_at=now())
            return 0
        except KeyboardInterrupt:
            progress.update(force=True, phase="interrupted", status="interrupted", completed_at=now())
            return 130
        except Exception as exc:
            progress.update(force=True, phase="failed", status="failed", error=f"{type(exc).__name__}: {exc}", completed_at=now())
            print(f"RecipeQA fetch failed: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
