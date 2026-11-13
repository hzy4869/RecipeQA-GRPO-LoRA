#!/usr/bin/env python3
"""Download a hash-pinned ModelScope model without any implicit hub cache.

All writes (including partial files, progress, manifest, and completion marker)
are confined to --root under /root/autodl-tmp. No SDK cache or system /tmp is
used. Source metadata must be an existing ModelScope repo/files API response.

ModelScope API URL follows its official client's get_file_download_url:
https://github.com/modelscope/modelscope/blob/master/modelscope/hub/file_download.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
HF_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
REQUIRED_FILES = {"config.json", "generation_config.json", "preprocessor_config.json",
                  "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt",
                  "chat_template.json", "model.safetensors"}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def inside(root: Path, path: Path) -> Path:
    path = path.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Refusing output outside data root: {path}") from error
    return path


def atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".part")
    if temporary.is_symlink():
        raise ValueError(f"Refusing symlink temporary path: {temporary}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path):
    source = path.read_bytes()
    payload = json.loads(source)
    files = payload.get("Data", {}).get("Files") if isinstance(payload, dict) else None
    if files is None and isinstance(payload, dict):
        files = payload.get("Files", payload.get("files"))
    if not isinstance(files, list):
        raise ValueError("Manifest must contain ModelScope Data.Files or Files")
    entries = []
    seen = set()
    for file in files:
        if file.get("Type", "blob") != "blob":
            continue
        name = file.get("Path")
        if not isinstance(name, str):
            raise ValueError("Manifest file missing Path")
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or "\\" in name or not parts.parts:
            raise ValueError(f"Unsafe manifest file path: {name}")
        if name in seen:
            raise ValueError(f"Duplicate manifest path: {name}")
        seen.add(name)
        size = file.get("Size")
        digest = str(file.get("Sha256", "")).lower()
        if type(size) is not int or size < 0 or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
            raise ValueError(f"File lacks authoritative size/SHA256: {name}")
        entries.append({"path": name, "size": size, "sha256": digest,
                        "source_revision": file.get("Revision")})
    if REQUIRED_FILES - seen:
        raise ValueError(f"Model manifest missing required files: {sorted(REQUIRED_FILES - seen)}")
    return entries, hashlib.sha256(source).hexdigest()


def source_urls(name, revision, hf_revision):
    query = urllib.parse.urlencode({"Revision": revision, "FilePath": name})
    encoded = urllib.parse.quote(name, safe="/")
    return [f"https://www.modelscope.cn/api/v1/models/{MODEL_ID}/repo?{query}",
            f"https://hf-mirror.com/{MODEL_ID}/resolve/{hf_revision}/{encoded}"]


class Downloader:
    def __init__(self, root, model_dir, metadata_dir, entries, source_hash, args):
        self.root = root
        self.model_dir = model_dir
        self.metadata_dir = metadata_dir
        self.args = args
        self.entries = entries
        self.lock = threading.RLock()
        self.progress_path = inside(root, metadata_dir / "model_download_progress.json")
        self.complete_path = inside(root, model_dir / "DOWNLOAD_COMPLETE.json")
        self.last_write = 0.0
        self.state = {"schema_version": 1, "model_id": MODEL_ID, "started_at": utcnow(),
                      "status": "running", "manifest_sha256": source_hash,
                      "modelscope_revision_requested": args.revision,
                      "hf_revision_requested": args.hf_revision,
                      "provenance_note": "Files are pinned by the supplied ModelScope size/SHA256; an HF revision request alone does not establish cross-registry snapshot identity.",
                      "root": str(root), "model_dir": str(model_dir),
                      "total_expected_bytes": sum(entry["size"] for entry in entries),
                      "files": {entry["path"]: {**entry, "bytes": 0, "status": "queued"} for entry in entries}}
        if self.complete_path.exists():
            # A completion marker is recreated only after this entire manifest
            # passes verification. Never trust a stale marker from a prior run.
            self.complete_path.unlink()
        self.update(force=True)

    def update(self, name=None, force=False, **changes):
        with self.lock:
            if name is not None:
                self.state["files"][name].update(changes)
            if force or time.monotonic() - self.last_write >= self.args.progress_interval:
                self.state["updated_at"] = utcnow()
                self.state["downloaded_or_verified_bytes"] = sum(file["bytes"] for file in self.state["files"].values())
                atomic_json(self.progress_path, self.state)
                self.last_write = time.monotonic()

    def verified(self, path, entry):
        return path.is_file() and path.stat().st_size == entry["size"] and sha256_file(path) == entry["sha256"]

    def stream(self, url, part, entry):
        offset = part.stat().st_size if part.exists() else 0
        if offset > entry["size"]:
            raise ValueError(f"Partial file larger than expected: {part}")
        headers = {"User-Agent": "RecipeQA-Research-Downloader/1.0", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=self.args.timeout) as response:
            status = response.status
            if status == 206:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if not match or int(match.group(1)) != offset or int(match.group(3)) != entry["size"]:
                    raise ValueError(f"Incorrect resume Content-Range: {content_range}")
            elif status == 200:
                # Range ignored: overwrite from zero, never append a full body.
                offset = 0
            else:
                raise ValueError(f"Unexpected HTTP status {status}")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != entry["size"] - offset:
                raise ValueError(f"Content-Length mismatch for {entry['path']}: {length}")
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise ValueError("Refusing HTML response in place of a model file")
            self.update(entry["path"], bytes=offset, status="downloading", source_url=url, force=True)
            with part.open("ab" if offset else "wb") as stream:
                received = offset
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    if received + len(chunk) > entry["size"]:
                        raise ValueError("Response body exceeds authoritative file size")
                    stream.write(chunk)
                    received += len(chunk)
                    self.update(entry["path"], bytes=received)
                stream.flush()
                os.fsync(stream.fileno())
            if received != entry["size"]:
                raise IOError(f"Incomplete body for {entry['path']}: {received}/{entry['size']}")

    def fetch(self, entry):
        name = entry["path"]
        destination = inside(self.root, self.model_dir / name)
        part = inside(self.root, destination.with_name(destination.name + ".part"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.verified(destination, entry):
            self.update(name, status="verified_existing", bytes=entry["size"], force=True)
            print(json.dumps({"file": name, "status": "verified_existing"}), flush=True)
            return
        errors = []
        urls = source_urls(name, self.args.revision, self.args.hf_revision)
        for attempt in range(self.args.retries):
            for url in urls:
                try:
                    if not (part.exists() and part.stat().st_size == entry["size"]):
                        self.stream(url, part, entry)
                    self.update(name, status="verifying", force=True)
                    observed_size = part.stat().st_size
                    observed_hash = sha256_file(part)
                    if observed_size != entry["size"] or observed_hash != entry["sha256"]:
                        self.update(name, status="integrity_failure", observed_size=observed_size,
                                    observed_sha256=observed_hash, force=True)
                        part.unlink()
                        raise ValueError(f"SHA256/size mismatch: {name}")
                    os.replace(part, destination)
                    self.update(name, status="complete", bytes=entry["size"], source_url=url,
                                verified_sha256=observed_hash, completed_at=utcnow(), force=True)
                    print(json.dumps({"file": name, "status": "complete", "bytes": entry["size"], "sha256": observed_hash}), flush=True)
                    return
                except (OSError, ValueError, urllib.error.URLError) as error:
                    errors.append({"attempt": attempt + 1, "url": url, "error": str(error)})
                    self.update(name, status="retrying", last_error=str(error), attempts=errors, force=True)
            if attempt + 1 < self.args.retries:
                time.sleep(min(2 ** attempt, 8))
        self.update(name, status="failed", attempts=errors, force=True)
        raise RuntimeError(f"Could not download {name}; see {self.progress_path}")

    def run(self):
        small = [entry for entry in self.entries if entry["size"] < self.args.large_threshold]
        large = [entry for entry in self.entries if entry["size"] >= self.args.large_threshold]
        # One large file at a time; small configurations/tokenizers run in a
        # bounded pool. All content gets SHA256 verification before promotion.
        try:
            with ThreadPoolExecutor(max_workers=self.args.small_workers) as pool:
                futures = [pool.submit(self.fetch, entry) for entry in small]
                for future in as_completed(futures):
                    future.result()
            for entry in large:
                self.fetch(entry)
            self.state["status"] = "complete"
            self.state["completed_at"] = utcnow()
            self.update(force=True)
            atomic_json(self.complete_path, self.state)
            print(json.dumps({"status": "complete", "marker": str(self.complete_path)}), flush=True)
        except BaseException as error:
            self.state["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            self.state["last_error"] = str(error)
            self.update(force=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/rl_recipeqa"))
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--hf-revision", default=HF_REVISION)
    parser.add_argument("--small-workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--progress-interval", type=float, default=2)
    parser.add_argument("--large-threshold", type=int, default=100 * 1024 * 1024)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        root.relative_to(Path("/root/autodl-tmp").resolve())
    except ValueError:
        parser.error("--root must be beneath /root/autodl-tmp; system /tmp and system disk outputs are forbidden")
    if not root.is_dir():
        parser.error("--root must already exist on the verified data disk")
    if args.small_workers < 1 or args.retries < 1 or args.timeout <= 0 or args.progress_interval <= 0:
        parser.error("workers/retries/timeout/progress-interval must be positive")
    metadata = inside(root, args.metadata_dir or root / "metadata")
    model_dir = inside(root, args.model_dir or root / "models/Qwen3-VL-2B-Instruct")
    manifest = inside(root, args.manifest or metadata / "modelscope_files.json")
    metadata.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    entries, source_hash = load_manifest(manifest)
    lock_path = inside(root, metadata / "model_download.lock")
    # Unix flock is released even after process interruption, avoiding two
    # writers resuming the same .part file concurrently.
    import fcntl
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another model download holds model_download.lock")
        atomic_json(inside(root, metadata / "model_download_manifest.json"),
                    {"model_id": MODEL_ID, "source_manifest": str(manifest),
                     "source_manifest_sha256": source_hash, "files": entries})
        Downloader(root, model_dir, metadata, entries, source_hash, args).run()


if __name__ == "__main__":
    main()
