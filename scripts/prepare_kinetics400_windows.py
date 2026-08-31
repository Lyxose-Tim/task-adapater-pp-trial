#!/usr/bin/env python3
"""Windows-native, resumable downloader for the official CVDF Kinetics-400.

The upstream project provides Bash scripts built around ``wget -c -i``.  This
tool keeps the same official S3 objects but uses Windows curl or aria2,
explicit remote sizes, atomic completion, and a durable JSON manifest.
Downloaded archives are never deleted by this program.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


BASE = "https://s3.amazonaws.com/kinetics/400"
PATH_LISTS = {
    "train": f"{BASE}/train/k400_train_path.txt",
    "val": f"{BASE}/val/k400_val_path.txt",
    "test": f"{BASE}/test/k400_test_path.txt",
}
EXTRAS = {
    "annotations/train.csv": f"{BASE}/annotations/train.csv",
    "annotations/val.csv": f"{BASE}/annotations/val.csv",
    "annotations/test.csv": f"{BASE}/annotations/test.csv",
    "replacements/replacement_for_corrupted_k400.tgz": (
        f"{BASE}/replacement_for_corrupted_k400.tgz"
    ),
    "metadata/official_readme.md": f"{BASE}/readme.md",
}


@dataclass(frozen=True)
class RemoteItem:
    group: str
    url: str
    relative_path: str
    expected_bytes: int
    etag: str
    last_modified: str

    @property
    def filename(self) -> str:
        return PurePosixPath(self.relative_path).name


def _request(url: str, method: str = "GET", timeout: int = 90):
    request = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": "task-adapter-pp-k400-windows/1.0"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def _download_text(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _request(url) as response:
        data = response.read()
    if not data:
        raise RuntimeError(f"empty response from {url}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(destination)


def _head(url: str, retries: int = 6) -> Mapping[str, object]:
    error: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            with _request(url, method="HEAD") as response:
                length = response.headers.get("Content-Length")
                if length is None:
                    raise RuntimeError(f"HEAD response has no Content-Length: {url}")
                return {
                    "expected_bytes": int(length),
                    "etag": response.headers.get("ETag", "").strip('"'),
                    "last_modified": response.headers.get("Last-Modified", ""),
                }
        except BaseException as exc:  # retain the final network error verbatim
            error = exc
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"HEAD failed after {retries} attempts: {url}") from error


def _manifest_path(root: Path, split: str) -> Path:
    return root / "manifests" / f"k400_{split}_path.txt"


def _read_urls(path: Path) -> List[str]:
    urls = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    urls = [url for url in urls if url]
    if not urls or len(urls) != len(set(urls)):
        raise ValueError(f"manifest is empty or contains duplicate URLs: {path}")
    return urls


def _item_specifications(root: Path) -> List[tuple[str, str, str]]:
    specs: List[tuple[str, str, str]] = []
    for split, list_url in PATH_LISTS.items():
        path = _manifest_path(root, split)
        _download_text(list_url, path)
        for url in _read_urls(path):
            filename = PurePosixPath(urllib.parse.urlparse(url).path).name
            if not filename.endswith(".tar.gz"):
                raise ValueError(f"unexpected K400 archive name: {url}")
            specs.append((split, url, f"{split}/{filename}"))
    for relative, url in EXTRAS.items():
        group = relative.split("/", 1)[0]
        specs.append((group, url, relative))
    return specs


def build_remote_manifest(root: Path, workers: int = 16) -> Path:
    specs = _item_specifications(root)
    root.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_head, url): (group, url, relative) for group, url, relative in specs}
        items: List[RemoteItem] = []
        done = 0
        for future in concurrent.futures.as_completed(futures):
            group, url, relative = futures[future]
            metadata = future.result()
            items.append(
                RemoteItem(
                    group=group,
                    url=url,
                    relative_path=relative,
                    expected_bytes=int(metadata["expected_bytes"]),
                    etag=str(metadata["etag"]),
                    last_modified=str(metadata["last_modified"]),
                )
            )
            done += 1
            if done % 25 == 0 or done == len(specs):
                print(f"HEAD {done}/{len(specs)}", flush=True)

    items.sort(key=lambda item: (item.group, item.relative_path))
    payload = {
        "schema_version": 1,
        "source": "https://github.com/cvdfoundation/kinetics-dataset",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "objects": [asdict(item) for item in items],
        "object_count": len(items),
        "total_bytes": sum(item.expected_bytes for item in items),
        "groups": {
            group: {
                "objects": sum(item.group == group for item in items),
                "bytes": sum(item.expected_bytes for item in items if item.group == group),
            }
            for group in sorted({item.group for item in items})
        },
    }
    destination = root / "manifests" / "remote_objects.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({key: payload[key] for key in ("object_count", "total_bytes", "groups")}, indent=2))
    return destination


def load_remote_manifest(root: Path) -> List[RemoteItem]:
    path = root / "manifests" / "remote_objects.json"
    if not path.is_file():
        raise FileNotFoundError(f"run metadata first: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [RemoteItem(**entry) for entry in payload["objects"]]


def _find_curl(explicit: Optional[str] = None) -> str:
    candidates = [explicit, shutil.which("curl.exe"), shutil.which("curl")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError("curl.exe was not found; install curl or pass --curl")


def _find_aria2(explicit: Optional[str] = None) -> str:
    candidates = [explicit, shutil.which("aria2c.exe"), shutil.which("aria2c")]
    bundled = Path("tools/aria2/aria2-1.37.0-win-64bit-build1/aria2c.exe").resolve()
    candidates.append(str(bundled))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise FileNotFoundError("aria2c.exe was not found; install aria2 or pass --aria2")


def _download_one(root: Path, item: RemoteItem, curl: str) -> Mapping[str, object]:
    destination = root / "raw" / Path(item.relative_path)
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and destination.stat().st_size == item.expected_bytes:
        return {"path": item.relative_path, "status": "already_complete", "bytes": item.expected_bytes}
    if destination.exists():
        if destination.stat().st_size > item.expected_bytes:
            raise RuntimeError(f"existing file is larger than official object: {destination}")
        if partial.exists():
            raise RuntimeError(f"both incomplete destination and .part exist: {destination}")
        destination.replace(partial)
    if partial.exists() and partial.stat().st_size > item.expected_bytes:
        raise RuntimeError(f"partial file is larger than official object: {partial}")

    command = [
        curl,
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--continue-at",
        "-",
        "--retry",
        "20",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "1024",
        "--speed-time",
        "120",
        "--output",
        str(partial),
        item.url,
    ]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"curl failed ({completed.returncode}) for {item.url}: {completed.stderr.strip()}"
        )
    actual = partial.stat().st_size if partial.exists() else -1
    if actual != item.expected_bytes:
        raise RuntimeError(
            f"size mismatch after curl for {item.relative_path}: {actual} != {item.expected_bytes}"
        )
    partial.replace(destination)
    return {"path": item.relative_path, "status": "downloaded", "bytes": actual}


def _download_one_aria2(
    root: Path,
    item: RemoteItem,
    aria2: str,
    connections: int,
) -> Mapping[str, object]:
    destination = root / "raw" / Path(item.relative_path)
    partial = destination.with_suffix(destination.suffix + ".part")
    control = Path(str(partial) + ".aria2")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if (
        destination.is_file()
        and destination.stat().st_size == item.expected_bytes
        and not Path(str(destination) + ".aria2").exists()
    ):
        return {"path": item.relative_path, "status": "already_complete", "bytes": item.expected_bytes}
    if destination.exists():
        if destination.stat().st_size > item.expected_bytes:
            raise RuntimeError(f"existing file is larger than official object: {destination}")
        if partial.exists():
            raise RuntimeError(f"both incomplete destination and .part exist: {destination}")
        destination.replace(partial)

    command = [
        aria2,
        "--continue=true",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        "--min-split-size=4M",
        "--file-allocation=none",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "--max-tries=0",
        "--retry-wait=5",
        "--connect-timeout=30",
        "--timeout=120",
        "--console-log-level=warn",
        "--summary-interval=0",
        "--download-result=hide",
        f"--dir={destination.parent}",
        f"--out={partial.name}",
        item.url,
    ]
    completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"aria2 failed ({completed.returncode}) for {item.url}: {completed.stderr.strip()}"
        )
    actual = partial.stat().st_size if partial.exists() else -1
    if control.exists():
        raise RuntimeError(f"aria2 exited successfully but left an incomplete control file: {control}")
    if actual != item.expected_bytes:
        raise RuntimeError(
            f"size mismatch after aria2 for {item.relative_path}: {actual} != {item.expected_bytes}"
        )
    partial.replace(destination)
    return {"path": item.relative_path, "status": "downloaded", "bytes": actual}


def _selected(items: Sequence[RemoteItem], groups: Sequence[str]) -> List[RemoteItem]:
    requested = set(groups)
    if "all" in requested:
        return list(items)
    available = {item.group for item in items}
    unknown = requested - available
    if unknown:
        raise ValueError(f"unknown group(s) {sorted(unknown)}; choose from {sorted(available)}")
    return [item for item in items if item.group in requested]


def download(
    root: Path,
    groups: Sequence[str],
    workers: int,
    curl_path: Optional[str],
    *,
    engine: str = "curl",
    aria2_path: Optional[str] = None,
    connections_per_file: int = 8,
) -> None:
    items = _selected(load_remote_manifest(root), groups)
    if engine == "curl":
        executable = _find_curl(curl_path)
    elif engine == "aria2":
        executable = _find_aria2(aria2_path)
    else:
        raise ValueError("engine must be 'curl' or 'aria2'")
    log_path = root / "logs" / "download_events.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failures: List[Mapping[str, str]] = []
    completed_count = 0
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            if engine == "curl":
                futures = {
                    executor.submit(_download_one, root, item, executable): item
                    for item in items
                }
            else:
                futures = {
                    executor.submit(
                        _download_one_aria2,
                        root,
                        item,
                        executable,
                        connections_per_file,
                    ): item
                    for item in items
                }
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    event = dict(future.result())
                    event["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    completed_count += 1
                    print(
                        f"{completed_count}/{len(items)} {event['status']} {item.relative_path}",
                        flush=True,
                    )
                except BaseException as exc:
                    event = {
                        "path": item.relative_path,
                        "status": "failed",
                        "error": str(exc),
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    failures.append({"path": item.relative_path, "error": str(exc)})
                    print(f"FAILED {item.relative_path}: {exc}", file=sys.stderr, flush=True)
                log_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} downloads failed; rerun the same command to resume")


def status(root: Path, groups: Sequence[str]) -> Mapping[str, object]:
    items = _selected(load_remote_manifest(root), groups)
    complete = partial = missing = bytes_present = aria2_in_progress = 0
    errors: List[str] = []
    for item in items:
        destination = root / "raw" / Path(item.relative_path)
        part = destination.with_suffix(destination.suffix + ".part")
        aria2_control = Path(str(part) + ".aria2")
        if destination.is_file() and destination.stat().st_size == item.expected_bytes:
            complete += 1
            bytes_present += item.expected_bytes
        elif part.is_file():
            partial += 1
            if aria2_control.exists():
                # aria2 writes non-contiguous ranges.  The logical file length
                # may already equal Content-Length while most pieces are still
                # absent, so never report it as downloaded bytes.
                aria2_in_progress += 1
            else:
                bytes_present += part.stat().st_size
            if destination.exists():
                errors.append(f"both final and partial exist: {item.relative_path}")
        else:
            missing += 1
            if destination.exists():
                bytes_present += destination.stat().st_size
                errors.append(f"wrong-sized final file: {item.relative_path}")
    result = {
        "objects": len(items),
        "complete": complete,
        "partial": partial,
        "aria2_in_progress": aria2_in_progress,
        "missing": missing,
        "bytes_present": bytes_present,
        "expected_bytes": sum(item.expected_bytes for item in items),
        "errors": errors,
        "valid": complete == len(items) and not errors,
    }
    print(json.dumps(result, indent=2))
    return result


def _safe_member(destination: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive member: {name}")
    target = (destination / Path(*relative.parts)).resolve()
    target.relative_to(destination.resolve())
    return target


def verify_archives(root: Path, groups: Sequence[str], sha256: bool = False) -> None:
    items = [
        item
        for item in _selected(load_remote_manifest(root), groups)
        if item.relative_path.endswith((".tar.gz", ".tgz"))
    ]
    output = root / "manifests" / "verified_archives.jsonl"
    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for index, item in enumerate(items, 1):
            path = root / "raw" / Path(item.relative_path)
            if not path.is_file() or path.stat().st_size != item.expected_bytes:
                raise FileNotFoundError(f"archive is missing or incomplete: {path}")
            digest = hashlib.sha256() if sha256 else None
            members = 0
            with path.open("rb") as raw:
                if digest is not None:
                    for block in iter(lambda: raw.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                    raw.seek(0)
                with tarfile.open(fileobj=raw, mode="r:*") as archive:
                    for member in archive:
                        _safe_member(Path("."), member.name)
                        if member.issym() or member.islnk():
                            raise ValueError(f"links are not allowed in archive: {path}::{member.name}")
                        members += 1
            event = {
                "path": item.relative_path,
                "bytes": item.expected_bytes,
                "members": members,
                "sha256": digest.hexdigest() if digest is not None else None,
                "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            handle.write(json.dumps(event) + "\n")
            print(f"verified {index}/{len(items)} {item.relative_path} ({members} members)", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/Kinetics400"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata = subparsers.add_parser("metadata", help="fetch official lists and HEAD every object")
    metadata.add_argument("--workers", type=int, default=16)

    downloader = subparsers.add_parser("download", help="download or resume official objects")
    downloader.add_argument("--groups", nargs="+", default=["all"])
    downloader.add_argument("--workers", type=int, default=8)
    downloader.add_argument("--curl", default=None)
    downloader.add_argument("--engine", choices=("curl", "aria2"), default="curl")
    downloader.add_argument("--aria2", default=None)
    downloader.add_argument("--connections-per-file", type=int, default=8)

    status_parser = subparsers.add_parser("status", help="report byte-exact download progress")
    status_parser.add_argument("--groups", nargs="+", default=["all"])

    verifier = subparsers.add_parser("verify", help="stream-test completed tar archives")
    verifier.add_argument("--groups", nargs="+", default=["train", "val", "test", "replacements"])
    verifier.add_argument("--sha256", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.expanduser().resolve()
    if args.command == "metadata":
        build_remote_manifest(root, workers=args.workers)
    elif args.command == "download":
        download(
            root,
            args.groups,
            args.workers,
            args.curl,
            engine=args.engine,
            aria2_path=args.aria2,
            connections_per_file=args.connections_per_file,
        )
    elif args.command == "status":
        status(root, args.groups)
    elif args.command == "verify":
        verify_archives(root, args.groups, sha256=args.sha256)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
