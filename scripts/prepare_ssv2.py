#!/usr/bin/env python3
"""Prepare the official Something-Something V2 archive for FSAR experiments.

The official video payload is a gzip-compressed tar archive split into numbered
parts.  ``extract`` streams those parts as one logical file, so it never needs
to create a second 19.4 GB joined archive.  ``build-small`` then recreates the
class-disjoint CMN split directly from the official labels ZIP and writes the
``path num_frames label`` annotations consumed by :mod:`fsar.data`.

The source archive parts and labels ZIP are read-only inputs.  Extraction uses
an ``.incomplete`` staging directory, records source hashes before and after
the operation, and only publishes the final directory after every check has
passed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fsar.data import (  # noqa: E402
    AnnotationRecord,
    probe_video_num_frames,
    validate_split_annotations,
    write_annotations,
)


DEFAULT_SOURCE = ROOT / "dataset" / "SSv2"
DEFAULT_SPLIT_SPEC = ROOT / "data" / "fsar_splits" / "somethingcmn.json"
DEFAULT_OUTPUT_DIR = ROOT / "dataset" / "smsm_cmn" / "annotations"
DEFAULT_LABELS_ARCHIVE = "20bn-something-something-download-package-labels.zip"
DEFAULT_PART_PREFIX = "20bn-something-something-v2"
DEFAULT_EXPECTED_VIDEOS = 220_847
SPLIT_ORDER = ("train", "val", "test")
VIDEO_SUFFIXES = (".webm", ".mp4")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 of *path* without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprints(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    return {
        path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    }


def discover_archive_parts(source: Path, prefix: str = DEFAULT_PART_PREFIX) -> List[Path]:
    """Find and validate consecutively numbered archive parts."""

    expression = re.compile(re.escape(prefix) + r"-(\d+)$")
    numbered: List[Tuple[int, Path]] = []
    for path in source.glob(prefix + "-*"):
        match = expression.fullmatch(path.name)
        if match and path.is_file():
            numbered.append((int(match.group(1)), path))
    numbered.sort(key=lambda item: item[0])
    if not numbered:
        raise FileNotFoundError(
            "no archive parts matching {}-* under {}".format(prefix, source)
        )
    numbers = [number for number, _ in numbered]
    expected = list(range(len(numbered)))
    if numbers != expected:
        raise ValueError(
            "archive parts must be consecutive from 00; found {}".format(numbers)
        )
    return [path for _, path in numbered]


class MultiPartReader(io.RawIOBase):
    """Read numbered files as one non-seekable binary stream."""

    def __init__(self, parts: Sequence[Path]) -> None:
        super().__init__()
        if not parts:
            raise ValueError("at least one archive part is required")
        self._parts = [Path(path) for path in parts]
        self._part_index = 0
        self._handle: Optional[BinaryIO] = None

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def _open_next(self) -> bool:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._part_index >= len(self._parts):
            return False
        self._handle = self._parts[self._part_index].open("rb")
        self._part_index += 1
        return True

    def readinto(self, buffer: bytearray) -> int:
        view = memoryview(buffer)
        total = 0
        while total < len(view):
            if self._handle is None and not self._open_next():
                break
            assert self._handle is not None
            count = self._handle.readinto(view[total:])
            if count:
                total += int(count)
                continue
            self._handle.close()
            self._handle = None
        return total

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def _safe_member_parts(name: str) -> Tuple[str, ...]:
    """Validate a tar member name for both POSIX and Windows extraction."""

    normalised = name.replace("\\", "/")
    path = PurePosixPath(normalised)
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if path.is_absolute() or not parts:
        raise ValueError("unsafe archive member path: {!r}".format(name))
    if any(part == ".." for part in parts):
        raise ValueError("unsafe archive member path: {!r}".format(name))
    if re.match(r"^[A-Za-z]:", parts[0]):
        raise ValueError("unsafe archive member path: {!r}".format(name))
    return parts


def _extract_video_tar(parts: Sequence[Path], staging: Path) -> Tuple[int, int]:
    videos = staging / "videos"
    videos.mkdir(parents=True, exist_ok=False)
    video_count = 0
    total_bytes = 0

    with MultiPartReader(parts) as raw:
        with io.BufferedReader(raw, buffer_size=1024 * 1024) as stream:
            with tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    member_parts = _safe_member_parts(member.name)
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise ValueError(
                            "unsupported non-regular archive member: {!r}".format(
                                member.name
                            )
                        )
                    if not member_parts[-1].lower().endswith(".webm"):
                        continue
                    destination = videos / member_parts[-1]
                    if destination.exists():
                        raise ValueError(
                            "duplicate video filename in archive: {}".format(
                                destination.name
                            )
                        )
                    source_handle = archive.extractfile(member)
                    if source_handle is None:
                        raise OSError("could not read archive member {}".format(member.name))
                    temporary = destination.with_name(destination.name + ".part")
                    try:
                        with source_handle, temporary.open("xb") as output:
                            shutil.copyfileobj(source_handle, output, length=1024 * 1024)
                        if temporary.stat().st_size != member.size:
                            raise OSError(
                                "size mismatch for {}: expected {}, wrote {}".format(
                                    member.name, member.size, temporary.stat().st_size
                                )
                            )
                        os.replace(str(temporary), str(destination))
                    finally:
                        if temporary.exists():
                            temporary.unlink()
                    video_count += 1
                    total_bytes += int(member.size)
    return video_count, total_bytes


def extract_archive(
    source: Path,
    *,
    destination: Optional[Path] = None,
    prefix: str = DEFAULT_PART_PREFIX,
    expected_video_count: Optional[int] = DEFAULT_EXPECTED_VIDEOS,
) -> Mapping[str, Any]:
    """Safely stream the split TGZ into ``destination/videos``."""

    source = Path(source).resolve()
    destination = (
        Path(destination).resolve()
        if destination is not None
        else source / "extracted"
    )
    staging = destination.with_name(destination.name + ".incomplete")
    marker = destination / "extraction.complete.json"
    parts = discover_archive_parts(source, prefix)

    if destination.exists():
        if not marker.is_file():
            raise FileExistsError(
                "destination exists without a completion marker: {}".format(destination)
            )
        payload = json.loads(marker.read_text(encoding="utf-8"))
        current = _source_fingerprints(parts)
        if current != payload.get("source_hashes_after"):
            raise ValueError("archive source files changed since extraction completed")
        return payload
    if staging.exists():
        raise FileExistsError(
            "incomplete extraction exists at {}; inspect failure.json and move or "
            "remove it before retrying".format(staging)
        )

    staging.mkdir(parents=True)
    started = _utc_now()
    try:
        hashes_before = _source_fingerprints(parts)
        _write_json_atomic(
            staging / "source_hashes.before.json",
            {"created_at": _utc_now(), "files": hashes_before},
        )
        video_count, total_bytes = _extract_video_tar(parts, staging)
        if expected_video_count is not None and video_count != expected_video_count:
            raise ValueError(
                "expected {} WebM files, extracted {}".format(
                    expected_video_count, video_count
                )
            )
        hashes_after = _source_fingerprints(parts)
        if hashes_after != hashes_before:
            raise ValueError("archive source hashes changed during extraction")
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "started_at": started,
            "completed_at": _utc_now(),
            "source": str(source),
            "parts": [path.name for path in parts],
            "source_hashes_before": hashes_before,
            "source_hashes_after": hashes_after,
            "videos_directory": "videos",
            "video_count": video_count,
            "uncompressed_video_bytes": total_bytes,
        }
        _write_json_atomic(staging / "extraction.complete.json", payload)
        os.replace(str(staging), str(destination))
        return payload
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "started_at": started,
            "failed_at": _utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            _write_json_atomic(staging / "failure.json", failure)
        except OSError:
            pass
        raise


def _normalise_template(value: str) -> str:
    # Official templates bracket placeholders including ``[something]``,
    # ``[somewhere]`` and ``[some substance]``; the CMN lists omit brackets.
    return " ".join(str(value).translate(str.maketrans("", "", "[]")).split())


def _read_zip_json(archive: zipfile.ZipFile, basename: str) -> Tuple[Any, bytes]:
    matches = [name for name in archive.namelist() if PurePosixPath(name).name == basename]
    if len(matches) != 1:
        raise ValueError(
            "labels ZIP must contain exactly one {}; found {}".format(
                basename, matches
            )
        )
    raw = archive.read(matches[0])
    return json.loads(raw.decode("utf-8-sig")), raw


def read_official_training_labels(labels_zip: Path) -> Tuple[List[Mapping[str, Any]], Mapping[str, str]]:
    """Read all labeled train/validation metadata without extracting the ZIP."""

    with zipfile.ZipFile(labels_zip, "r") as archive:
        train, _ = _read_zip_json(archive, "train.json")
        validation, _ = _read_zip_json(archive, "validation.json")
        labels, _ = _read_zip_json(archive, "labels.json")
    for name, rows in (("train.json", train), ("validation.json", validation)):
        if not isinstance(rows, list) or not all(isinstance(item, dict) for item in rows):
            raise ValueError("official {} must be a list of objects".format(name))
    if not isinstance(labels, dict):
        raise ValueError("official labels.json must be an object")
    return list(train) + list(validation), {
        str(key): str(value) for key, value in labels.items()
    }


def _load_split_spec(path: Path) -> Tuple[List[str], Dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("split specification must be a JSON object")
    classes = payload.get("classes")
    counts = payload.get("counts")
    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ValueError("split specification classes must be a list of strings")
    if not isinstance(counts, dict) or set(counts) != set(SPLIT_ORDER):
        raise ValueError("split counts must contain train, val and test")
    parsed_counts = {split: int(counts[split]) for split in SPLIT_ORDER}
    if any(count <= 0 for count in parsed_counts.values()):
        raise ValueError("split class counts must be positive")
    if sum(parsed_counts.values()) != len(classes):
        raise ValueError("split counts do not add up to the ordered class list")
    normalised = [_normalise_template(value) for value in classes]
    if len(normalised) != len(set(normalised)):
        raise ValueError("split specification contains duplicate classes")
    return classes, parsed_counts


def _split_classes(classes: Sequence[str], counts: Mapping[str, int]) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = {}
    offset = 0
    for split in SPLIT_ORDER:
        end = offset + int(counts[split])
        output[split] = list(classes[offset:end])
        offset = end
    return output


def _load_canonical_manifest(path: Path) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("canonical manifest must be a JSON object")
    if "classes" in payload:
        payload = payload["classes"]
    elif set(SPLIT_ORDER).issubset(payload):
        merged: Dict[str, Any] = {}
        for split in SPLIT_ORDER:
            split_payload = payload[split]
            if not isinstance(split_payload, dict):
                raise ValueError("canonical split manifests must map classes to IDs")
            overlap = set(merged).intersection(split_payload)
            if overlap:
                raise ValueError("classes repeated in canonical manifest: {}".format(sorted(overlap)))
            merged.update(split_payload)
        payload = merged
    if not isinstance(payload, dict):
        raise ValueError("canonical manifest must map class names to video ID lists")
    result: Dict[str, List[str]] = {}
    for class_name, video_ids in payload.items():
        if not isinstance(video_ids, list):
            raise ValueError("canonical video IDs for each class must be a list")
        ids = [str(video_id) for video_id in video_ids]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate canonical video ID for {!r}".format(class_name))
        result[_normalise_template(str(class_name))] = ids
    return result


def _canonical_list_directory(path: Path) -> bool:
    return all((path / (split + ".list")).is_file() for split in SPLIT_ORDER)


def _discover_canonical_list_directory(source: Path) -> Optional[Path]:
    candidates = (
        source / "smsm-100",
        source / "CMN" / "smsm-100",
        ROOT / "data" / "fsar_splits" / "ssv2_cmn",
        ROOT / "data" / "fsar_splits" / "smsm-100",
        ROOT / "dataset" / "smsm-100",
    )
    return next((path.resolve() for path in candidates if _canonical_list_directory(path)), None)


def _load_cmn_lists(
    directory: Path,
    classes_by_split: Mapping[str, Sequence[str]],
    per_class: int,
) -> Dict[str, List[str]]:
    """Load the official CMN ``smsm-100/{train,val,test}.list`` files."""

    result: Dict[str, List[str]] = {}
    all_ids: set = set()
    for split in SPLIT_ORDER:
        path = directory / (split + ".list")
        if not path.is_file():
            raise FileNotFoundError("canonical CMN list not found: {}".format(path))
        first_seen_classes: List[str] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            value = line.strip().replace("\\", "/")
            if not value or value.startswith("#"):
                continue
            class_name, separator, video_id = value.rpartition("/")
            if not separator or not class_name or not video_id:
                raise ValueError(
                    "{}:{}: expected 'class name/video_id'".format(path, line_number)
                )
            normalised = _normalise_template(class_name)
            if normalised not in result:
                result[normalised] = []
                first_seen_classes.append(normalised)
            if video_id in all_ids:
                raise ValueError(
                    "duplicate video ID {} in canonical CMN lists".format(video_id)
                )
            all_ids.add(video_id)
            result[normalised].append(video_id)

        expected_order = [
            _normalise_template(class_name) for class_name in classes_by_split[split]
        ]
        if first_seen_classes != expected_order:
            raise ValueError(
                "{} class order does not match the split specification".format(path)
            )
        for class_name in expected_order:
            if len(result[class_name]) != per_class:
                raise ValueError(
                    "canonical CMN class {!r} has {} IDs, expected {}".format(
                        class_name, len(result[class_name]), per_class
                    )
                )
    return result


def _stable_video_key(seed: int, class_name: str, video_id: str) -> Tuple[str, str]:
    value = "{}\0{}\0{}".format(seed, class_name, video_id).encode("utf-8")
    return hashlib.sha256(value).hexdigest(), video_id


def _video_index(video_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    suffix_rank = {suffix: rank for rank, suffix in enumerate(VIDEO_SUFFIXES)}
    for path in video_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffix_rank:
            continue
        video_id = path.stem
        previous = index.get(video_id)
        if previous is None:
            index[video_id] = path
            continue
        old_rank = suffix_rank[previous.suffix.lower()]
        new_rank = suffix_rank[path.suffix.lower()]
        if new_rank < old_rank:
            index[video_id] = path
        elif new_rank == old_rank:
            raise ValueError(
                "duplicate video ID {} at {} and {}".format(video_id, previous, path)
            )
    return index


def _relative_video_path(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def build_small(
    source: Path,
    *,
    labels_zip: Optional[Path] = None,
    video_root: Optional[Path] = None,
    split_spec: Path = DEFAULT_SPLIT_SPEC,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    provenance_path: Optional[Path] = None,
    canonical_manifest: Optional[Path] = None,
    canonical_manifest_dir: Optional[Path] = None,
    per_class: int = 100,
    seed: int = 916,
    workers: int = 4,
    frame_count_fn: Optional[Callable[[Path], int]] = None,
    allow_stable_hash_fallback: bool = False,
) -> Mapping[str, Any]:
    """Build deterministic SSv2-Small annotations from official train.json."""

    if per_class <= 0:
        raise ValueError("per_class must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    source = Path(source).resolve()
    labels_zip = (
        Path(labels_zip).resolve()
        if labels_zip is not None
        else source / DEFAULT_LABELS_ARCHIVE
    )
    video_root = (
        Path(video_root).resolve()
        if video_root is not None
        else source / "extracted" / "videos"
    )
    split_spec = Path(split_spec).resolve()
    output_dir = Path(output_dir).resolve()
    provenance_path = (
        Path(provenance_path).resolve()
        if provenance_path is not None
        else output_dir.parent / "ssv2_small_provenance.json"
    )
    canonical_manifest = (
        Path(canonical_manifest).resolve()
        if canonical_manifest is not None
        else None
    )
    canonical_manifest_dir = (
        Path(canonical_manifest_dir).resolve()
        if canonical_manifest_dir is not None
        else None
    )
    if canonical_manifest is not None and canonical_manifest_dir is not None:
        raise ValueError("provide only one of canonical_manifest and canonical_manifest_dir")
    if not labels_zip.is_file():
        raise FileNotFoundError("labels ZIP not found: {}".format(labels_zip))
    if not video_root.is_dir():
        raise FileNotFoundError("video root not found: {}".format(video_root))

    classes, split_counts = _load_split_spec(split_spec)
    classes_by_split = _split_classes(classes, split_counts)
    official_rows, official_labels = read_official_training_labels(labels_zip)
    official_class_names = {_normalise_template(name) for name in official_labels}
    missing_official = [
        name for name in classes if _normalise_template(name) not in official_class_names
    ]
    if missing_official:
        raise ValueError(
            "CMN classes absent from official labels.json: {}".format(missing_official)
        )

    candidates: Dict[str, List[str]] = {name: [] for name in classes}
    normalised_to_class = {_normalise_template(name): name for name in classes}
    seen_ids: set = set()
    for row in official_rows:
        if "id" not in row or "template" not in row:
            raise ValueError("official train.json entries require id and template")
        video_id = str(row["id"])
        if video_id in seen_ids:
            raise ValueError("duplicate video ID in official train.json: {}".format(video_id))
        seen_ids.add(video_id)
        class_name = normalised_to_class.get(_normalise_template(str(row["template"])))
        if class_name is not None:
            candidates[class_name].append(video_id)

    auto_manifest_dir = None
    if (
        canonical_manifest is None
        and canonical_manifest_dir is None
        and not allow_stable_hash_fallback
    ):
        auto_manifest_dir = _discover_canonical_list_directory(source)
    selected_manifest_dir = canonical_manifest_dir or auto_manifest_dir
    if selected_manifest_dir is not None:
        canonical = _load_cmn_lists(
            selected_manifest_dir, classes_by_split, per_class
        )
        canonical_kind: Optional[str] = "official_cmn_lists"
    elif canonical_manifest is not None:
        canonical = _load_canonical_manifest(canonical_manifest)
        canonical_kind = "canonical_json_manifest"
    else:
        if not allow_stable_hash_fallback:
            raise FileNotFoundError(
                "official CMN train.list/val.list/test.list were not found; provide "
                "--canonical-manifest-dir or explicitly allow the reconstructed "
                "selection with --stable-hash-fallback"
            )
        canonical = None
        canonical_kind = None
    selected: Dict[str, List[str]] = {}
    for class_name in classes:
        available = candidates[class_name]
        if len(available) < per_class:
            raise ValueError(
                "class {!r} has {} official training videos, needs {}".format(
                    class_name, len(available), per_class
                )
            )
        if canonical is not None:
            key = _normalise_template(class_name)
            if key not in canonical:
                raise ValueError("canonical manifest is missing class {!r}".format(class_name))
            ids = canonical[key]
            if len(ids) != per_class:
                raise ValueError(
                    "canonical class {!r} has {} IDs, expected {}".format(
                        class_name, len(ids), per_class
                    )
                )
            unavailable = sorted(set(ids) - set(available))
            if unavailable:
                raise ValueError(
                    "canonical IDs are absent from official train/validation labels for {!r}: {}".format(
                        class_name, unavailable
                    )
                )
            selected[class_name] = list(ids)
        else:
            selected[class_name] = sorted(
                available,
                key=lambda video_id: _stable_video_key(seed, class_name, video_id),
            )[:per_class]

    index = _video_index(video_root)
    missing_videos = [
        video_id
        for class_name in classes
        for video_id in selected[class_name]
        if video_id not in index
    ]
    if missing_videos:
        raise FileNotFoundError(
            "{} selected videos are missing under {} (first: {})".format(
                len(missing_videos), video_root, missing_videos[:10]
            )
        )

    ordered_paths = [
        index[video_id]
        for class_name in classes
        for video_id in selected[class_name]
    ]
    counter = frame_count_fn or probe_video_num_frames
    with ThreadPoolExecutor(max_workers=workers) as executor:
        frame_counts = list(executor.map(counter, ordered_paths))
    invalid_counts = [count for count in frame_counts if int(count) <= 0]
    if invalid_counts:
        raise ValueError("all selected videos must have a positive decoded frame count")

    label_ids = {class_name: index for index, class_name in enumerate(classes)}
    records_by_split: Dict[str, List[AnnotationRecord]] = {
        split: [] for split in SPLIT_ORDER
    }
    provenance_rows: Dict[str, List[Dict[str, Any]]] = {
        split: [] for split in SPLIT_ORDER
    }
    count_offset = 0
    class_to_split = {
        class_name: split
        for split, split_classes in classes_by_split.items()
        for class_name in split_classes
    }
    for class_name in classes:
        split = class_to_split[class_name]
        label = label_ids[class_name]
        for video_id in selected[class_name]:
            path = index[video_id]
            frame_count = int(frame_counts[count_offset])
            count_offset += 1
            relative_path = _relative_video_path(path, video_root)
            record = AnnotationRecord(relative_path, frame_count, label)
            records_by_split[split].append(record)
            provenance_rows[split].append(
                {
                    "id": video_id,
                    "path": relative_path,
                    "num_frames": frame_count,
                    "label": label,
                    "class": class_name,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        split: write_annotations(records_by_split[split], output_dir / (split + ".txt"))
        for split in SPLIT_ORDER
    }
    report = validate_split_annotations(
        outputs["train"],
        outputs["val"],
        outputs["test"],
        check_paths=True,
        path_root=video_root,
        require_disjoint_labels=True,
        expected_class_counts=split_counts,
        min_samples_per_class=per_class,
    )
    if not report.valid:
        raise ValueError("generated annotations failed validation: {}".format(report.errors))

    source_hashes: Dict[str, Any] = {
        "labels_zip": {
            "path": str(labels_zip),
            "size": labels_zip.stat().st_size,
            "sha256": sha256_file(labels_zip),
        },
        "split_spec": {
            "path": str(split_spec),
            "size": split_spec.stat().st_size,
            "sha256": sha256_file(split_spec),
        },
    }
    if canonical_manifest is not None:
        source_hashes["canonical_manifest"] = {
            "path": str(canonical_manifest),
            "size": canonical_manifest.stat().st_size,
            "sha256": sha256_file(canonical_manifest),
        }
    if selected_manifest_dir is not None:
        source_hashes["canonical_cmn_lists"] = {
            split: {
                "path": str(selected_manifest_dir / (split + ".list")),
                "size": (selected_manifest_dir / (split + ".list")).stat().st_size,
                "sha256": sha256_file(selected_manifest_dir / (split + ".list")),
            }
            for split in SPLIT_ORDER
        }
    extraction_marker = video_root.parent / "extraction.complete.json"
    archive_hashes: Optional[Mapping[str, Any]] = None
    if extraction_marker.is_file():
        marker_payload = json.loads(extraction_marker.read_text(encoding="utf-8"))
        archive_hashes = marker_payload.get("source_hashes_after")

    provenance: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "dataset": "somethingcmn",
        "selection_method": (
            canonical_kind if canonical_kind is not None else "stable_hash_reconstruction"
        ),
        "reconstruction_notice": (
            None
            if canonical is not None
            else "Reproducible CMN reconstruction; not an upstream canonical video-ID manifest."
        ),
        "seed": int(seed),
        "per_class": int(per_class),
        "video_root": str(video_root),
        "ordered_classes": classes,
        "class_counts": split_counts,
        "records": {split: len(records_by_split[split]) for split in SPLIT_ORDER},
        "annotations": {split: str(path) for split, path in outputs.items()},
        "source_hashes": source_hashes,
        "video_archive_source_hashes": archive_hashes,
        "selection": provenance_rows,
        "validation": report.to_dict(),
    }
    _write_json_atomic(provenance_path, provenance)
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely extract SSv2 and build the class-disjoint CMN subset"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract", help="stream the official numbered TGZ parts into WebM files"
    )
    extract.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    extract.add_argument("--destination", type=Path, default=None)
    extract.add_argument("--part-prefix", default=DEFAULT_PART_PREFIX)
    extract.add_argument(
        "--expected-video-count",
        type=int,
        default=DEFAULT_EXPECTED_VIDEOS,
        help="use 0 to disable the count check",
    )

    build = subparsers.add_parser(
        "build-small", help="select and annotate the 100-class CMN subset"
    )
    build.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    build.add_argument("--labels-zip", type=Path, default=None)
    build.add_argument("--video-root", type=Path, default=None)
    build.add_argument("--split-spec", type=Path, default=DEFAULT_SPLIT_SPEC)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    build.add_argument("--provenance", type=Path, default=None)
    build.add_argument("--canonical-manifest", type=Path, default=None)
    build.add_argument(
        "--canonical-manifest-dir",
        type=Path,
        default=None,
        help="directory containing official CMN train.list, val.list and test.list",
    )
    build.add_argument("--per-class", type=int, default=100)
    build.add_argument("--seed", type=int, default=916)
    build.add_argument("--workers", type=int, default=4)
    build.add_argument(
        "--stable-hash-fallback",
        action="store_true",
        help="explicitly use seed/class/video-ID hashing instead of canonical CMN lists",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            expected = (
                None if args.expected_video_count == 0 else args.expected_video_count
            )
            payload = extract_archive(
                args.source,
                destination=args.destination,
                prefix=args.part_prefix,
                expected_video_count=expected,
            )
        else:
            payload = build_small(
                args.source,
                labels_zip=args.labels_zip,
                video_root=args.video_root,
                split_spec=args.split_spec,
                output_dir=args.output_dir,
                provenance_path=args.provenance,
                canonical_manifest=args.canonical_manifest,
                canonical_manifest_dir=args.canonical_manifest_dir,
                per_class=args.per_class,
                seed=args.seed,
                workers=args.workers,
                allow_stable_hash_fallback=args.stable_hash_fallback,
            )
        summary = dict(payload)
        if "selection" in summary:
            summary["selection"] = {
                split: len(rows) for split, rows in summary["selection"].items()
            }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, ImportError, zipfile.BadZipFile, tarfile.TarError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
