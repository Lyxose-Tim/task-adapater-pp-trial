from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from fsar.data import read_annotations
from scripts.prepare_ssv2 import build_small, extract_archive


def _write_split_archive(source: Path, members, split_at=None) -> None:
    payload = io.BytesIO()
    with gzip.GzipFile(fileobj=payload, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for name, content in members:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
    raw = payload.getvalue()
    if split_at is None:
        split_at = max(1, len(raw) // 2)
    (source / "20bn-something-something-v2-00").write_bytes(raw[:split_at])
    (source / "20bn-something-something-v2-01").write_bytes(raw[split_at:])


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_labels_zip(path: Path, classes, per_class=4, validation_ids=()) -> dict:
    rows = []
    ids = {}
    next_id = 100
    for class_name in classes:
        official = class_name.replace("something", "[something]")
        ids[class_name] = []
        for _ in range(per_class):
            video_id = str(next_id)
            next_id += 1
            ids[class_name].append(video_id)
            rows.append({"id": video_id, "template": official, "label": "example"})
    labels = {
        class_name.replace("something", "[something]"): str(index)
        for index, class_name in enumerate(classes)
    }
    validation_ids = {str(video_id) for video_id in validation_ids}
    train_rows = [row for row in rows if row["id"] not in validation_ids]
    validation_rows = [row for row in rows if row["id"] in validation_ids]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("labels/train.json", json.dumps(train_rows))
        archive.writestr("labels/validation.json", json.dumps(validation_rows))
        archive.writestr("labels/labels.json", json.dumps(labels))
    return ids


def test_extract_streams_parts_preserves_sources_and_writes_marker(tmp_path):
    source = tmp_path / "SSv2"
    source.mkdir()
    _write_split_archive(
        source,
        [
            ("20bn-something-something-v2/100.webm", b"video-100"),
            ("20bn-something-something-v2/101.webm", b"video-101"),
        ],
        split_at=17,
    )
    parts = sorted(source.glob("*-0*"))
    before = {path.name: _digest(path) for path in parts}

    result = extract_archive(source, expected_video_count=2)

    assert result["video_count"] == 2
    assert (source / "extracted/videos/100.webm").read_bytes() == b"video-100"
    assert (source / "extracted/videos/101.webm").read_bytes() == b"video-101"
    assert (source / "extracted/extraction.complete.json").is_file()
    assert not (source / "extracted.incomplete").exists()
    assert {path.name: _digest(path) for path in parts} == before
    assert not (source / "20bn-something-something-v2.tgz").exists()

    # A completed extraction is idempotent and validates the original hashes.
    assert extract_archive(source, expected_video_count=2)["video_count"] == 2


@pytest.mark.parametrize(
    "unsafe_name",
    ["../escape.webm", "folder/../../escape.webm", "C:/escape.webm", "..\\escape.webm"],
)
def test_extract_rejects_path_traversal_and_keeps_failure_log(tmp_path, unsafe_name):
    source = tmp_path / "SSv2"
    source.mkdir()
    _write_split_archive(source, [(unsafe_name, b"bad")])

    with pytest.raises(ValueError, match="unsafe archive member path"):
        extract_archive(source, expected_video_count=1)

    assert not (tmp_path / "escape.webm").exists()
    failure = json.loads(
        (source / "extracted.incomplete/failure.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed"
    assert failure["error_type"] == "ValueError"


def test_build_small_stable_hash_is_reproducible_and_preserves_class_order(tmp_path):
    source = tmp_path / "SSv2"
    video_root = source / "extracted/videos"
    video_root.mkdir(parents=True)
    classes = [
        "Holding something",
        "Moving something",
        "Putting something",
        "Dropping something",
        "Opening something",
        "Closing something",
    ]
    split_spec = tmp_path / "split.json"
    split_spec.write_text(
        json.dumps({"classes": classes, "counts": {"train": 3, "val": 1, "test": 2}}),
        encoding="utf-8",
    )
    labels_zip = source / "labels.zip"
    all_ids = _write_labels_zip(labels_zip, classes, per_class=4)
    for video_ids in all_ids.values():
        for video_id in video_ids:
            (video_root / (video_id + ".webm")).write_bytes(b"synthetic")

    def frames(path):
        return int(Path(path).stem) % 7 + 1

    first = build_small(
        source,
        labels_zip=labels_zip,
        video_root=video_root,
        split_spec=split_spec,
        output_dir=tmp_path / "first/annotations",
        per_class=2,
        seed=916,
        workers=2,
        frame_count_fn=frames,
        allow_stable_hash_fallback=True,
    )
    second = build_small(
        source,
        labels_zip=labels_zip,
        video_root=video_root,
        split_spec=split_spec,
        output_dir=tmp_path / "second/annotations",
        per_class=2,
        seed=916,
        workers=2,
        frame_count_fn=frames,
        allow_stable_hash_fallback=True,
    )

    assert first["selection_method"] == "stable_hash_reconstruction"
    assert first["records"] == {"train": 6, "val": 2, "test": 4}
    for split in ("train", "val", "test"):
        first_text = (tmp_path / "first/annotations" / (split + ".txt")).read_text()
        second_text = (tmp_path / "second/annotations" / (split + ".txt")).read_text()
        assert first_text == second_text
        records = read_annotations(tmp_path / "first/annotations" / (split + ".txt"))
        assert all("\\" not in record.path for record in records)
    assert {record.label for record in read_annotations(tmp_path / "first/annotations/train.txt")} == {0, 1, 2}
    assert {record.label for record in read_annotations(tmp_path / "first/annotations/val.txt")} == {3}
    assert {record.label for record in read_annotations(tmp_path / "first/annotations/test.txt")} == {4, 5}
    provenance = json.loads(
        (tmp_path / "first/ssv2_small_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["validation"]["valid"] is True
    assert sum(len(rows) for rows in provenance["selection"].values()) == 12


def test_build_small_uses_exact_canonical_ids(tmp_path):
    source = tmp_path / "SSv2"
    video_root = source / "extracted/videos"
    video_root.mkdir(parents=True)
    classes = ["Holding something", "Moving something", "Opening something"]
    split_spec = tmp_path / "split.json"
    split_spec.write_text(
        json.dumps({"classes": classes, "counts": {"train": 1, "val": 1, "test": 1}}),
        encoding="utf-8",
    )
    labels_zip = source / "labels.zip"
    all_ids = _write_labels_zip(labels_zip, classes, per_class=3)
    for ids in all_ids.values():
        for video_id in ids:
            (video_root / (video_id + ".webm")).write_bytes(b"synthetic")
    canonical = tmp_path / "canonical.json"
    chosen = {class_name: [ids[-1]] for class_name, ids in all_ids.items()}
    canonical.write_text(json.dumps({"classes": chosen}), encoding="utf-8")

    result = build_small(
        source,
        labels_zip=labels_zip,
        video_root=video_root,
        split_spec=split_spec,
        output_dir=tmp_path / "annotations",
        canonical_manifest=canonical,
        per_class=1,
        workers=1,
        frame_count_fn=lambda path: 5,
    )

    assert result["selection_method"] == "canonical_json_manifest"
    selected = [row["id"] for split in result["selection"].values() for row in split]
    assert selected == [chosen[class_name][0] for class_name in classes]


def test_build_small_prefers_official_cmn_list_directory_and_validation_ids(tmp_path):
    source = tmp_path / "SSv2"
    video_root = source / "extracted/videos"
    video_root.mkdir(parents=True)
    classes = ["Holding something", "Moving something", "Opening something"]
    split_spec = tmp_path / "split.json"
    split_spec.write_text(
        json.dumps({"classes": classes, "counts": {"train": 1, "val": 1, "test": 1}}),
        encoding="utf-8",
    )
    labels_zip = source / "labels.zip"
    # Construct IDs first so each canonical list can deliberately select an ID
    # that exists only in official validation.json.
    provisional = source / "provisional.zip"
    all_ids = _write_labels_zip(provisional, classes, per_class=2)
    provisional.unlink()
    validation_ids = [ids[-1] for ids in all_ids.values()]
    all_ids = _write_labels_zip(
        labels_zip, classes, per_class=2, validation_ids=validation_ids
    )
    for ids in all_ids.values():
        for video_id in ids:
            (video_root / (video_id + ".webm")).write_bytes(b"synthetic")
    cmn = source / "smsm-100"
    cmn.mkdir()
    for split, class_name in zip(("train", "val", "test"), classes):
        (cmn / (split + ".list")).write_text(
            class_name + "/" + all_ids[class_name][-1] + "\n", encoding="utf-8"
        )

    result = build_small(
        source,
        labels_zip=labels_zip,
        video_root=video_root,
        split_spec=split_spec,
        output_dir=tmp_path / "annotations",
        per_class=1,
        workers=1,
        frame_count_fn=lambda path: 5,
    )

    assert result["selection_method"] == "official_cmn_lists"
    selected = [row["id"] for rows in result["selection"].values() for row in rows]
    assert selected == [all_ids[class_name][-1] for class_name in classes]
