from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_kinetics400_windows import (
    RemoteItem,
    _read_urls,
    _safe_member,
    _selected,
    status,
)


def _item(group: str, relative: str, size: int = 10) -> RemoteItem:
    return RemoteItem(
        group=group,
        url=f"https://example.invalid/{relative}",
        relative_path=relative,
        expected_bytes=size,
        etag="multipart-etag-2",
        last_modified="now",
    )


def test_safe_member_rejects_traversal_and_absolute_paths(tmp_path: Path) -> None:
    assert _safe_member(tmp_path, "class/video.mp4") == (tmp_path / "class" / "video.mp4").resolve()
    with pytest.raises(ValueError):
        _safe_member(tmp_path, "../escape.mp4")
    with pytest.raises(ValueError):
        _safe_member(tmp_path, "/absolute.mp4")


def test_read_urls_rejects_duplicate_manifest_rows(tmp_path: Path) -> None:
    path = tmp_path / "paths.txt"
    path.write_text("https://example.invalid/a\nhttps://example.invalid/a\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _read_urls(path)


def test_group_selection_is_explicit() -> None:
    items = [_item("train", "train/a.tar.gz"), _item("val", "val/b.tar.gz")]
    assert _selected(items, ["val"]) == [items[1]]
    assert _selected(items, ["all"]) == items
    with pytest.raises(ValueError):
        _selected(items, ["unknown"])


def test_status_requires_byte_exact_final_files_and_tracks_partials(tmp_path: Path) -> None:
    items = [_item("train", "train/a.tar.gz"), _item("val", "val/b.tar.gz")]
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "remote_objects.json").write_text(
        json.dumps({"objects": [item.__dict__ for item in items]}), encoding="utf-8"
    )
    complete = tmp_path / "raw" / "train" / "a.tar.gz"
    complete.parent.mkdir(parents=True)
    complete.write_bytes(b"x" * 10)
    partial = tmp_path / "raw" / "val" / "b.tar.gz.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"x" * 4)

    report = status(tmp_path, ["all"])

    assert report["complete"] == 1
    assert report["partial"] == 1
    assert report["missing"] == 0
    assert report["bytes_present"] == 14
    assert report["valid"] is False


def test_status_does_not_count_aria2_sparse_logical_length_as_downloaded(
    tmp_path: Path,
) -> None:
    item = _item("train", "train/a.tar.gz", size=1_000)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "remote_objects.json").write_text(
        json.dumps({"objects": [item.__dict__]}), encoding="utf-8"
    )
    partial = tmp_path / "raw" / "train" / "a.tar.gz.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"x" * 1_000)
    Path(str(partial) + ".aria2").write_bytes(b"control")

    report = status(tmp_path, ["all"])

    assert report["partial"] == 1
    assert report["aria2_in_progress"] == 1
    assert report["bytes_present"] == 0
    assert report["complete"] == 0
