from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fsar.data import (
    AnnotationRecord,
    DirectVideoDataset,
    audit_annotations,
    build_video_transform,
    generate_split_annotations,
    parse_annotation_line,
    read_annotations,
    sample_frame_indices,
    validate_annotations,
    validate_split_annotations,
    write_annotations,
)


def test_parse_annotation_from_right_preserves_spaces() -> None:
    row = parse_annotation_line("D:/data/a directory/clip one.webm 37 12")
    assert row == AnnotationRecord("D:/data/a directory/clip one.webm", 37, 12)
    assert row.video_path == row.path


@pytest.mark.parametrize(
    "line",
    ["clip.webm", "clip.webm nope 1", "clip.webm 0 1", "clip.webm 8 -1"],
)
def test_parse_annotation_rejects_malformed_rows(line: str) -> None:
    with pytest.raises(ValueError):
        parse_annotation_line(line)


def test_annotation_round_trip_and_validation_with_space_path(tmp_path: Path) -> None:
    video = tmp_path / "folder with spaces" / "clip one.webm"
    video.parent.mkdir()
    video.touch()
    annotation = tmp_path / "train.txt"
    write_annotations([AnnotationRecord(str(video), 19, 4)], annotation)

    assert read_annotations(annotation) == [AnnotationRecord(str(video), 19, 4)]
    report = validate_annotations(annotation)
    assert report.valid
    assert report["stats"]["records"] == 1
    mismatch = validate_annotations(
        annotation, verify_frame_counts=True, frame_count_fn=lambda _: 20
    )
    assert not mismatch.valid
    assert mismatch.stats["frame_count_mismatches"][0]["actual"] == 20


def test_interval_and_seeded_random_sampling_are_window_bounded() -> None:
    assert sample_frame_indices(16, 8) == [1, 3, 5, 7, 9, 11, 13, 15]
    random_a = sample_frame_indices(80, 8, random_select=True, seed=7)
    random_b = sample_frame_indices(80, 8, strategy="random", seed=7)
    assert random_a == random_b
    assert all(index * 10 <= frame < (index + 1) * 10 for index, frame in enumerate(random_a))

    head_removed = sample_frame_indices(80, 8, temporal_window=(0.25, 1.0))
    tail_removed = sample_frame_indices(80, 8, temporal_window=(0.0, 0.75))
    middle = sample_frame_indices(80, 8, temporal_window=(0.125, 0.875))
    assert min(head_removed) >= 20 and max(head_removed) < 80
    assert min(tail_removed) >= 0 and max(tail_removed) < 60
    assert min(middle) >= 10 and max(middle) < 70


def test_short_video_sampling_repeats_valid_indices() -> None:
    sampled = sample_frame_indices(3, 8)
    assert len(sampled) == 8
    assert set(sampled) <= {0, 1, 2}


def test_direct_video_dataset_sampling_seed_changes_reproducibly_by_epoch(
    tmp_path: Path,
) -> None:
    annotation = write_annotations(
        [AnnotationRecord("clip.webm", 8, 0)], tmp_path / "train.txt"
    )
    dataset = DirectVideoDataset(annotation, is_train=True, seed=916)
    epoch_zero = dataset._seed_for_index(0)
    dataset.set_epoch(3)
    epoch_three = dataset._seed_for_index(0)

    replay = DirectVideoDataset(annotation, is_train=True, seed=916)
    replay.set_epoch(3)
    assert epoch_zero != epoch_three
    assert epoch_three == replay._seed_for_index(0)


def test_ssv2_transform_never_enables_horizontal_flip() -> None:
    transform = build_video_transform(
        image_size=8,
        is_train=True,
        dataset_name="Something-Something-V2",
        horizontal_flip=True,
    )
    assert transform.horizontal_flip is False
    clip = torch.rand(8, 3, 10, 12)
    output = transform(clip)
    assert output.shape == (8, 3, 8, 8)


def test_generate_and_validate_three_split_annotations_without_video_decode(
    tmp_path: Path,
) -> None:
    video_root = tmp_path / "video root"
    video_root.mkdir()
    manifests = {}
    for split, label in (("train", 0), ("val", 1), ("test", 2)):
        video = video_root / "{} clip.webm".format(split)
        video.touch()
        manifest = tmp_path / "{}.list".format(split)
        manifest.write_text("{} {}\n".format(video.name, label), encoding="utf-8")
        manifests[split] = manifest

    outputs = generate_split_annotations(
        video_root,
        tmp_path / "annotations",
        split_manifests=manifests,
        frame_count_fn=lambda _: 24,
    )
    assert set(outputs) == {"train", "val", "test"}
    report = validate_split_annotations(
        outputs["train"], outputs["val"], outputs["test"]
    )
    assert report.valid, report.errors
    assert read_annotations(outputs["train"])[0].num_frames == 24


def test_split_validation_detects_class_and_path_leakage(tmp_path: Path) -> None:
    video = tmp_path / "same video.webm"
    video.touch()
    train = write_annotations([AnnotationRecord(str(video), 8, 0)], tmp_path / "train.txt")
    val = write_annotations([AnnotationRecord(str(video), 8, 1)], tmp_path / "val.txt")
    other = tmp_path / "other.webm"
    other.touch()
    test = write_annotations([AnnotationRecord(str(other), 8, 0)], tmp_path / "test.txt")

    report = validate_split_annotations(train, val, test)
    assert not report.valid
    assert any("multiple splits" in error for error in report.errors)
    assert any("share labels" in error for error in report.errors)


def test_audit_is_json_serializable_and_seeded(tmp_path: Path) -> None:
    records = [
        AnnotationRecord("missing clip {}.webm".format(index), 8 + index, index % 2)
        for index in range(6)
    ]
    annotation = write_annotations(records, tmp_path / "audit.txt")
    first = audit_annotations(
        annotation, check_paths=False, sample_count=3, seed=916, histogram_bins=4
    )
    second = audit_annotations(
        annotation, check_paths=False, sample_count=3, seed=916, histogram_bins=4
    )
    assert first["random_samples"] == second["random_samples"]
    assert first["splits"]["all"]["class_counts"] == {"0": 3, "1": 3}
    json.dumps(first)
