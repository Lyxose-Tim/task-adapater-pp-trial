#!/usr/bin/env python3
"""Prepare the locally available UCF101 and HMDB51 videos for FSAR.

The official Task-Adapter class order defines disjoint base/validation/test
class splits.  This script keeps that order, probes AVI frame counts without
extracting every frame, and writes the ``path num_frames label`` annotations
consumed by :mod:`fsar.data`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fsar.data import AnnotationRecord, probe_video_num_frames, validate_split_annotations, write_annotations


DATASETS: Mapping[str, Mapping[str, object]] = {
    "ucf101": {
        "video_root": ROOT / "dataset" / "UCF101" / "UCF-101",
        "split": ROOT / "data" / "fsar_splits" / "ucf101.json",
        "output": ROOT / "dataset" / "UCF101" / "annotations",
        "corpus": ROOT / "corpus" / "classes_ucf101_fsar.yml",
        "source_corpus": ROOT / "data" / "sub_actions" / "ucf101_sub_actions.json",
    },
    "hmdb51": {
        "video_root": ROOT / "dataset" / "HMDB51",
        "split": ROOT / "data" / "fsar_splits" / "hmdb51.json",
        "output": ROOT / "dataset" / "HMDB51" / "annotations",
        "corpus": ROOT / "corpus" / "classes_hmdb51_fsar.yml",
        "source_corpus": ROOT / "corpus" / "classes_hmdb51.yml",
    },
}


def _read_split(path: Path) -> Tuple[List[str], Dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    classes = payload.get("classes")
    counts = payload.get("counts")
    if not isinstance(classes, list) or not classes or not isinstance(counts, dict):
        raise ValueError("invalid canonical split file: {}".format(path))
    normalized_counts = {name: int(counts[name]) for name in ("train", "val", "test")}
    if sum(normalized_counts.values()) != len(classes):
        raise ValueError("split class counts do not cover every class in {}".format(path))
    return [str(name) for name in classes], normalized_counts


def _class_split(index: int, counts: Mapping[str, int]) -> str:
    if index < counts["train"]:
        return "train"
    if index < counts["train"] + counts["val"]:
        return "val"
    return "test"


def _probe(path: Path) -> Tuple[Path, int]:
    return path, probe_video_num_frames(path, num_threads=1)


def _probe_all(paths: Iterable[Path], workers: int) -> Dict[Path, int]:
    ordered = list(paths)
    result: Dict[Path, int] = {}
    errors: List[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_probe, path): path for path in ordered}
        for future in as_completed(futures):
            path = futures[future]
            try:
                resolved, count = future.result()
            except Exception as exc:  # report every corrupt video together
                errors.append("{}: {}".format(path, exc))
            else:
                result[resolved] = count
    if errors:
        preview = "\n".join(errors[:20])
        suffix = "\n... {} more".format(len(errors) - 20) if len(errors) > 20 else ""
        raise RuntimeError("failed to probe {} videos:\n{}{}".format(len(errors), preview, suffix))
    return result


def _humanize(name: str) -> str:
    if "_" in name:
        return name.replace("_", " ")
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def _load_source_corpus(dataset: str, path: Path) -> Mapping[str, object]:
    if dataset == "ucf101":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("source corpus must be a mapping: {}".format(path))
    return value


def _stages(entry: object) -> List[str]:
    if isinstance(entry, list):
        values = entry
    elif isinstance(entry, dict):
        values = entry.get("sub_act_en_li") or entry.get("sub_actions")
    else:
        values = None
    if not isinstance(values, list) or len(values) != 3 or not all(str(x).strip() for x in values):
        raise ValueError("every source corpus entry must contain exactly three stages")
    return [str(value).strip() for value in values]


def _write_ordered_corpus(dataset: str, classes: Sequence[str], source: Path, destination: Path) -> None:
    raw = _load_source_corpus(dataset, source)
    by_normalized = {_normalized_name(str(key)): value for key, value in raw.items()}
    output: Dict[str, object] = {}
    for class_name in classes:
        key = _normalized_name(class_name)
        if key not in by_normalized:
            raise KeyError("class {!r} is missing from {}".format(class_name, source))
        readable = _humanize(class_name)
        output[readable] = {"label": readable, "sub_act_en_li": _stages(by_normalized[key])}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(output, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    temporary.replace(destination)


def prepare(dataset: str, workers: int) -> Mapping[str, object]:
    specification = DATASETS[dataset]
    video_root = Path(specification["video_root"])
    split_path = Path(specification["split"])
    output_dir = Path(specification["output"])
    corpus_path = Path(specification["corpus"])
    source_corpus = Path(specification["source_corpus"])
    classes, counts = _read_split(split_path)

    actual_dirs = {path.name: path for path in video_root.iterdir() if path.is_dir()}
    missing = [name for name in classes if name not in actual_dirs]
    extras = sorted(set(actual_dirs).difference(classes))
    if missing or extras:
        raise ValueError("class directories differ from canonical split; missing={}, extra={}".format(missing, extras))

    videos_by_class: Dict[str, List[Path]] = {}
    for name in classes:
        videos = sorted(
            path for path in actual_dirs[name].iterdir()
            if path.is_file() and path.suffix.casefold() == ".avi"
        )
        if not videos:
            raise ValueError("class contains no AVI videos: {}".format(actual_dirs[name]))
        videos_by_class[name] = videos

    frame_counts = _probe_all(
        (path for name in classes for path in videos_by_class[name]), workers
    )
    records: Dict[str, List[AnnotationRecord]] = {name: [] for name in ("train", "val", "test")}
    for label, class_name in enumerate(classes):
        split = _class_split(label, counts)
        records[split].extend(
            AnnotationRecord(str(path.resolve()), frame_counts[path], label)
            for path in videos_by_class[class_name]
        )

    outputs = {
        split: write_annotations(rows, output_dir / "{}.txt".format(split), sort_records=True)
        for split, rows in records.items()
    }
    _write_ordered_corpus(dataset, classes, source_corpus, corpus_path)
    report = validate_split_annotations(
        outputs["train"], outputs["val"], outputs["test"],
        expected_class_counts=counts, require_disjoint_labels=True,
    )
    if not report.valid:
        raise RuntimeError("generated annotations failed validation: {}".format(report.errors))

    summary = {
        "dataset": dataset,
        "video_root": str(video_root.resolve()),
        "videos": sum(len(rows) for rows in records.values()),
        "classes": len(classes),
        "split_classes": counts,
        "split_videos": {split: len(rows) for split, rows in records.items()},
        "annotations": {split: str(path.resolve()) for split, path in outputs.items()},
        "corpus": str(corpus_path.resolve()),
        "validation": report.to_dict(),
    }
    (output_dir / "preparation_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("ucf101", "hmdb51", "all"), nargs="?", default="all")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    selected = tuple(DATASETS) if args.dataset == "all" else (args.dataset,)
    summaries = [prepare(name, args.workers) for name in selected]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
