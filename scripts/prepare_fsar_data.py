#!/usr/bin/env python3
"""Generate, validate and audit FSAR train/val/test annotations.

Examples
--------
Generate from split manifests (each row may be ``path label`` or the complete
``path num_frames label`` format)::

    python scripts/prepare_fsar_data.py generate --video-root dataset/ssv2 \
      --output-dir dataset/smsm_cmn/annotations --train train.list \
      --val val.list --test test.list --label-map labels.json

Audit existing annotations without extracting all frames::

    python scripts/prepare_fsar_data.py audit \
      --annotation-dir dataset/smsm_cmn/annotations --output audit.json \
      --preview-dir audit_previews
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fsar.data import (
    audit_annotations,
    generate_split_annotations,
    save_audit_previews,
    validate_split_annotations,
)


def _add_split_inputs(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument("--annotation-dir", type=Path, default=None)
    parser.add_argument("--train", "--train-manifest", dest="train", type=Path, required=required)
    parser.add_argument("--val", "--val-manifest", dest="val", type=Path, required=required)
    parser.add_argument("--test", "--test-manifest", dest="test", type=Path, required=required)


def _split_paths(args: argparse.Namespace) -> Dict[str, Path]:
    paths: Dict[str, Optional[Path]] = {
        "train": getattr(args, "train", None),
        "val": getattr(args, "val", None),
        "test": getattr(args, "test", None),
    }
    if args.annotation_dir is not None:
        for split in paths:
            if paths[split] is None:
                paths[split] = args.annotation_dir / "{}.txt".format(split)
    missing = [split for split, path in paths.items() if path is None]
    if missing:
        raise ValueError(
            "provide --annotation-dir or explicit files for: {}".format(
                ", ".join(missing)
            )
        )
    return {split: path for split, path in paths.items() if path is not None}


def _expected_class_counts(value: Optional[str]) -> Optional[Mapping[str, int]]:
    if value is None:
        return None
    fields = [part.strip() for part in value.split(",")]
    if len(fields) == 3 and all("=" not in part for part in fields):
        return dict(zip(("train", "val", "test"), (int(part) for part in fields)))
    output: Dict[str, int] = {}
    for field in fields:
        split, separator, count = field.partition("=")
        if not separator or split not in {"train", "val", "test"}:
            raise argparse.ArgumentTypeError(
                "expected class counts such as 64,12,24 or train=64,val=12,test=24"
            )
        output[split] = int(count)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and audit direct-video few-shot action data"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="generate train/val/test annotation files"
    )
    generate.add_argument("--video-root", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--train", "--train-manifest", dest="train", type=Path)
    generate.add_argument("--val", "--val-manifest", dest="val", type=Path)
    generate.add_argument("--test", "--test-manifest", dest="test", type=Path)
    generate.add_argument(
        "--label-map",
        type=Path,
        default=None,
        help="JSON class-name-to-id mapping (not needed for numeric labels)",
    )
    generate.add_argument(
        "--relative-paths",
        action="store_true",
        help="store paths relative to --video-root instead of absolute paths",
    )
    generate.add_argument(
        "--allow-missing",
        action="store_true",
        help="permit manifest paths that are not present (only with supplied frame counts)",
    )
    generate.add_argument(
        "--allow-class-overlap",
        action="store_true",
        help="permit the same class id in multiple splits",
    )
    generate.add_argument(
        "--expected-class-counts",
        default=None,
        help="for SSv2-CMN use 64,12,24",
    )

    validate = subparsers.add_parser(
        "validate", help="validate train/val/test annotation files"
    )
    _add_split_inputs(validate)
    validate.add_argument("--path-root", type=Path, default=None)
    validate.add_argument("--skip-path-check", action="store_true")
    validate.add_argument(
        "--verify-frame-counts",
        action="store_true",
        help="decode each video header and compare its actual length to annotation",
    )
    validate.add_argument("--allow-class-overlap", action="store_true")
    validate.add_argument("--expected-class-counts", default=None)
    validate.add_argument("--min-samples-per-class", type=int, default=1)
    validate.add_argument("--output", type=Path, default=None)

    audit = subparsers.add_parser(
        "audit", help="report class counts, frame histogram and random samples"
    )
    _add_split_inputs(audit)
    audit.add_argument("--path-root", type=Path, default=None)
    audit.add_argument("--skip-path-check", action="store_true")
    audit.add_argument("--histogram-bins", type=int, default=10)
    audit.add_argument("--sample-count", type=int, default=3)
    audit.add_argument("--seed", type=int, default=916)
    audit.add_argument("--output", type=Path, default=None)
    audit.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="write 8-frame contact sheets for the random samples",
    )
    return parser


def _print_or_write(payload: Mapping[str, object], output: Optional[Path]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print("wrote {}".format(output))


def _run_generate(args: argparse.Namespace) -> int:
    supplied = {split: getattr(args, split) for split in ("train", "val", "test")}
    present = [path is not None for path in supplied.values()]
    if any(present) and not all(present):
        raise ValueError("provide all of --train, --val and --test, or none to scan split directories")
    manifests = supplied if all(present) else None
    counts = _expected_class_counts(args.expected_class_counts)
    outputs = generate_split_annotations(
        args.video_root,
        args.output_dir,
        split_manifests=manifests,
        label_map=args.label_map,
        absolute_paths=not args.relative_paths,
        require_paths=not args.allow_missing,
        validate=True,
        require_disjoint_labels=not args.allow_class_overlap,
        expected_class_counts=counts,
    )
    payload = {"generated": {split: str(path) for split, path in outputs.items()}}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    paths = _split_paths(args)
    report = validate_split_annotations(
        paths["train"],
        paths["val"],
        paths["test"],
        check_paths=not args.skip_path_check,
        path_root=args.path_root,
        require_disjoint_labels=not args.allow_class_overlap,
        expected_class_counts=_expected_class_counts(args.expected_class_counts),
        min_samples_per_class=args.min_samples_per_class,
        verify_frame_counts=args.verify_frame_counts,
    )
    _print_or_write(report.to_dict(), args.output)
    return 0 if report.valid else 2


def _run_audit(args: argparse.Namespace) -> int:
    paths = _split_paths(args)
    report = audit_annotations(
        paths,
        histogram_bins=args.histogram_bins,
        sample_count=args.sample_count,
        seed=args.seed,
        check_paths=not args.skip_path_check,
        path_root=args.path_root,
    )
    if args.preview_dir is not None:
        report["previews"] = save_audit_previews(
            report, args.preview_dir, path_root=args.path_root
        )
    _print_or_write(report, args.output)
    return 0 if report["valid"] else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            return _run_generate(args)
        if args.command == "validate":
            return _run_validate(args)
        if args.command == "audit":
            return _run_audit(args)
    except (OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
