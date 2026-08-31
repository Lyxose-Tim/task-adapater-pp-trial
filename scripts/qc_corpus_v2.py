#!/usr/bin/env python
"""Run all-class text QC and guarded video/recipe QC for a v2 corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fsar.corpus_v2 import (  # noqa: E402
    CorpusV2Error,
    TextQCConfig,
    apply_base_video_qc,
    load_corpus_v2,
    run_text_qc,
    save_corpus_v2,
    select_recipe_on_validation,
)


def load_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle) if path.suffix.lower() == ".json" else yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise CorpusV2Error(f"expected a mapping in {path}")
    return value


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--report", type=Path, required=True)
    value.add_argument("--min-chars", type=int, default=3)
    value.add_argument("--max-chars", type=int, default=160)
    value.add_argument("--redundancy-threshold", type=float, default=0.85)
    value.add_argument("--banned-term", action="append", default=[])
    value.add_argument(
        "--splits",
        type=Path,
        help="class-to-split JSON/YAML; mandatory for any video-derived scores",
    )
    value.add_argument(
        "--base-video-scores",
        type=Path,
        help="class -> gen_id -> OT/video-fit cost; base classes only",
    )
    value.add_argument("--base-top-n", type=int, default=2)
    value.add_argument(
        "--validation-recipe-scores",
        type=Path,
        help="class -> recipe_id -> validation score; validation classes only",
    )
    value.add_argument(
        "--video-score-higher-is-better",
        action="store_true",
        help="base-video scores default to OT costs (lower is better)",
    )
    value.add_argument(
        "--recipe-score-lower-is-better",
        action="store_true",
        help="validation recipe scores default to accuracy (higher is better)",
    )
    return value


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    corpus = load_corpus_v2(args.input)
    defaults = TextQCConfig()
    text_config = TextQCConfig(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        redundancy_threshold=args.redundancy_threshold,
        banned_terms=tuple(defaults.banned_terms) + tuple(args.banned_term),
    )
    filtered, text_report = run_text_qc(corpus, text_config)
    report: Dict[str, Any] = {"text_qc": text_report.to_dict()}

    needs_splits = args.base_video_scores is not None or args.validation_recipe_scores is not None
    if needs_splits and args.splits is None:
        parser().error("--splits is mandatory when video-derived scores are supplied")
    splits = load_mapping(args.splits) if args.splits is not None else {}
    if args.base_video_scores is not None:
        scores = load_mapping(args.base_video_scores)
        filtered, video_report = apply_base_video_qc(
            filtered,
            scores,
            splits,
            top_n=args.base_top_n,
            lower_is_better=not args.video_score_higher_is_better,
        )
        report["base_video_qc"] = video_report.to_dict()
    if args.validation_recipe_scores is not None:
        scores = load_mapping(args.validation_recipe_scores)
        choice = select_recipe_on_validation(
            scores,
            splits,
            lower_is_better=args.recipe_score_lower_is_better,
        )
        report["validation_recipe_selection"] = {
            "scope": "validation_recipe_only",
            "recipe_id": choice.recipe_id,
            "mean_score": choice.mean_score,
            "per_class_scores": choice.per_class_scores,
        }

    # Raw K=2..5 candidates remain serialized; rejected items are auditable via
    # active=false and their QC reasons.
    save_corpus_v2(filtered, args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"text QC checked all {len(text_report.classes_checked)} classes; "
        f"accepted={text_report.accepted}, rejected={text_report.rejected}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CorpusV2Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
