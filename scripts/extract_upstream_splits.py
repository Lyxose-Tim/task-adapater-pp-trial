#!/usr/bin/env python3
"""Extract canonical FSAR class order from the official Task-Adapter utils.py."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


COUNTS = {
    "hmdb51": {"train": 31, "val": 10, "test": 10},
    "ucf101": {"train": 70, "val": 10, "test": 21},
    "somethingcmn": {"train": 64, "val": 12, "test": 24},
    "kinetics": {"train": 64, "val": 12, "test": 24},
}
VARIABLES = {
    "hmdb51": "hmdb_cls",
    "ucf101": "ucf_cls",
    "somethingcmn": "smsm_cls",
    "kinetics": "kinetics_cls",
}


def extract(source: Path):
    tree = ast.parse(source.read_text(encoding="utf-8"))
    found = {}
    wanted = set(VARIABLES.values())
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted or target.id in found:
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{target.id} is not a static string list")
        found[target.id] = value
    missing = wanted - set(found)
    if missing:
        raise KeyError(f"missing class lists: {sorted(missing)}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?", default=Path("data/fsar_splits"))
    args = parser.parse_args()
    found = extract(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    for dataset, variable in VARIABLES.items():
        payload = {
            "dataset": dataset,
            "source": "https://github.com/bedman367/Task-Adapter",
            "classes": found[variable],
            "counts": COUNTS[dataset],
        }
        path = args.output / f"{dataset}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{path}: {len(payload['classes'])} classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

