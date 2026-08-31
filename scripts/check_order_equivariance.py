#!/usr/bin/env python3
"""Run the real-weight O-MSA equivariance gate required by Innovation 3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fsar.config import load_config, resolve_path
from fsar.equivariance import check_text_order_equivariance
from fsar.model import EpisodicTaskAdapter
from fsar.utils import load_corpus_yaml, setup_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/innovation3.yaml")
    parser.add_argument("--checkpoint", required=True, help="CLIP ViT-B/16 base checkpoint")
    parser.add_argument(
        "--weights",
        default=None,
        help="optional trained experiment checkpoint; its text-adapter weights are checked",
    )
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config, overrides={"model": {"checkpoint": args.checkpoint}})
    base = Path(args.config).resolve().parent
    corpus_path = resolve_path(config["data"]["corpus"], base)
    corpus = load_corpus_yaml(corpus_path, require_sub_actions=True)
    if not 1 <= args.classes <= len(corpus):
        raise ValueError("--classes is outside the corpus size")
    device = setup_runtime(config, device=args.device)
    model = EpisodicTaskAdapter(
        args.classes,
        1,
        1,
        corpus,
        visual_depth=int(config["model"].get("adapter_depth", 6)),
        text_depth=int(config["model"].get("text_depth", 2)),
        checkpoint_path=args.checkpoint,
        visual_encoder=torch.nn.Identity(),
    ).to(device)
    if args.weights:
        payload = torch.load(args.weights, map_location=device)
        state = payload.get("state", payload.get("model", payload)) if isinstance(payload, dict) else payload
        text_state = {
            key.removeprefix("text_encoder."): value
            for key, value in state.items()
            if key.startswith("text_encoder.")
        }
        if not text_state:
            raise ValueError("--weights contains no text_encoder.* parameters")
        model.text_encoder.load_state_dict(text_state, strict=True)
    class_indices = torch.arange(args.classes, dtype=torch.long)
    result = check_text_order_equivariance(model, class_indices)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
