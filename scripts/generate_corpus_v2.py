#!/usr/bin/env python
"""Generate or dry-run an Innovation-2 v2 sub-action corpus.

Examples
--------
Inspect every provider call without network access::

    python scripts/generate_corpus_v2.py --classes classes.json \
        --dataset somethingcmn --output generation_jobs.json --dry-run

Generate with an explicitly installed provider plugin::

    python scripts/generate_corpus_v2.py --classes classes.json \
        --dataset somethingcmn --provider my_provider:provider \
        --output corpus/classes_somethingcmn_v2.json

No provider is selected implicitly and this script contains no API client.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fsar.corpus_v2 import (  # noqa: E402
    DEFAULT_PROMPT_TEMPLATE,
    CorpusV2Error,
    generate_corpus_v2,
    generation_manifest,
    load_provider,
    save_corpus_v2,
)


PROVIDER_TEMPLATE = '''"""Provider plugin template for generate_corpus_v2.py.

Install/configure your preferred LLM client in this module.  The generator
loads ``provider`` only when --provider is passed; dry-run never imports it.
"""


class Provider:
    def generate(self, *, prompt, class_name, k, sample_index, seed, temperature):
        # Call your provider here and return either:
        #   ["visible stage 1", ..., "visible stage K"]
        # or a JSON string/object containing {"subs": [...]}.
        raise RuntimeError("configure an LLM client before using this provider")


provider = Provider()
'''


def _load_structured(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            return json.load(handle)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle)
        return [line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")]


def load_class_names(path: Path) -> List[str]:
    """Accept a text list, JSON/YAML list, or class-keyed mapping."""

    value = _load_structured(path)
    if isinstance(value, list):
        names = value
    elif isinstance(value, Mapping):
        if isinstance(value.get("classes"), list):
            names = value["classes"]
        elif isinstance(value.get("classes"), Mapping):
            names = list(value["classes"])
        else:
            names = list(value)
    else:
        raise CorpusV2Error("class file must be a list or class-keyed mapping")
    return [str(name) for name in names]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--classes", type=Path, help="JSON/YAML/TXT class list")
    value.add_argument("--dataset", help="dataset name stored in the corpus")
    value.add_argument("--output", type=Path, help="v2 corpus or dry-run JSON path")
    value.add_argument(
        "--provider",
        help="explicit provider plugin in python.module:object syntax",
    )
    value.add_argument("--samples-per-k", type=int, default=3)
    value.add_argument("--temperature", type=float, default=0.7)
    value.add_argument("--seed", type=int, default=916)
    value.add_argument("--recipe-id", default="default")
    value.add_argument(
        "--template",
        type=Path,
        help="optional UTF-8 prompt template with {class_name} and {k}",
    )
    value.add_argument(
        "--dry-run",
        action="store_true",
        help="write provider jobs only; imports no provider and performs zero network requests",
    )
    value.add_argument(
        "--emit-provider-template",
        type=Path,
        help="write a provider plugin template and exit",
    )
    return value


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.emit_provider_template is not None:
        args.emit_provider_template.parent.mkdir(parents=True, exist_ok=True)
        args.emit_provider_template.write_text(PROVIDER_TEMPLATE, encoding="utf-8")
        print(f"wrote provider template: {args.emit_provider_template}")
        return 0
    missing = [name for name in ("classes", "dataset", "output") if getattr(args, name) is None]
    if missing:
        parser().error("the following arguments are required: " + ", ".join("--" + name for name in missing))
    template = (
        args.template.read_text(encoding="utf-8")
        if args.template is not None
        else DEFAULT_PROMPT_TEMPLATE
    )
    class_names = load_class_names(args.classes)
    if args.dry_run:
        manifest = generation_manifest(
            class_names,
            samples_per_k=args.samples_per_k,
            seed=args.seed,
            template=template,
        )
        manifest.update(
            {
                "dataset": args.dataset,
                "temperature": args.temperature,
                "recipe_id": args.recipe_id,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"dry-run: wrote {len(manifest['jobs'])} jobs; network_requests=0")
        return 0
    if not args.provider:
        parser().error(
            "generation requires --provider python.module:object; use --dry-run "
            "to inspect jobs or --emit-provider-template to scaffold a plugin"
        )
    provider = load_provider(args.provider)
    corpus = generate_corpus_v2(
        class_names,
        provider,
        dataset=args.dataset,
        samples_per_k=args.samples_per_k,
        temperature=args.temperature,
        seed=args.seed,
        recipe_id=args.recipe_id,
        template=template,
    )
    save_corpus_v2(corpus, args.output)
    candidate_count = sum(len(value.candidates) for value in corpus.classes.values())
    print(
        f"wrote {candidate_count} candidates for {len(corpus.classes)} classes: "
        f"{args.output}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CorpusV2Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
