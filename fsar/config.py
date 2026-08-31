"""Configuration loading and validation for the standalone FSAR experiments.

The original Task-Adapter++ entry point relies on a mutable global YAML file.
Innovation 3 runs several paired conditions from the same episode stream, so a
small, side-effect-free configuration layer is useful.  Loading a configuration
does not require the dataset to be present; callers opt into filesystem checks
at the point where they are about to train or evaluate.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Union

import yaml


PathLike = Union[str, os.PathLike]


class ConfigError(ValueError):
    """Raised when an FSAR configuration is structurally invalid."""


DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {"name": "innovation3", "phase": "m0"},
    "runtime": {
        "seed": 916,
        "device": "auto",
        "num_gpus": 1,
        "gpu_ids": [0],
        # Windows uses the ``spawn`` multiprocessing start method.  Four
        # workers is a safer default for direct VP9 video decoding; individual
        # experiment configs may raise this after their smoke run succeeds.
        "num_workers": 4,
        "deterministic": True,
        # AMP and TF32 are opt-in at the library default.  CUDA experiment
        # configs enable them explicitly; CPU callers retain exact FP32 paths.
        "amp": False,
        "allow_tf32": False,
    },
    "model": {
        "method": "taskadapter",
        "backbone": "ViT-B-16",
        "openai_clip_name": "ViT-B/16",
        "pretrained": "openai",
        "checkpoint": None,
        "checkpoint_env": "CLIP_CHECKPOINT",
        "num_frames": 8,
        "semantic_backend": "fixed_window",
        "ot_frame_source": "aligned",
        "ot": {
            "epsilon": 0.05,
            "lambda_pos": 0.0,
            "rho": float("inf"),
            "iterations": 30,
            "mass_epsilon": 1.0e-6,
        },
        "fusion": {
            "mode": "legacy_product",
            "alpha": 0.5,
            "visual_temperature": 1.0,
            "semantic_temperature": 1.0,
        },
    },
    "data": {
        "name": "somethingcmn",
        "root": "dataset/smsm_cmn",
        "corpus": "corpus/classes_somethingcmn.yml",
        "splits": {
            "train": "dataset/smsm_cmn/annotations/train.txt",
            "val": "dataset/smsm_cmn/annotations/val.txt",
            "test": "dataset/smsm_cmn/annotations/test.txt",
        },
        "require_independent_splits": True,
    },
    "episodes": {
        "train_n_way": 5,
        "test_n_way": 5,
        "n_shot": 1,
        "n_query": 1,
        "train": 1000,
        "val": 1000,
        "test": 10000,
    },
    "m0": {
        "enabled": True,
        "smoke_train_episodes": 2,
        "smoke_val_episodes": 2,
        "smoke_test_episodes": 2,
        "data_audit": True,
    },
    "diagnosis_3a": {
        "enabled": False,
        "conditions": ["C0", "C1", "C2", "C3", "C4"],
        "include_c5": False,
        "permutation_seed": 916,
        "paired_episodes": True,
        "sync_query_frame_permutation": True,
        "report_branch_metrics": True,
        "confidence": 0.95,
    },
    "regularizer_3b": {
        "enabled": False,
        "kind": "margin",
        "weight": 0.1,
        "margin": 0.1,
        "negative_permutations": 1,
        "reuse_equivariant_features": True,
    },
    "corpus_v2": {
        "enabled": False,
        "path": "corpus/classes_somethingcmn_v2.json",
        "online_support_selection": True,
        "training_selection": True,
        "tie_break": "smaller_k",
        "candidate_k": [2, 3, 4, 5],
    },
    "pseudo_validation": {
        "enabled": False,
        "views": 4,
        "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "global_alpha": 0.5,
        "flat_tolerance": 1.0e-12,
        "horizontal_flip": False,
    },
    "output": {
        "root": "outputs/innovation3",
        "run_metrics": "outputs/innovation3/run_metrics.csv",
        "save_config_snapshot": True,
        "save_episode_records": True,
        "save_checkpoint": True,
    },
}


def deep_merge(
    base: Mapping[str, Any], override: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    result: Dict[str, Any] = copy.deepcopy(dict(base))
    if override is None:
        return result
    if not isinstance(override, Mapping):
        raise ConfigError("configuration overrides must be a mapping")
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return dict(loaded)


def _read_with_defaults(path: Path, seen: Optional[set] = None) -> Dict[str, Any]:
    """Read a YAML mapping and its optional ``defaults`` include(s)."""

    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ConfigError(f"cyclic configuration defaults include: {path}")
    seen.add(path)
    current = _read_yaml_mapping(path)
    includes = current.pop("defaults", None)
    merged: Dict[str, Any] = {}
    if includes is not None:
        include_list: Sequence[Any]
        if isinstance(includes, (str, os.PathLike)):
            include_list = [includes]
        elif isinstance(includes, Sequence) and not isinstance(includes, (bytes, str)):
            include_list = includes
        else:
            raise ConfigError("'defaults' must be a path or a list of paths")
        for include in include_list:
            include_path = Path(os.path.expandvars(os.path.expanduser(str(include))))
            if not include_path.is_absolute():
                include_path = path.parent / include_path
            merged = deep_merge(merged, _read_with_defaults(include_path, seen))
    seen.remove(path)
    return deep_merge(merged, current)


def resolve_path(path: PathLike, base_dir: Optional[PathLike] = None) -> Path:
    """Expand environment/user markers and resolve a path against ``base_dir``."""

    if path is None or not str(path).strip():
        raise ConfigError("path must be a non-empty string")
    expanded = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if not expanded.is_absolute():
        base = Path(base_dir) if base_dir is not None else Path.cwd()
        expanded = base / expanded
    return expanded.resolve()


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        relation = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{field} must be a {relation} integer, got {value!r}")


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"'{name}' must be a mapping")
    return value


def validate_config(
    config: Mapping[str, Any],
    *,
    base_dir: Optional[PathLike] = None,
    validate_paths: bool = False,
    require_checkpoint: bool = False,
) -> Dict[str, Any]:
    """Validate invariant fields and optionally all runtime input paths.

    A deep copy is returned so callers can safely retain the validated result.
    Relative input paths are interpreted from ``base_dir`` (the YAML directory
    when invoked through :func:`load_config`).
    """

    if not isinstance(config, Mapping):
        raise ConfigError("configuration must be a mapping")
    cfg = copy.deepcopy(dict(config))
    runtime = _section(cfg, "runtime")
    model = _section(cfg, "model")
    data = _section(cfg, "data")
    episodes = _section(cfg, "episodes")
    diagnosis = _section(cfg, "diagnosis_3a")
    regularizer = _section(cfg, "regularizer_3b")
    output = _section(cfg, "output")
    corpus_v2 = _section(cfg, "corpus_v2")
    pseudo_validation = _section(cfg, "pseudo_validation")

    for field in ("seed", "num_gpus", "num_workers"):
        _positive_int(runtime.get(field), f"runtime.{field}", allow_zero=field != "num_gpus")
    for field in ("deterministic", "amp", "allow_tf32"):
        if not isinstance(runtime.get(field), bool):
            raise ConfigError(f"runtime.{field} must be a boolean")
    gpu_ids = runtime.get("gpu_ids")
    if not isinstance(gpu_ids, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in gpu_ids
    ):
        raise ConfigError("runtime.gpu_ids must be a list of non-negative integers")
    if runtime["num_gpus"] != len(gpu_ids):
        raise ConfigError("runtime.num_gpus must equal len(runtime.gpu_ids)")

    if model.get("method") != "taskadapter":
        raise ConfigError("model.method must be 'taskadapter'")
    if not isinstance(model.get("backbone"), str) or not model["backbone"].strip():
        raise ConfigError("model.backbone must be a non-empty string")
    _positive_int(model.get("num_frames"), "model.num_frames")
    if model.get("semantic_backend", "fixed_window") not in {"fixed_window", "ot"}:
        raise ConfigError("model.semantic_backend must be 'fixed_window' or 'ot'")
    if model.get("ot_frame_source", "aligned") not in {"aligned", "raw"}:
        raise ConfigError("model.ot_frame_source must be 'aligned' or 'raw'")
    ot = model.get("ot")
    if not isinstance(ot, Mapping):
        raise ConfigError("model.ot must be a mapping")
    for field in ("epsilon", "rho", "lambda_pos", "mass_epsilon"):
        value = ot.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigError(f"model.ot.{field} must be numeric")
    if float(ot["epsilon"]) <= 0 or float(ot["mass_epsilon"]) <= 0:
        raise ConfigError("model.ot epsilon and mass_epsilon must be positive")
    if float(ot["lambda_pos"]) < 0 or float(ot["rho"]) <= 0:
        raise ConfigError("model.ot lambda_pos must be non-negative and rho positive")
    _positive_int(ot.get("iterations"), "model.ot.iterations")
    fusion = model.get("fusion")
    if not isinstance(fusion, Mapping):
        raise ConfigError("model.fusion must be a mapping")
    if fusion.get("mode") not in {"legacy_product", "probability_product", "logit_convex"}:
        raise ConfigError("model.fusion.mode is invalid")
    alpha = fusion.get("alpha")
    if not isinstance(alpha, (int, float)) or not 0 <= float(alpha) <= 1:
        raise ConfigError("model.fusion.alpha must be in [0,1]")
    for field in ("visual_temperature", "semantic_temperature"):
        value = fusion.get(field)
        if not isinstance(value, (int, float)) or float(value) <= 0:
            raise ConfigError(f"model.fusion.{field} must be positive")

    splits = data.get("splits")
    if not isinstance(splits, Mapping):
        raise ConfigError("data.splits must be a mapping")
    missing_splits = [name for name in ("train", "val", "test") if not splits.get(name)]
    if missing_splits:
        raise ConfigError(f"data.splits is missing: {', '.join(missing_splits)}")
    if data.get("require_independent_splits", True):
        split_paths = [str(splits[name]).casefold() for name in ("train", "val", "test")]
        if len(set(split_paths)) != 3:
            raise ConfigError("train, val, and test annotations must be independent files")

    for field in ("train_n_way", "test_n_way", "n_shot", "n_query", "train", "val", "test"):
        _positive_int(episodes.get(field), f"episodes.{field}")

    conditions = diagnosis.get("conditions")
    if not isinstance(conditions, list) or "C0" not in conditions:
        raise ConfigError("diagnosis_3a.conditions must be a list containing C0")
    confidence = diagnosis.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 < float(confidence) < 1.0:
        raise ConfigError("diagnosis_3a.confidence must be between 0 and 1")

    if regularizer.get("kind") not in {"margin", "infonce"}:
        raise ConfigError("regularizer_3b.kind must be 'margin' or 'infonce'")
    for field in ("weight", "margin"):
        value = regularizer.get(field)
        if not isinstance(value, (int, float)) or float(value) < 0:
            raise ConfigError(f"regularizer_3b.{field} must be non-negative")
    _positive_int(
        regularizer.get("negative_permutations"),
        "regularizer_3b.negative_permutations",
    )

    if not output.get("root") or not output.get("run_metrics"):
        raise ConfigError("output.root and output.run_metrics must be configured")

    if not isinstance(corpus_v2.get("enabled"), bool):
        raise ConfigError("corpus_v2.enabled must be boolean")
    if corpus_v2.get("enabled") and not corpus_v2.get("path"):
        raise ConfigError("corpus_v2.path is required when enabled")
    if corpus_v2.get("tie_break") != "smaller_k":
        raise ConfigError("corpus_v2.tie_break must be 'smaller_k'")
    candidate_k = corpus_v2.get("candidate_k")
    if not isinstance(candidate_k, list) or sorted(candidate_k) != [2, 3, 4, 5]:
        raise ConfigError("corpus_v2.candidate_k must cover [2,3,4,5]")

    if not isinstance(pseudo_validation.get("enabled"), bool):
        raise ConfigError("pseudo_validation.enabled must be boolean")
    _positive_int(pseudo_validation.get("views"), "pseudo_validation.views")
    grid = pseudo_validation.get("alpha_grid")
    if (
        not isinstance(grid, list)
        or not grid
        or any(not isinstance(value, (int, float)) or not 0 <= float(value) <= 1 for value in grid)
    ):
        raise ConfigError("pseudo_validation.alpha_grid must be a non-empty [0,1] list")
    global_alpha = pseudo_validation.get("global_alpha")
    if not isinstance(global_alpha, (int, float)) or not 0 <= float(global_alpha) <= 1:
        raise ConfigError("pseudo_validation.global_alpha must be in [0,1]")

    if validate_paths:
        root = resolve_path(data.get("root"), base_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist or is not a directory: {root}")
        corpus = resolve_path(data.get("corpus"), base_dir)
        if not corpus.is_file():
            raise FileNotFoundError(f"corpus YAML does not exist: {corpus}")
        if corpus_v2.get("enabled"):
            v2_path = resolve_path(corpus_v2.get("path"), base_dir)
            if not v2_path.is_file():
                raise FileNotFoundError(f"v2 corpus does not exist: {v2_path}")
        for split_name in ("train", "val", "test"):
            split = resolve_path(splits[split_name], base_dir)
            if not split.is_file():
                raise FileNotFoundError(
                    f"{split_name} annotation does not exist: {split}"
                )
        checkpoint = model.get("checkpoint")
        checkpoint_env = str(model.get("checkpoint_env") or "CLIP_CHECKPOINT")
        checkpoint = checkpoint or os.environ.get(checkpoint_env)
        if require_checkpoint and not checkpoint:
            raise FileNotFoundError(
                "CLIP checkpoint is required; set model.checkpoint or "
                f"environment variable {checkpoint_env}"
            )
        if checkpoint:
            checkpoint_path = resolve_path(checkpoint, base_dir)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"CLIP checkpoint does not exist: {checkpoint_path}")
    return cfg


def load_config(
    path: Optional[PathLike] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    validate_paths: bool = False,
    require_checkpoint: bool = False,
) -> Dict[str, Any]:
    """Load an FSAR YAML file on top of the innovation-3 defaults.

    ``defaults`` in a YAML file may name one path or a list of paths, resolved
    relative to that YAML file.  ``overrides`` has the highest precedence.
    """

    base_dir: Path = Path.cwd()
    loaded: Mapping[str, Any] = {}
    if path is not None:
        config_path = resolve_path(path)
        base_dir = config_path.parent
        loaded = _read_with_defaults(config_path)
    merged = deep_merge(DEFAULT_CONFIG, loaded)
    merged = deep_merge(merged, overrides)
    return validate_config(
        merged,
        base_dir=base_dir,
        validate_paths=validate_paths,
        require_checkpoint=require_checkpoint,
    )


# Compatibility names kept intentionally small for scripts that use generic
# configuration terminology.
load_yaml_config = load_config
merge_config = deep_merge


__all__ = [
    "ConfigError",
    "DEFAULT_CONFIG",
    "deep_merge",
    "load_config",
    "load_yaml_config",
    "merge_config",
    "resolve_path",
    "validate_config",
]
