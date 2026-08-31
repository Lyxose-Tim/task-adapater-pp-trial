"""Runtime helpers shared by the standalone few-shot action experiments."""

from __future__ import annotations

import importlib
import os
import random
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np
import torch
import yaml

from .config import ConfigError, resolve_path


PathLike = Union[str, os.PathLike]
Tokenizer = Callable[[Sequence[str]], torch.LongTensor]


class _OrderedSafeLoader(yaml.SafeLoader):
    pass


def _construct_ordered_mapping(
    loader: _OrderedSafeLoader, node: yaml.nodes.MappingNode
) -> OrderedDict:
    loader.flatten_mapping(node)
    result: OrderedDict = OrderedDict()
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=False)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConfigError("corpus YAML contains an unhashable mapping key") from exc
        if duplicate:
            raise ConfigError(f"corpus YAML contains duplicate key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_OrderedSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_ordered_mapping,
)


def load_corpus_yaml(
    path: PathLike,
    *,
    base_dir: Optional[PathLike] = None,
    require_sub_actions: bool = False,
) -> OrderedDict:
    """Read a corpus mapping while preserving its authored class order.

    When ``require_sub_actions`` is true, every class must contain exactly three
    non-empty strings under ``sub_act_en_li``.  The stricter mode is intended for
    Innovation 3; the relaxed default also supports simple label corpora.
    """

    corpus_path = resolve_path(path, base_dir)
    if not corpus_path.is_file():
        raise FileNotFoundError(f"corpus YAML does not exist: {corpus_path}")
    try:
        with corpus_path.open("r", encoding="utf-8") as handle:
            corpus = yaml.load(handle, Loader=_OrderedSafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid corpus YAML in {corpus_path}: {exc}") from exc
    if not isinstance(corpus, Mapping) or not corpus:
        raise ConfigError(f"corpus YAML must contain a non-empty mapping: {corpus_path}")
    if require_sub_actions:
        for class_name, entry in corpus.items():
            if not isinstance(entry, Mapping):
                raise ConfigError(f"corpus entry {class_name!r} must be a mapping")
            stages = entry.get("sub_act_en_li")
            if (
                not isinstance(stages, list)
                or len(stages) != 3
                or any(not isinstance(stage, str) or not stage.strip() for stage in stages)
            ):
                raise ConfigError(
                    f"corpus entry {class_name!r} must have exactly three non-empty "
                    "sub_act_en_li strings"
                )
    return corpus


def cosine_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Return the pairwise cosine matrix in numerically stable float32.

    Inputs must be ``[N, D]`` and ``[M, D]``.  Computation is deliberately
    promoted from fp16/bfloat16 before norms and the matrix product; zero rows
    consequently produce finite all-zero similarities rather than NaNs.
    """

    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("cosine_matrix expects torch.Tensor inputs")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            f"cosine_matrix expects 2-D tensors, got shapes {tuple(x.shape)} and {tuple(y.shape)}"
        )
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"feature dimensions differ: {x.shape[1]} versus {y.shape[1]}"
        )
    if x.device != y.device:
        raise ValueError(f"inputs must share a device, got {x.device} and {y.device}")
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("eps must be positive")

    x32 = x.to(dtype=torch.float32)
    y32 = y.to(dtype=torch.float32)
    x_norm = torch.linalg.vector_norm(x32, dim=-1, keepdim=True).clamp_min(float(eps))
    y_norm = torch.linalg.vector_norm(y32, dim=-1, keepdim=True).clamp_min(float(eps))
    return (x32 / x_norm) @ (y32 / y_norm).transpose(0, 1)


def _as_long_tensor(tokens: Any) -> torch.LongTensor:
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens)
    return tokens.to(dtype=torch.long)


@lru_cache(maxsize=None)
def _resolve_clip_tokenizer(model_name: str) -> Tokenizer:
    """Resolve a tokenizer lazily, preferring the original OpenAI package."""

    clip_error: Optional[BaseException] = None
    try:
        clip_module = importlib.import_module("clip")
        clip_tokenize = getattr(clip_module, "tokenize", None)
        if callable(clip_tokenize):

            def openai_tokenizer(texts: Sequence[str]) -> torch.LongTensor:
                normalized = [str(text) for text in texts]
                try:
                    result = clip_tokenize(normalized, truncate=True)
                except TypeError:
                    result = clip_tokenize(normalized)
                return _as_long_tensor(result)

            return openai_tokenizer
        clip_error = AttributeError(
            "installed 'clip' module has no tokenize(); it may be the unrelated PyPI package"
        )
    except (ImportError, ModuleNotFoundError) as exc:
        clip_error = exc

    try:
        open_clip = importlib.import_module("open_clip")
        open_clip_name = model_name.replace("/", "-")
        tokenizer = open_clip.get_tokenizer(open_clip_name)

        def open_clip_tokenizer(texts: Sequence[str]) -> torch.LongTensor:
            return _as_long_tensor(tokenizer([str(text) for text in texts]))

        return open_clip_tokenizer
    except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "No compatible CLIP tokenizer is available. Install OpenAI CLIP "
            "(git+https://github.com/openai/CLIP.git) or open-clip-torch. "
            f"OpenAI CLIP resolution failed with: {clip_error!r}; "
            f"open_clip resolution failed with: {exc!r}"
        ) from exc


def get_clip_tokenizer(model_name: str = "ViT-B/16") -> Tokenizer:
    """Return ``callable(texts) -> LongTensor`` without eager CLIP imports."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    return _resolve_clip_tokenizer(model_name.strip())


def clear_clip_tokenizer_cache() -> None:
    """Clear tokenizer resolution state (primarily useful for isolated tests)."""

    _resolve_clip_tokenizer.cache_clear()


def resolve_checkpoint(
    checkpoint: Optional[Union[PathLike, Mapping[str, Any]]] = None,
    *,
    config: Optional[Mapping[str, Any]] = None,
    config_value: Optional[PathLike] = None,
    env_var: str = "CLIP_CHECKPOINT",
    base_dir: Optional[PathLike] = None,
    required: bool = True,
    must_exist: bool = True,
) -> Optional[Path]:
    """Resolve a checkpoint with CLI argument > environment > config precedence.

    For convenience a full configuration mapping may be supplied either through
    ``config=`` or as the first positional value.  ``model.checkpoint_env`` is
    honoured when present.
    """

    if isinstance(checkpoint, Mapping):
        if config is not None:
            raise TypeError("configuration was supplied twice")
        config = checkpoint
        checkpoint = None

    configured = config_value
    if config is not None:
        model = config.get("model", config)
        if not isinstance(model, Mapping):
            raise ConfigError("config.model must be a mapping")
        if configured is None:
            configured = model.get("checkpoint")
        env_var = str(model.get("checkpoint_env") or env_var)

    candidate = checkpoint or os.environ.get(env_var) or configured
    if candidate is None or not str(candidate).strip():
        if required:
            raise FileNotFoundError(
                "CLIP checkpoint was not provided; use --checkpoint, set "
                f"{env_var}, or configure model.checkpoint"
            )
        return None
    resolved = resolve_path(candidate, base_dir)
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {resolved}")
    return resolved


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch CPU/CUDA RNGs."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if deterministic:
        # Required by CUDA >= 10.2 for deterministic cuBLAS kernels.  Set this
        # before the first CUDA matrix multiplication in each CLI process.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    if hasattr(torch, "use_deterministic_algorithms"):
        # warn_only avoids turning unsupported third-party kernels into an
        # unrelated hard failure while still selecting deterministic kernels.
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def select_device(device: Optional[Union[str, int, torch.device]] = "auto") -> torch.device:
    """Choose an available device and fail clearly for impossible requests."""

    if isinstance(device, int) and not isinstance(device, bool):
        device = f"cuda:{device}"
    if device is None or str(device).casefold() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {selected}")
        index = 0 if selected.index is None else selected.index
        count = torch.cuda.device_count()
        if index < 0 or index >= count:
            raise RuntimeError(
                f"CUDA device index {index} is unavailable; detected {count} device(s)"
            )
        selected = torch.device(f"cuda:{index}")
    return selected


def setup_runtime(
    runtime: Optional[Union[Mapping[str, Any], str, int, torch.device]] = None,
    *,
    seed: Optional[int] = None,
    device: Optional[Union[str, int, torch.device]] = None,
    deterministic: Optional[bool] = None,
    allow_tf32: Optional[bool] = None,
) -> torch.device:
    """Configure reproducibility and return the selected torch device.

    ``runtime`` may be the full loaded config, its ``runtime`` section, or a
    device specifier.  Explicit keyword arguments take precedence.
    """

    runtime_cfg: Mapping[str, Any] = {}
    if isinstance(runtime, Mapping):
        candidate = runtime.get("runtime", runtime)
        if not isinstance(candidate, Mapping):
            raise ConfigError("config.runtime must be a mapping")
        runtime_cfg = candidate
    elif runtime is not None:
        device = runtime if device is None else device

    resolved_seed = runtime_cfg.get("seed", 916) if seed is None else seed
    resolved_device = runtime_cfg.get("device", "auto") if device is None else device
    resolved_deterministic = (
        runtime_cfg.get("deterministic", True)
        if deterministic is None
        else deterministic
    )
    resolved_allow_tf32 = (
        runtime_cfg.get("allow_tf32", False)
        if allow_tf32 is None
        else allow_tf32
    )
    set_seed(int(resolved_seed), bool(resolved_deterministic))
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(resolved_allow_tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(resolved_allow_tf32)
    return select_device(resolved_device)


def require_data_paths(
    config: Optional[Mapping[str, Any]] = None,
    *,
    data_root: Optional[PathLike] = None,
    corpus: Optional[PathLike] = None,
    splits: Optional[Mapping[str, PathLike]] = None,
    base_dir: Optional[PathLike] = None,
    require_sub_actions: bool = True,
) -> Dict[str, Path]:
    """Fail fast unless the dataset, corpus, and independent splits are usable."""

    if config is not None:
        data = config.get("data", config)
        if not isinstance(data, Mapping):
            raise ConfigError("config.data must be a mapping")
        data_root = data_root or data.get("root")
        corpus = corpus or data.get("corpus")
        splits = splits or data.get("splits")
    if data_root is None:
        raise ConfigError("dataset root is not configured")
    if corpus is None:
        raise ConfigError("corpus YAML is not configured")
    if not isinstance(splits, Mapping):
        raise ConfigError("train/val/test split paths are not configured")

    root_path = resolve_path(data_root, base_dir)
    if not root_path.is_dir():
        raise FileNotFoundError(f"dataset root does not exist or is not a directory: {root_path}")
    corpus_path = resolve_path(corpus, base_dir)
    # Parsing here catches malformed or incomplete corpora before any GPU work.
    load_corpus_yaml(corpus_path, require_sub_actions=require_sub_actions)

    result: Dict[str, Path] = {"root": root_path, "corpus": corpus_path}
    resolved_splits = []
    for split_name in ("train", "val", "test"):
        split_value = splits.get(split_name)
        if split_value is None or not str(split_value).strip():
            raise ConfigError(f"{split_name} annotation is not configured")
        split_path = resolve_path(split_value, base_dir)
        if not split_path.is_file():
            raise FileNotFoundError(f"{split_name} annotation does not exist: {split_path}")
        if split_path.stat().st_size == 0:
            raise ConfigError(f"{split_name} annotation is empty: {split_path}")
        result[split_name] = split_path
        resolved_splits.append(os.path.normcase(str(split_path)))
    if len(set(resolved_splits)) != 3:
        raise ConfigError("train, val, and test annotations must be independent files")
    return result


# The original repository calls this operation ``cosine_similarity``.  Retain
# the name as an alias while making the matrix orientation explicit above.
cosine_similarity = cosine_matrix
setup_seed = set_seed
get_device = select_device


__all__ = [
    "clear_clip_tokenizer_cache",
    "cosine_matrix",
    "cosine_similarity",
    "get_clip_tokenizer",
    "get_device",
    "load_corpus_yaml",
    "require_data_paths",
    "resolve_checkpoint",
    "select_device",
    "set_seed",
    "setup_runtime",
    "setup_seed",
]
