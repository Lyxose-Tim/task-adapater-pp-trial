from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

from fsar.config import ConfigError, load_config, validate_config
from fsar.utils import (
    clear_clip_tokenizer_cache,
    cosine_matrix,
    get_clip_tokenizer,
    load_corpus_yaml,
    require_data_paths,
    resolve_checkpoint,
    setup_runtime,
)


def _write(path: Path, content: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_load_config_merges_defaults_and_yaml(tmp_path: Path) -> None:
    base = _write(
        tmp_path / "base.yaml",
        "runtime:\n  num_workers: 3\nregularizer_3b:\n  margin: 0.25\n",
    )
    child = _write(
        tmp_path / "child.yaml",
        f"defaults: {base.name}\nruntime:\n  seed: 7\n",
    )

    cfg = load_config(child, overrides={"runtime": {"num_workers": 2}})

    assert cfg["runtime"]["seed"] == 7
    assert cfg["runtime"]["num_workers"] == 2
    assert cfg["runtime"]["num_gpus"] == 1
    assert cfg["runtime"]["gpu_ids"] == [0]
    assert cfg["runtime"]["amp"] is False
    assert cfg["runtime"]["allow_tf32"] is False
    assert cfg["model"]["backbone"] == "ViT-B-16"
    assert cfg["regularizer_3b"]["margin"] == 0.25
    assert cfg["data"]["splits"]["train"] != cfg["data"]["splits"]["val"]


def test_config_rejects_shared_train_val_test_annotation() -> None:
    cfg = load_config()
    cfg["data"]["splits"] = {"train": "same.txt", "val": "same.txt", "test": "same.txt"}

    with pytest.raises(ConfigError, match="independent"):
        validate_config(cfg)


@pytest.mark.parametrize("field", ("amp", "allow_tf32", "deterministic"))
def test_config_rejects_non_boolean_runtime_switches(field: str) -> None:
    cfg = load_config()
    cfg["runtime"][field] = 1

    with pytest.raises(ConfigError, match=f"runtime.{field} must be a boolean"):
        validate_config(cfg)


def test_load_config_optional_path_validation_fails_early(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "missing-data.yaml",
        "data:\n  root: absent\n  corpus: absent.yml\n",
    )

    # Pure loading remains possible while a dataset is being prepared.
    load_config(cfg_path, validate_paths=False)
    with pytest.raises(FileNotFoundError, match="dataset root"):
        load_config(cfg_path, validate_paths=True)


def test_corpus_yaml_preserves_order_and_validates_three_stages(tmp_path: Path) -> None:
    corpus_path = _write(
        tmp_path / "classes.yml",
        """
third:
  label: Third
  sub_act_en_li: [start, continue, finish]
first:
  label: First
  sub_act_en_li: [open, act, close]
second:
  label: Second
  sub_act_en_li: [before, during, after]
""".lstrip(),
    )

    corpus = load_corpus_yaml(corpus_path, require_sub_actions=True)

    assert list(corpus) == ["third", "first", "second"]
    assert corpus["first"]["sub_act_en_li"] == ["open", "act", "close"]


def test_corpus_yaml_rejects_duplicate_keys(tmp_path: Path) -> None:
    corpus_path = _write(tmp_path / "duplicate.yml", "class_a: 1\nclass_a: 2\n")

    with pytest.raises(ConfigError, match="duplicate"):
        load_corpus_yaml(corpus_path)


def test_cosine_matrix_promotes_fp16_and_handles_zero_rows() -> None:
    x = torch.tensor([[10000.0, 10000.0], [0.0, 0.0]], dtype=torch.float16)
    y = torch.tensor([[10000.0, 10000.0], [10000.0, -10000.0]], dtype=torch.float16)

    similarity = cosine_matrix(x, y)

    assert similarity.dtype == torch.float32
    assert torch.isfinite(similarity).all()
    torch.testing.assert_close(
        similarity,
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )


def test_clip_tokenizer_is_lazy_and_prefers_openai_clip(monkeypatch) -> None:
    calls = []
    fake_clip = types.SimpleNamespace(
        tokenize=lambda texts, truncate=True: calls.append((list(texts), truncate))
        or torch.ones(len(texts), 4, dtype=torch.int32)
    )
    fake_open_clip = types.SimpleNamespace(
        get_tokenizer=lambda _name: pytest.fail("open_clip fallback must not be used")
    )
    clear_clip_tokenizer_cache()
    monkeypatch.setitem(sys.modules, "clip", fake_clip)
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    tokenizer = get_clip_tokenizer("ViT-B/16")
    tokens = tokenizer(["one", "two"])

    assert calls == [(["one", "two"], True)]
    assert tokens.dtype == torch.long
    assert tokens.shape == (2, 4)
    clear_clip_tokenizer_cache()


def test_clip_tokenizer_falls_back_to_open_clip(monkeypatch) -> None:
    requested = []
    fake_clip = types.SimpleNamespace()  # the unrelated PyPI package
    fake_open_clip = types.SimpleNamespace(
        get_tokenizer=lambda name: requested.append(name)
        or (lambda texts: torch.zeros(len(texts), 3, dtype=torch.int16))
    )
    clear_clip_tokenizer_cache()
    monkeypatch.setitem(sys.modules, "clip", fake_clip)
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    tokens = get_clip_tokenizer("ViT-B/16")(["action"])

    assert requested == ["ViT-B-16"]
    assert tokens.dtype == torch.long
    assert tokens.shape == (1, 3)
    clear_clip_tokenizer_cache()


def test_resolve_checkpoint_precedence_and_missing(monkeypatch, tmp_path: Path) -> None:
    cli = _write(tmp_path / "cli.pt")
    env = _write(tmp_path / "env.pt")
    configured = _write(tmp_path / "configured.pt")
    monkeypatch.setenv("TEST_CLIP_CHECKPOINT", str(env))
    config = {
        "model": {
            "checkpoint": str(configured),
            "checkpoint_env": "TEST_CLIP_CHECKPOINT",
        }
    }

    assert resolve_checkpoint(cli, config=config) == cli.resolve()
    assert resolve_checkpoint(config=config) == env.resolve()
    monkeypatch.delenv("TEST_CLIP_CHECKPOINT")
    assert resolve_checkpoint(config=config) == configured.resolve()
    assert resolve_checkpoint(None, required=False, env_var="UNSET_FOR_TEST") is None
    with pytest.raises(FileNotFoundError, match="not provided"):
        resolve_checkpoint(None, env_var="UNSET_FOR_TEST")


def test_setup_runtime_seeds_all_cpu_rngs() -> None:
    setup_runtime({"seed": 123, "device": "cpu", "deterministic": True})
    first = torch.rand(3)
    device = setup_runtime({"seed": 123, "device": "cpu", "deterministic": True})
    second = torch.rand(3)

    assert device == torch.device("cpu")
    torch.testing.assert_close(first, second)


def test_setup_runtime_applies_tf32_without_disabling_deterministic_algorithms() -> None:
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        setup_runtime(
            {
                "seed": 123,
                "device": "cpu",
                "deterministic": True,
                "allow_tf32": True,
            }
        )
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


def test_require_data_paths_returns_independent_inputs(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    corpus = _write(
        tmp_path / "corpus.yml",
        "class_a:\n  label: A\n  sub_act_en_li: [one, two, three]\n",
    )
    splits = {
        name: _write(tmp_path / "annotations" / f"{name}.txt", f"video 8 0 # {name}\n")
        for name in ("train", "val", "test")
    }

    resolved = require_data_paths(
        data_root=root,
        corpus=corpus,
        splits=splits,
        require_sub_actions=True,
    )

    assert resolved["root"] == root.resolve()
    assert resolved["corpus"] == corpus.resolve()
    assert len({resolved["train"], resolved["val"], resolved["test"]}) == 3


def test_require_data_paths_fails_before_gpu_work(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dataset root"):
        require_data_paths(
            data_root=tmp_path / "absent",
            corpus=tmp_path / "absent.yml",
            splits={"train": "x", "val": "y", "test": "z"},
        )
