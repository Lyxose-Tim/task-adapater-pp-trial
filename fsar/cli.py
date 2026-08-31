"""Command-line orchestration for the canonical episodic FSAR experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import yaml

from fsar.config import load_config, resolve_path
from fsar.corpus_v2 import load_corpus_v2
from fsar.data import DirectVideoDataset, parse_temporal_window
from fsar.episodes import build_episode_loader
from fsar.experiment import (
    calibrate_global_alpha,
    diagnose_order,
    evaluate,
    evaluate_adaptive_fusion,
    create_grad_scaler,
    cuda_amp_enabled,
    save_diagnostics,
    train_one_epoch,
)
from fsar.model import EpisodicTaskAdapter
from fsar.utils import load_corpus_yaml, require_data_paths, setup_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task-Adapter++ episodic FSAR innovations")
    parser.add_argument(
        "command",
        choices=(
            "validate",
            "smoke",
            "train",
            "evaluate",
            "calibrate-fusion",
            "evaluate-adaptive",
            "diagnose",
        ),
    )
    parser.add_argument("--config", default="configs/innovation3.yaml")
    parser.add_argument("--checkpoint", default=None, help="OpenAI CLIP ViT-B/16 JIT checkpoint")
    parser.add_argument("--weights", default=None, help="Trained adapter checkpoint")
    parser.add_argument(
        "--resume",
        default=None,
        help="resume train/smoke from an epoch checkpoint including optimizer/scaler state",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable CUDA float16 autocast (overrides runtime.amp)",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable CUDA matmul/cuDNN TF32 (overrides runtime.allow_tf32)",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override eval/diagnostic episodes")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--temporal-window",
        type=parse_temporal_window,
        default=None,
        help="diagnostic sampling interval, e.g. 0.25,1 or 0,0.75",
    )
    parser.add_argument(
        "--save-transport",
        action="store_true",
        help="save OT plan PNG/NumPy artifacts under the output directory",
    )
    return parser


def _config_base(path: str | Path) -> Path:
    return Path(path).expanduser().resolve().parent


def _make_dataset(
    annotation: Path,
    data_root: Path,
    config: Mapping[str, Any],
    *,
    train: bool,
    temporal_window=None,
) -> DirectVideoDataset:
    return DirectVideoDataset(
        annotation,
        num_frames=int(config["model"]["num_frames"]),
        is_train=train,
        dataset_name=str(config["data"]["name"]),
        path_root=data_root,
        temporal_window=temporal_window,
        seed=int(config["runtime"]["seed"]),
    )


def _make_loader(
    dataset: DirectVideoDataset,
    config: Mapping[str, Any],
    *,
    train: bool,
    episodes: int,
    seed_offset: int,
):
    episode_cfg = config["episodes"]
    return build_episode_loader(
        dataset,
        n_way=int(episode_cfg["train_n_way"] if train else episode_cfg["test_n_way"]),
        n_support=int(episode_cfg["n_shot"]),
        n_query=int(episode_cfg["n_query"]),
        episodes=int(episodes),
        num_workers=int(config["runtime"]["num_workers"]),
        seed=int(config["runtime"]["seed"]) + seed_offset,
    )


def _load_weights(model: torch.nn.Module, path: Optional[str], device: torch.device) -> None:
    if not path:
        return
    payload = torch.load(path, map_location=device)
    state = payload.get("state", payload.get("model", payload)) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)


def _trainable_parameters(model: torch.nn.Module):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("model exposes no trainable parameters")
    return parameters


def _promote_trainable_parameters_to_fp32(model: torch.nn.Module) -> int:
    """Keep optimizer-owned parameters in fp32 for CUDA GradScaler.

    The released CLIP text builder stores both its frozen backbone and its
    trainable adapters in fp16.  Autocast is allowed to choose fp16 kernels,
    but GradScaler requires the master parameters and gradients to be fp32.
    Frozen CLIP weights remain untouched to avoid unnecessary GPU memory use.
    """

    promoted = 0
    for parameter in model.parameters():
        if (
            parameter.requires_grad
            and parameter.is_floating_point()
            and parameter.dtype != torch.float32
        ):
            parameter.data = parameter.data.float()
            promoted += 1
    return promoted


def _append_run_metrics(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _snapshot_config(config: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, allow_unicode=True, sort_keys=False)


def _build_model(config: Mapping[str, Any], corpus, checkpoint: Optional[str]):
    episodes = config["episodes"]
    model_cfg = config["model"]
    fusion_cfg = model_cfg.get("fusion", {})
    return EpisodicTaskAdapter(
        int(episodes["train_n_way"]),
        int(episodes["n_shot"]),
        int(episodes["n_query"]),
        corpus,
        visual_depth=int(model_cfg.get("adapter_depth", 6)),
        text_depth=int(model_cfg.get("text_depth", 2)),
        num_frames=int(model_cfg["num_frames"]),
        checkpoint_path=checkpoint,
        cross_attention_residual=str(model_cfg.get("cross_attention_residual", "current")),
        semantic_backend=str(model_cfg.get("semantic_backend", "fixed_window")),
        ot_options=model_cfg.get("ot"),
        ot_frame_source=str(model_cfg.get("ot_frame_source", "aligned")),
        fusion_mode=str(fusion_cfg.get("mode", "legacy_product")),
        fusion_alpha=float(fusion_cfg.get("alpha", 0.5)),
        visual_temperature=float(fusion_cfg.get("visual_temperature", 1.0)),
        semantic_temperature=float(fusion_cfg.get("semantic_temperature", 1.0)),
    )


def _load_experiment_corpus(config: Mapping[str, Any], paths, base_dir: Path):
    v2 = config.get("corpus_v2", {})
    if bool(v2.get("enabled", False)):
        path = resolve_path(v2["path"], base_dir)
        corpus = load_corpus_v2(path)
        baseline = load_corpus_yaml(paths["corpus"], require_sub_actions=True)

        def normalized(value: str) -> str:
            return re.sub(r"[^a-z0-9]", "", str(value).casefold())

        expected = [normalized(name) for name in baseline]
        actual = [normalized(name) for name in corpus.classes]
        if actual != expected:
            raise ValueError(
                "v2 corpus class order must exactly match the annotation/v1 corpus order"
            )
        return corpus, path
    return load_corpus_yaml(paths["corpus"], require_sub_actions=True), paths["corpus"]


def _checkpoint_argument(
    config: Mapping[str, Any], cli_value: Optional[str], config_base: Path
) -> Optional[str]:
    """Resolve CLI/env paths from cwd and YAML paths from the YAML directory."""

    model = config["model"]
    env_name = str(model.get("checkpoint_env") or "CLIP_CHECKPOINT")
    env_value = os.environ.get(env_name)
    if cli_value:
        return str(resolve_path(cli_value))
    if env_value:
        return str(resolve_path(env_value))
    configured = model.get("checkpoint")
    return str(resolve_path(configured, config_base)) if configured else None


def _peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / 1024**2)


def _report_runtime(
    device: torch.device,
    runtime: Mapping[str, Any],
    *,
    include_peak: bool = False,
) -> None:
    """Report runtime behavior to stderr without changing result JSON schemas."""

    payload: Dict[str, Any] = {
        "device": str(device),
        "amp_requested": bool(runtime.get("amp", False)),
        "amp_enabled": cuda_amp_enabled(device, bool(runtime.get("amp", False))),
        "autocast_dtype": "float16" if cuda_amp_enabled(
            device, bool(runtime.get("amp", False))
        ) else "float32",
        "allow_tf32": bool(runtime.get("allow_tf32", False)),
        "deterministic": bool(runtime.get("deterministic", True)),
    }
    if include_peak:
        payload["peak_memory_mb"] = _peak_memory_mb(device)
    print("runtime: " + json.dumps(payload, ensure_ascii=False), file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: Dict[str, Any] = {}
    if args.checkpoint:
        overrides["model"] = {"checkpoint": args.checkpoint}
    runtime_overrides: Dict[str, Any] = {}
    if args.amp is not None:
        runtime_overrides["amp"] = bool(args.amp)
    if args.allow_tf32 is not None:
        runtime_overrides["allow_tf32"] = bool(args.allow_tf32)
    if runtime_overrides:
        overrides["runtime"] = runtime_overrides
    config = load_config(args.config, overrides=overrides)
    base_dir = _config_base(args.config)
    paths = require_data_paths(config, base_dir=base_dir)
    corpus, active_corpus_path = _load_experiment_corpus(config, paths, base_dir)
    if args.command == "validate":
        payload = {key: str(value) for key, value in paths.items()}
        payload["active_corpus"] = str(active_corpus_path)
        payload["corpus_schema"] = "v2" if bool(config["corpus_v2"]["enabled"]) else "v1"
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    device = setup_runtime(config, device=args.device)
    amp = bool(config["runtime"].get("amp", False))
    if device.type == "cuda":
        # PyTorch 2.0 on Windows rejects reset_peak_memory_stats before the
        # lazy CUDA context exists.  Initialise it explicitly so runtime
        # reporting never prevents the experiment itself from starting.
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    _report_runtime(device, config["runtime"])
    checkpoint = _checkpoint_argument(config, args.checkpoint, base_dir)
    model = _build_model(config, corpus, checkpoint)
    _promote_trainable_parameters_to_fp32(model)
    model.to(device)
    if args.resume and args.command not in {"train", "smoke"}:
        raise ValueError("--resume is only valid for train or smoke")
    if args.resume and args.weights:
        raise ValueError("--resume and --weights are mutually exclusive")
    resume_payload: Optional[Mapping[str, Any]] = None
    if args.resume:
        loaded_resume = torch.load(args.resume, map_location=device)
        if not isinstance(loaded_resume, Mapping) or "state" not in loaded_resume:
            raise ValueError("--resume must contain a training checkpoint with a state entry")
        resume_payload = loaded_resume
        model.load_state_dict(resume_payload["state"], strict=True)
    else:
        _load_weights(model, args.weights, device)

    output_dir = (
        resolve_path(args.output)
        if args.output
        else resolve_path(config["output"]["root"], base_dir)
    )
    _snapshot_config(config, output_dir)
    test_episodes = int(args.episodes or config["episodes"]["test"])

    test_dataset = _make_dataset(
        paths["test"],
        paths["root"],
        config,
        train=False,
        temporal_window=args.temporal_window,
    )
    test_loader = _make_loader(
        test_dataset, config, train=False, episodes=test_episodes, seed_offset=20_000
    )

    if args.command == "calibrate-fusion":
        validation_dataset = _make_dataset(
            paths["val"], paths["root"], config, train=False
        )
        validation_loader = _make_loader(
            validation_dataset,
            config,
            train=False,
            episodes=int(args.episodes or config["episodes"]["val"]),
            seed_offset=10_000,
        )
        pseudo = config["pseudo_validation"]
        selection = calibrate_global_alpha(
            validation_loader,
            model,
            device,
            alpha_grid=pseudo["alpha_grid"],
            mode=str(config["model"]["fusion"].get("mode", "probability_product")),
            tie_anchor=float(pseudo.get("global_alpha", 0.5)),
            flat_tolerance=float(pseudo.get("flat_tolerance", 1.0e-12)),
            amp=amp,
        )
        result = selection.as_dict()
        destination = output_dir / "global_alpha_validation.json"
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        _report_runtime(device, config["runtime"], include_peak=True)
        return 0

    if args.command == "diagnose":
        result = diagnose_order(
            test_loader,
            model,
            device,
            seed=int(config["diagnosis_3a"]["permutation_seed"]),
            include_double_reverse=bool(config["diagnosis_3a"].get("include_c5", False)),
            amp=amp,
        )
        json_path, csv_path = save_diagnostics(result, output_dir)
        print(f"diagnostics: {json_path}\nepisodes: {csv_path}")
        _report_runtime(device, config["runtime"], include_peak=True)
        return 0

    if args.command == "evaluate":
        result = evaluate(
            test_loader,
            model,
            device,
            transport_dir=output_dir / "transport" if args.save_transport else None,
            amp=amp,
        )
        print(json.dumps(result, indent=2))
        _report_runtime(device, config["runtime"], include_peak=True)
        return 0

    if args.command == "evaluate-adaptive":
        pseudo = config["pseudo_validation"]
        result = evaluate_adaptive_fusion(
            test_loader,
            model,
            device,
            alpha_grid=pseudo["alpha_grid"],
            global_alpha=float(pseudo["global_alpha"]),
            views=int(pseudo["views"]),
            seed=int(config["runtime"]["seed"]),
            mode=str(config["model"]["fusion"].get("mode", "probability_product")),
            flat_tolerance=float(pseudo.get("flat_tolerance", 1.0e-12)),
            amp=amp,
        )
        result_path = output_dir / "adaptive_fusion.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        _report_runtime(device, config["runtime"], include_peak=True)
        return 0

    smoke = args.command == "smoke"
    counts = config["m0"] if smoke else config["episodes"]
    train_count = int(counts["smoke_train_episodes"] if smoke else counts["train"])
    val_count = int(counts["smoke_val_episodes"] if smoke else counts["val"])
    if smoke:
        test_episodes = int(counts["smoke_test_episodes"])
        test_loader = _make_loader(
            test_dataset, config, train=False, episodes=test_episodes, seed_offset=20_000
        )
    train_dataset = _make_dataset(paths["train"], paths["root"], config, train=True)
    val_dataset = _make_dataset(paths["val"], paths["root"], config, train=False)
    train_loader = _make_loader(train_dataset, config, train=True, episodes=train_count, seed_offset=0)
    val_loader = _make_loader(val_dataset, config, train=False, episodes=val_count, seed_offset=10_000)

    optimizer = torch.optim.SGD(
        _trainable_parameters(model), lr=float(config["optimization"]["learning_rate"])
    )
    scaler = create_grad_scaler(device, amp)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[12, 24, 36, 48, 60, 72], gamma=0.8
    )
    start_epoch = int(config["optimization"]["start_epoch"])
    stop_epoch = start_epoch + 1 if smoke else int(config["optimization"]["stop_epoch"])
    warmup = -1 if smoke else int(config["optimization"]["warm_up_epoch"])
    regularizer = config["regularizer_3b"]
    order_weight = float(regularizer["weight"]) if regularizer.get("enabled") else 0.0
    best = float("-inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history = []
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        scaler_state = resume_payload.get("scaler")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(resume_payload["epoch"]) + 1
        best = float(resume_payload.get("best", best))
        history = list(resume_payload.get("history", history))
    started = time.perf_counter()
    for epoch in range(start_epoch, stop_epoch):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        train_metrics = train_one_epoch(
            train_loader,
            model,
            optimizer,
            device,
            amp=amp,
            scaler=scaler,
            order_weight=order_weight,
            order_mode=str(regularizer["kind"]),
            order_margin=float(regularizer["margin"]),
        )
        scheduler.step()
        row: Dict[str, Any] = {"epoch": epoch, **{"train_" + k: v for k, v in train_metrics.items()}}
        if epoch >= warmup:
            val_metrics = evaluate(val_loader, model, device, amp=amp)
            row.update({"val_" + key: value for key, value in val_metrics.items()})
            if val_metrics["fused_accuracy"] > best:
                best = val_metrics["fused_accuracy"]
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": epoch, "state": model.state_dict(), "config": config}, best_path)
        history.append(row)
        output_dir.mkdir(parents=True, exist_ok=True)
        last_tmp = last_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "epoch": epoch,
                "state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best": best,
                "history": history,
                "config": config,
            },
            last_tmp,
        )
        last_tmp.replace(last_path)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if best_path.exists():
        _load_weights(model, str(best_path), device)
    test_metrics = evaluate(test_loader, model, device, amp=amp)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"history": history, "test": test_metrics}, handle, indent=2, ensure_ascii=False)
    elapsed = time.perf_counter() - started
    run_metrics = resolve_path(config["output"]["run_metrics"], base_dir)
    _append_run_metrics(
        run_metrics,
        {
            "experiment": str(config["experiment"]["name"]),
            "phase": "smoke" if smoke else str(config["experiment"]["phase"]),
            "accuracy": test_metrics["fused_accuracy"],
            "confidence_interval": test_metrics["fused_ci95"],
            "order_sensitivity": "",
            "wall_time_seconds": elapsed,
            "peak_memory_mb": (
                _peak_memory_mb(device)
            ),
            "notes": str(config["experiment"].get("message", "")),
        },
    )
    print(json.dumps(test_metrics, indent=2))
    _report_runtime(device, config["runtime"], include_peak=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
