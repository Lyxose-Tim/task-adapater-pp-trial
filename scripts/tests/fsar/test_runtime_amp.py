from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn

import fsar.experiment as experiment
from fsar.cli import _promote_trainable_parameters_to_fp32, build_parser
from fsar.config import load_config


class _TinyEpisodeModel(nn.Module):
    n_way = 2
    n_query = 1

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.1))

    def forward(self, images, labels, return_aux=True):
        rows = torch.stack(
            (
                torch.stack((self.anchor + 0.2, self.anchor - 0.1)),
                torch.stack((self.anchor - 0.1, self.anchor + 0.2)),
            )
        )
        return {
            "visual_logits": rows,
            "semantic_logits": rows,
            "final_logits": rows,
        }


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters) -> None:
        super().__init__(parameters, lr=0.01)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class _FakeScaler:
    def __init__(self) -> None:
        self.calls = []

    def scale(self, loss):
        self.calls.append("scale")
        return loss

    def step(self, optimizer):
        self.calls.append("step")
        optimizer.step()

    def update(self):
        self.calls.append("update")


def _batch():
    return (
        torch.randn(2, 2, 2, 3, 2, 2),
        torch.tensor([[0, 0], [1, 1]]),
    )


def test_cuda_autocast_uses_float16_and_disabled_path_is_noop(monkeypatch) -> None:
    calls = []

    @contextmanager
    def fake_autocast(**kwargs):
        calls.append(kwargs)
        yield

    monkeypatch.setattr(torch, "autocast", fake_autocast)
    with experiment._autocast_context(torch.device("cuda:0"), amp=True):
        pass
    with experiment._autocast_context(torch.device("cuda:0"), amp=False):
        pass
    with experiment._autocast_context(torch.device("cpu"), amp=True):
        pass

    assert calls == [{"device_type": "cuda", "dtype": torch.float16}]


def test_train_amp_uses_scaler_while_disabled_path_uses_plain_step(monkeypatch) -> None:
    entered = []

    @contextmanager
    def fake_context(device, amp=False):
        entered.append(bool(amp))
        yield

    monkeypatch.setattr(experiment, "_autocast_context", fake_context)
    # Simulate CUDA AMP control flow while keeping tensors on CPU for the test.
    monkeypatch.setattr(experiment, "cuda_amp_enabled", lambda device, amp=False: bool(amp))

    enabled_model = _TinyEpisodeModel()
    enabled_optimizer = _CountingSGD(enabled_model.parameters())
    scaler = _FakeScaler()
    experiment.train_one_epoch(
        [_batch()],
        enabled_model,
        enabled_optimizer,
        torch.device("cpu"),
        amp=True,
        scaler=scaler,
    )
    assert scaler.calls == ["scale", "step", "update"]
    assert enabled_optimizer.step_calls == 1

    disabled_model = _TinyEpisodeModel()
    disabled_optimizer = _CountingSGD(disabled_model.parameters())
    disabled_scaler = _FakeScaler()
    experiment.train_one_epoch(
        [_batch()],
        disabled_model,
        disabled_optimizer,
        torch.device("cpu"),
        amp=False,
        scaler=disabled_scaler,
    )
    assert disabled_scaler.calls == []
    assert disabled_optimizer.step_calls == 1
    assert entered == [True, False]


def test_ssv2_runtime_defaults_and_cli_overrides() -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_config(root / "configs" / "innovation3_ssv2_smoke.yaml")
    assert config["runtime"]["amp"] is True
    assert config["runtime"]["allow_tf32"] is True
    assert config["m0"]["smoke_train_episodes"] == 1
    assert config["m0"]["smoke_val_episodes"] == 1
    assert config["m0"]["smoke_test_episodes"] == 1

    args = build_parser().parse_args(
        ["evaluate", "--no-amp", "--no-allow-tf32"]
    )
    assert args.amp is False
    assert args.allow_tf32 is False

    resume = build_parser().parse_args(["train", "--resume", "last.pt"])
    assert resume.resume == "last.pt"


def test_cli_initializes_cuda_before_resetting_peak_memory() -> None:
    root = Path(__file__).resolve().parents[3]
    source = (root / "fsar" / "cli.py").read_text(encoding="utf-8")
    assert source.index("torch.cuda.init()") < source.index(
        "torch.cuda.reset_peak_memory_stats(device)"
    )


def test_trainable_fp16_parameters_are_promoted_but_frozen_weights_are_not() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)).half()
    for parameter in model[1].parameters():
        parameter.requires_grad_(False)

    promoted = _promote_trainable_parameters_to_fp32(model)

    assert promoted == 2
    assert {parameter.dtype for parameter in model[0].parameters()} == {torch.float32}
    assert {parameter.dtype for parameter in model[1].parameters()} == {torch.float16}
