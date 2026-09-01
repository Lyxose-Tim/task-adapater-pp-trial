"""诊断插桩单测（手册 §2.3/§2.2）：验证 forward 的 permutation 参数

- 正确重排每类子动作顺序（只动语义分支的文本输入）；
- **不改变仅视觉分**（§2.2 内置 sanity）；
- 恒等排列 == 不传排列（no-op）。

在 CPU 上用合成编码器跑（绕过真实 CLIP/CUDA），秒级。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

TASK_DIR = Path(__file__).resolve().parents[3]
_SHADOWED = ("models", "utils", "module_adapter", "module_sem_adapter")


@pytest.fixture()
def official(monkeypatch):
    saved = {name: sys.modules.pop(name, None) for name in _SHADOWED}
    monkeypatch.chdir(TASK_DIR)
    monkeypatch.syspath_prepend(str(TASK_DIR))
    try:
        import models as official_models

        yield official_models
    finally:
        for name in _SHADOWED:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


class FakeVisualBackbone(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        value = videos.float().mean(dim=(1, 3, 4))
        index = (value.round().long() - 1).clamp(0, self.num_classes - 1)
        return F.one_hot(index, self.num_classes).float()


class _Tokens:
    """token 包装：`.to(device)` 忽略设备（保持 CPU），绕过 forward 里的 .to('cuda')。"""

    def __init__(self, tensor):
        self._tensor = tensor

    def to(self, _device):
        return self._tensor


class SpyTokenizer:
    """记录收到的 prompt，返回可被 FakeText 使用的 token（子动作文本字节和）。"""

    def __init__(self):
        self.calls = []

    def __call__(self, prompts):
        self.calls.append(list(prompts))
        # token 值 = prompt 字节和，保证不同子动作 → 不同 token
        tensor = torch.tensor([[sum(p.encode("utf-8")) % 89] for p in prompts], dtype=torch.long)
        return _Tokens(tensor)


class OrderSensitiveText(nn.Module):
    def __init__(self, n_way: int):
        super().__init__()
        self.n_way = n_way

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # 每个 token 映射到一个可分嵌入（one-hot over n_way），随子动作内容变化
        idx = (tokens.squeeze(-1).long() % self.n_way)
        return F.one_hot(idx, self.n_way).float()


def _build(official, n_way=3, n_support=1, n_query=1):
    model = official.TaskAdapter.__new__(official.TaskAdapter)
    nn.Module.__init__(model)
    model.n_way, model.n_support, model.n_query = n_way, n_support, n_query
    model.cls_name = [f"class-{i}" for i in range(n_way)]
    # 每类三个可区分的子动作（诊断的干预对象）
    model.entxt = {name: {"sub_act_en_li": [f"{name}-start", f"{name}-middle", f"{name}-end"]}
                   for name in model.cls_name}
    model.feature = FakeVisualBackbone(n_way)
    model.text = OrderSensitiveText(n_way)
    model.ca = official.CrossAttention()
    return model


def _episode(n_way=3, n_support=1, n_query=1):
    sq = n_support + n_query
    episode = torch.stack([torch.full((sq, 8, 3, 4, 4), float(c + 1)) for c in range(n_way)])
    x = episode.reshape(-1, 3, 4, 4)
    labels = torch.arange(n_way).unsqueeze(1).repeat(1, sq)
    return x, labels


def test_permutation_reorders_subactions_in_prompts(official, monkeypatch):
    spy = SpyTokenizer()
    monkeypatch.setattr(official.clip, "tokenize", spy)
    monkeypatch.setattr(official, "cosine_similarity",
                        lambda a, b: F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).transpose(-2, -1))
    model = _build(official)
    x, labels = _episode()

    spy.calls.clear()
    model(x, labels, permutation=(2, 1, 0))
    # class-0 的 prompt 应按 end/middle/start（反转）出现
    flat = [p for call in spy.calls for p in call]
    c0 = [p for p in flat if "class-0" in p]
    assert c0 == ["A video of action about class-0: class-0-end",
                  "A video of action about class-0: class-0-middle",
                  "A video of action about class-0: class-0-start"]


def test_text_permutation_leaves_visual_scores_unchanged(official, monkeypatch):
    # §2.2 sanity：文本扰动只应改变语义分支，不动仅视觉分
    monkeypatch.setattr(official.clip, "tokenize", SpyTokenizer())
    monkeypatch.setattr(official, "cosine_similarity",
                        lambda a, b: F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).transpose(-2, -1))
    model = _build(official)
    x, labels = _episode()

    vis0, sem0 = model(x, labels, permutation=(0, 1, 2))
    vis_id, sem_id = model(x, labels, permutation=None)
    vis_rev, sem_rev = model(x, labels, permutation=(2, 1, 0))

    # 恒等排列 == 不传排列（no-op）
    torch.testing.assert_close(vis0, vis_id)
    torch.testing.assert_close(sem0, sem_id)
    # §2.2 sanity：视觉分在任何文本排列下都不变（重排只改变语义分支的文本输入）
    torch.testing.assert_close(vis0, vis_rev)
    # 排列确实进入了语义分支的文本前向：见 test_permutation_reorders_subactions_in_prompts
    # （子动作在 prompt 中被反序）。sem 的顺序敏感度是真实权重上的实证量（OS），
    # 由云端诊断测量，不在合成单测里钉死具体数值。
