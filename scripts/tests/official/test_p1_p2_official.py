"""P1/P2 synthetic checks against the official Task-Adapter-pp code.

手册 §1.3 要求：P2 修复后加形状断言，并用 n_query>1 合成单测验证。
本测试直接导入 `Task-Adapter-pp/models.py`（绕过 __init__ 注入合成编码器，
CPU 秒级），验证：
- P1：forward 第一个返回值是视觉分、第二个是语义分；
- P2：两分支形状均为 (N_Q, n_way) 且行是查询、列是类。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# 结构对齐上游后，官方代码在仓库根目录（原 Task-Adapter-pp/ 已扁平化）。
TASK_DIR = Path(__file__).resolve().parents[3]

# 与 fsar 工具层共享的顶层模块名；导入官方版本前后都要隔离/恢复，
# 避免污染同一 pytest 会话中的其他测试。
_SHADOWED = ("models", "utils", "module_adapter", "module_sem_adapter")


@pytest.fixture()
def official(monkeypatch):
    saved = {name: sys.modules.pop(name, None) for name in _SHADOWED}
    monkeypatch.chdir(TASK_DIR)  # models.py 导入时读 ./config.yaml
    monkeypatch.syspath_prepend(str(TASK_DIR))
    try:
        import models as official_models

        yield official_models
    finally:
        for name in _SHADOWED:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


class FakeTokens:
    """官方 forward 内硬编码 .to('cuda')；测试中忽略目标设备。"""

    def __init__(self, count: int):
        self._tensor = torch.zeros(count, 77, dtype=torch.long)

    def to(self, _device):
        return self._tensor


class FakeVisualBackbone(nn.Module):
    """像素值 c+1 的恒值视频 -> 每帧 one-hot(c) 特征，[B,T,D]。"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        value = videos.float().mean(dim=(1, 3, 4))  # [B, T]
        index = (value.round().long() - 1).clamp(0, self.num_classes - 1)
        return F.one_hot(index, self.num_classes).float()


class ShiftedFakeTextEncoder(nn.Module):
    """第 i 次调用返回 one-hot((i+1) % n_way) 的 K 行。

    刻意让语义分支的最优类 = (真类+1) % n_way，与视觉分支(真类)可区分，
    从而能检出 P1 的返回顺序互换。
    """

    def __init__(self, n_way: int):
        super().__init__()
        self.n_way = n_way
        self.calls = 0

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        shifted = (self.calls + 1) % self.n_way
        self.calls += 1
        row = F.one_hot(torch.tensor(shifted), self.n_way).float()
        return row.expand(tokens.shape[0], self.n_way).clone()


def _build_model(official, n_way: int, n_support: int, n_query: int):
    model = official.TaskAdapter.__new__(official.TaskAdapter)
    nn.Module.__init__(model)
    model.n_way = n_way
    model.n_support = n_support
    model.n_query = n_query
    model.cls_name = [f"class-{i}" for i in range(n_way)]
    model.entxt = {
        name: {"sub_act_en_li": ["start", "middle", "end"]} for name in model.cls_name
    }
    model.feature = FakeVisualBackbone(n_way)
    model.text = ShiftedFakeTextEncoder(n_way)
    model.ca = official.CrossAttention()
    return model


def _cosine_similarity_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """utils.cosine_similarity 的逐行 fp32 镜像。

    官方实现把两侧强转 float16 后做 matmul，而 CPU 不支持 half matmul；
    此处仅改精度、不改任何结构（归一化 + x @ y^T），朝向语义与真实
    实现一致。真实 fp16 路径由 GPU smoke 覆盖。
    """

    assert x.shape[-1] == y.shape[-1]
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return x @ y.transpose(-2, -1)


def test_p1_p2_orientation_with_two_queries(official, monkeypatch):
    monkeypatch.setattr(official.clip, "tokenize", lambda prompts: FakeTokens(len(prompts)))
    monkeypatch.setattr(official, "cosine_similarity", _cosine_similarity_fp32)

    n_way, n_support, n_query = 3, 1, 2
    model = _build_model(official, n_way, n_support, n_query)

    episode = torch.stack(
        [
            torch.full((n_support + n_query, 8, 3, 4, 4), float(c + 1))
            for c in range(n_way)
        ]
    )  # [n_way, s+q, T, C, H, W]
    x = episode.reshape(-1, 3, 4, 4)  # 官方入口是按 (类, 样本, 帧) 展平的帧批
    labels = torch.arange(n_way).unsqueeze(1).repeat(1, n_support + n_query)

    vis, sem = model(x, labels)

    n_q_total = n_way * n_query
    # P2：两分支同为 (N_Q, n_way)。修复前语义分是 (n_way, N_Q)，n_query>1 时在此失败。
    assert vis.shape == (n_q_total, n_way)
    assert sem.shape == (n_q_total, n_way)

    query_classes = torch.arange(n_way).repeat_interleave(n_query)
    # P2：行是查询——第 q 行的最优类由第 q 个查询的内容决定。
    assert torch.equal(vis.float().argmax(dim=1), query_classes)
    # P1：第一个返回值是视觉分（最优类=真类），第二个是语义分。类 c 的
    # 合成文本嵌入被移位为 one-hot((c+1)%n_way)，因此与查询 q 匹配的文本
    # 行是 (q-1)%n_way——语义 argmax 应为 (q-1)%n_way，与视觉可区分；
    # P1 互换时视觉/语义两条 argmax 断言同时失败。
    assert torch.equal(sem.float().argmax(dim=1), (query_classes - 1) % n_way)
