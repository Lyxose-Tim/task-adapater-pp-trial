"""创新点 3b 顺序对比正则单测（手册 §3.1/§3.2）。

- order_margin_loss 有限、非负、可回传；
- **免费午餐等价**：enh 正序编码一次后按 K 轴 index_select 得到的排列分，
  与「实际重排子动作再重新编码」逐元素一致（前提=文本编码器等变，O-1）。
  用逐行独立（因而等变）的合成编码器在 CPU 上验证。
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


class _Tokens:
    def __init__(self, tensor):
        self._tensor = tensor

    def to(self, _device):
        return self._tensor


class RowIndepTokenizer:
    """每个 prompt → 独立 token（字节和）；逐行独立保证编码器等变。"""

    def __call__(self, prompts):
        return _Tokens(torch.tensor([[sum(p.encode("utf-8")) % 97] for p in prompts], dtype=torch.long))


class EquivariantText(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # 每行 token 独立映射到 dim 维嵌入（与行序无关 → 等变）
        base = tokens.float().view(-1, 1)
        cols = torch.arange(1, self.dim + 1, dtype=torch.float32)
        return torch.sin(base * cols * 0.1) + 0.3 * base


def _build(official, n_way=3, n_support=1, n_query=1, dim=8):
    model = official.TaskAdapter.__new__(official.TaskAdapter)
    nn.Module.__init__(model)
    model.n_way, model.n_support, model.n_query = n_way, n_support, n_query
    model.cls_name = [f"class-{i}" for i in range(n_way)]
    model.entxt = {name: {"sub_act_en_li": [f"{name}-s0", f"{name}-s1", f"{name}-s2"]}
                   for name in model.cls_name}
    model.text = EquivariantText(dim)
    return model


@pytest.fixture()
def patched(official, monkeypatch):
    # utils.cosine_similarity 在 fp16 下 matmul，CPU 需 fp32 镜像（结构不变）
    monkeypatch.setattr(official, "cosine_similarity",
                        lambda a, b: F.normalize(a.float(), dim=-1) @ F.normalize(b.float(), dim=-1).transpose(-2, -1))
    monkeypatch.setattr(official.clip, "tokenize", RowIndepTokenizer())
    return official


def test_order_loss_finite_nonneg_and_differentiable(patched):
    model = _build(patched)
    dim = model.text.dim
    q_aft = torch.randn(7, model.n_way * model.n_query, dim, requires_grad=True)
    from fsar.order import NON_IDENTITY_STAGE_PERMUTATIONS
    loss = model.order_margin_loss(q_aft, [0, 1, 2], NON_IDENTITY_STAGE_PERMUTATIONS, margin=0.1)
    assert torch.isfinite(loss) and loss.item() >= 0.0
    loss.backward()
    assert q_aft.grad is not None and torch.isfinite(q_aft.grad).all()


def test_free_lunch_reindex_equals_reencode(patched):
    # 手册 §3.2：order_margin_loss 用 enh 正序编码+index_select；
    # 与逐排列重新编码（semantic_scores(perm)）的 margin 应逐元素一致。
    model = _build(patched)
    dim = model.text.dim
    q_aft = torch.randn(7, model.n_way * model.n_query, dim)
    label_idx = [0, 1, 2]
    margin = 0.15
    from fsar.order import NON_IDENTITY_STAGE_PERMUTATIONS as PERMS

    got = model.order_margin_loss(q_aft, label_idx, PERMS, margin)

    # 参考：逐排列真编码
    y = torch.arange(model.n_way).repeat_interleave(model.n_query).view(-1, 1)
    sem_id = model.semantic_scores(q_aft, label_idx, None)
    s_pos = sem_id.gather(1, y).squeeze(1)
    ref_terms = []
    for perm in PERMS:
        sem_p = model.semantic_scores(q_aft, label_idx, perm)  # 重新编码
        s_perm = sem_p.gather(1, y).squeeze(1)
        ref_terms.append(F.relu(margin - (s_pos - s_perm)))
    ref = torch.stack(ref_terms, dim=0).mean()

    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)
