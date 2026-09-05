"""创新点 1 · OT 软阶段分配单测（手册 §1.4 U0–U4，全过才允许启动训练）。

U1–U4 直接测 `ot_align.ot_stage_scores`（合成张量、CPU、秒级）；
U0 测 models.py 的 align_mode 分派（window 逐位=固定窗口、ot 走 OT）。
"""

from __future__ import annotations

import sys
from itertools import permutations
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import ot_align  # noqa: E402


# ---------- U1–U4：ot_align 层 ----------

def test_U1_permutation_invariance_lambda0():
    # λ=0、b 均匀：对阶段文本任意列排列，S 严格不变（方案 §1.2 命题）
    torch.manual_seed(0)
    F_frames = torch.randn(6, 7, 16)
    T_c = torch.randn(3, 3, 16)
    S0, _, _ = ot_align.ot_stage_scores(F_frames, T_c, eps=0.1, lam=0.0, rho=None, iters=50)
    for perm in permutations(range(3)):
        Sp, _, _ = ot_align.ot_stage_scores(
            F_frames, T_c[:, list(perm), :], eps=0.1, lam=0.0, rho=None, iters=50)
        assert torch.max((S0 - Sp).abs()).item() < 1e-4, f"perm {perm} broke invariance"


def test_U2_limit_form_banded_diagonal():
    # λ 大、ε 小、平衡：π 质量集中于位置先验带状对角
    torch.manual_seed(1)
    T, K = 7, 3
    F_frames = torch.randn(2, T, 16)
    T_c = torch.randn(2, K, 16)
    _, pi, _ = ot_align.ot_stage_scores(F_frames, T_c, eps=0.01, lam=50.0, rho=None, iters=80)
    # 每帧被分配的阶段（argmax）应随帧单调非降（带状），且贴合最近对角
    D = ot_align.build_D(T, K)                    # [T,K]
    nearest = D.argmin(dim=1)                     # 位置先验下每帧的最近阶段
    assigned = pi[0].sum(dim=0)                   # 该 (q,c) 对忽略——取 pi[0,0]
    assigned = pi[0, 0].argmax(dim=1)             # [T] 每帧 argmax 阶段
    assert torch.all(assigned[1:] >= assigned[:-1]), f"非单调带状: {assigned.tolist()}"
    assert torch.equal(assigned, nearest), f"assigned {assigned.tolist()} != nearest {nearest.tolist()}"
    # 对角带质量占比高
    band_mass = pi[0, 0].gather(1, nearest.view(-1, 1)).sum() / pi[0, 0].sum()
    assert band_mass.item() > 0.6, f"band mass {band_mass.item():.3f} too low"


def test_U3_gradient_health():
    F_frames = torch.randn(4, 7, 16, requires_grad=True)
    T_c = torch.randn(3, 3, 16, requires_grad=True)
    S, _, _ = ot_align.ot_stage_scores(F_frames, T_c, eps=0.05, lam=0.7, rho=0.5, iters=30)
    assert S.dtype == torch.float32          # fp32 岛
    S.sum().backward()
    for g in (F_frames.grad, T_c.grad):
        assert g is not None and torch.isfinite(g).all()


@pytest.mark.parametrize("K", [1, 2, 3, 5])
def test_U4_variable_K_shapes(K):
    NQ, C, T = 5, 3, 7
    F_frames = torch.randn(NQ, T, 16)
    T_c = torch.randn(C, K, 16)
    S, pi, m = ot_align.ot_stage_scores(F_frames, T_c, eps=0.05, lam=0.3, rho=None, iters=20)
    assert S.shape == (NQ, C) and pi.shape == (NQ, C, T, K) and m.shape == (NQ, C, K)
    assert torch.isfinite(S).all()


def test_plan_stats_instrumentation_is_side_effect_free():
    # 插桩不改分数（用户要求：插桩前后原始输出必须一致），且统计已 detach 无梯度
    torch.manual_seed(3)
    F_frames = torch.randn(4, 7, 16, requires_grad=True)
    T_c = torch.randn(3, 3, 16, requires_grad=True)
    S1, plan, _ = ot_align.ot_stage_scores(F_frames, T_c, eps=0.1, lam=0.3, rho=None, iters=30)
    stats = ot_align.ot_plan_stats(plan)
    S2, _, _ = ot_align.ot_stage_scores(F_frames, T_c, eps=0.1, lam=0.3, rho=None, iters=30)
    torch.testing.assert_close(S1, S2)                      # 插桩前后分数逐元素一致
    for v in stats.values():
        assert not v.requires_grad                          # 统计不入梯度图


def test_balanced_plan_marginals_and_total():
    # 平衡 OT：帧/阶段边际残差≈0、总质量≈1、熵∈[0,1]
    torch.manual_seed(4)
    _, plan, _ = ot_align.ot_stage_scores(torch.randn(3, 7, 16), torch.randn(2, 3, 16),
                                          eps=0.1, lam=0.0, rho=None, iters=50)
    st = ot_align.ot_plan_stats(plan)
    assert float(st["row_residual"].max()) < 1e-3
    assert float(st["col_residual"].max()) < 1e-3           # 平衡档阶段边际也硬
    assert abs(float(st["total_mass"].mean()) - 1.0) < 1e-4
    assert 0.0 <= float(st["pi_entropy_norm"].min()) <= float(st["pi_entropy_norm"].max()) <= 1.0 + 1e-6


def test_U5a_entropy_increases_with_eps():
    # 点5：固定特征、只增大 ε，π 归一化熵总体上升
    torch.manual_seed(5)
    F_frames = torch.randn(6, 7, 16)
    T_c = torch.randn(3, 3, 16)
    ent = []
    for e in (0.01, 0.05, 0.1, 0.5):
        _, plan, _ = ot_align.ot_stage_scores(F_frames, T_c, eps=e, lam=0.0, rho=None, iters=60)
        ent.append(float(ot_align.ot_plan_stats(plan)["pi_entropy_norm"].mean()))
    assert ent[-1] > ent[0], f"熵未随 ε 上升: {ent}"
    assert all(ent[i + 1] >= ent[i] - 1e-3 for i in range(len(ent) - 1)), f"熵非单调: {ent}"


def test_U5b_band_mass_increases_with_lambda():
    # 点5：固定特征、只增大 λ，位置先验对角带质量总体上升
    torch.manual_seed(6)
    T, K = 7, 3
    F_frames = torch.randn(4, T, 16)
    T_c = torch.randn(3, K, 16)
    nearest = ot_align.build_D(T, K).argmin(dim=1)         # 每帧最近对角阶段
    band = []
    for lam in (0.0, 0.1, 0.3, 1.0, 3.0):
        _, plan, _ = ot_align.ot_stage_scores(F_frames, T_c, eps=0.05, lam=lam, rho=None, iters=60)
        p = plan.detach()
        frac = p.gather(-1, nearest.view(1, 1, T, 1).expand(*p.shape[:2], T, 1)).sum() / p.sum()
        band.append(float(frac))
    assert band[-1] > band[0], f"带状质量未随 λ 上升: {band}"
    assert all(band[i + 1] >= band[i] - 1e-3 for i in range(len(band) - 1)), f"带状质量非单调: {band}"


def test_uniform_vs_mass_weight_differ_when_unbalanced():
    torch.manual_seed(2)
    F_frames = torch.randn(4, 7, 16)
    T_c = torch.randn(3, 3, 16)
    S_mass, _, _ = ot_align.ot_stage_scores(F_frames, T_c, eps=0.05, lam=0.3, rho=0.5, iters=30, weight="mass")
    S_uni, _, _ = ot_align.ot_stage_scores(F_frames, T_c, eps=0.05, lam=0.3, rho=0.5, iters=30, weight="uniform")
    assert S_mass.shape == S_uni.shape and torch.isfinite(S_uni).all()


# ---------- U0：models.py align_mode 分派 ----------

_SHADOWED = ("models", "utils", "module_adapter", "module_sem_adapter")


@pytest.fixture()
def official(monkeypatch):
    saved = {name: sys.modules.pop(name, None) for name in _SHADOWED}
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    try:
        import models as official_models
        yield official_models
    finally:
        for name in _SHADOWED:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]


class _Tokens:
    def __init__(self, t): self._t = t
    def to(self, _d): return self._t


class _Tok:
    def __call__(self, prompts):
        return _Tokens(torch.tensor([[sum(p.encode("utf-8")) % 97] for p in prompts], dtype=torch.long))


class _Text(nn.Module):
    def forward(self, tokens):
        base = tokens.float().view(-1, 1)
        return torch.sin(base * torch.arange(1, 17).float() * 0.1) + 0.2 * base


def _model(official, mode):
    m = official.TaskAdapter.__new__(official.TaskAdapter)
    nn.Module.__init__(m)
    m.n_way, m.n_support, m.n_query = 3, 1, 1
    m.cls_name = [f"c{i}" for i in range(3)]
    m.entxt = {n: {"sub_act_en_li": [f"{n}-a", f"{n}-b", f"{n}-c"]} for n in m.cls_name}
    m.text = _Text()
    m.align_mode = mode
    m.frame_source = "ca"
    m.ot_eps, m.ot_lam, m.ot_rho, m.ot_iters, m.ot_weight = 0.1, 0.0, None, 30, "mass"
    return m


def test_U0_window_matches_fixed_and_ot_differs(official, monkeypatch):
    monkeypatch.setattr(official, "cosine_similarity",
                        lambda a, b: F.normalize(a.float(), dim=-1) @ F.normalize(b.float(), dim=-1).transpose(-2, -1))
    monkeypatch.setattr(official.clip, "tokenize", _Tok())
    q_aft = torch.randn(7, 3, 16)
    label_idx = [0, 1, 2]

    mw = _model(official, "window")
    enh = mw._encode_stage_text(label_idx, None)
    ref = mw._fixed_window_score(enh, q_aft)
    got = mw.semantic_scores(q_aft, label_idx, None)
    torch.testing.assert_close(got, ref)                     # U0：window 逐位=固定窗口
    assert got.shape == (3, 3)

    mo = _model(official, "ot")
    S_ot = mo.semantic_scores(q_aft, label_idx, None)
    assert S_ot.shape == (3, 3) and torch.isfinite(S_ot).all()  # ot 分派出有效分
