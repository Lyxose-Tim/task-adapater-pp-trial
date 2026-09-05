"""创新点 1 · 语义条件、顺序保持的软阶段分配（不平衡 OT 重写式(16)）。

官方训练路径（根 models.py）的 OT 打分接口，手册《创新点1_实验流程手册_v1》§1.1/§1.2。
核心 log 域 Sinkhorn / 位置先验 / 质量加权打分复用**已单测**的 `fsar.ot`（同一数学：
方案设计 §1.1）；本模块只提供手册规定的对外接口并处理官方张量朝向。

全模块 fp32；调用侧（models.py）入口 `.float()`、出口 `.half()`。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from fsar.ot import (
    OrderAwareOptimalTransport,
    position_distance,
    sinkhorn_log as _fsar_sinkhorn_log,
)

Tensor = torch.Tensor


def build_D(num_frames: int, num_stages: int, *, device: Optional[torch.device] = None) -> Tensor:
    """位置偏差矩阵 D(i,s)=((i/(T-1))-(s/(K-1)))²，[T,K]；K=1 时全零。

    与 `fsar.ot.position_distance` 同式（linspace(0,1,T)[i]=i/(T-1)）。
    """
    return position_distance(num_frames, num_stages, device=device)


def sinkhorn_log(cost: Tensor, eps: float, rho: Optional[float] = None, iters: int = 30) -> Tensor:
    """单侧不平衡 log 域 Sinkhorn（帧侧硬约束、阶段侧软约束 τ=ρ/(ρ+ε)）。

    `cost` 以 [...,T,K] 结尾；`rho=None` 为平衡档（τ=1）。返回传输计划 π（fp32）。
    边际固定为均匀 a=1/T、b=1/K（方案 §1.1）。
    """
    return _fsar_sinkhorn_log(
        cost, epsilon=eps, rho=(math.inf if rho is None else rho), iterations=iters
    )


def ot_stage_scores(
    F_frames: Tensor,
    T_c: Tensor,
    eps: float,
    lam: float = 0.0,
    rho: Optional[float] = None,
    iters: int = 30,
    weight: str = "mass",
) -> Tuple[Tensor, Tensor, Tensor]:
    """对每个 (查询 q, 类 c) 求解 T×K 传输问题，返回语义分。

    Args:
        F_frames: 对齐帧特征 [NQ, T, d]（Option A 取 q_aft_tm 的 7 帧，B 取 8 帧）。
        T_c: 阶段文本特征 [C, K, d]。
        eps/lam/rho/iters: ε（熵）、λ（位置先验权重）、ρ（None=平衡）、Sinkhorn 迭代数。
        weight: "mass" = 质量加权 ŵ_s=m_s/Σm（方案主线）；"uniform" = 等权 1/K（消融）。

    Returns:
        S [NQ, C]（语义分，直接替换式(16) 的 cos_score，朝向 (查询, 类)）、
        π [NQ, C, T, K]、m [NQ, C, K]。全 fp32。
    """
    if weight not in {"mass", "uniform"}:
        raise ValueError("weight must be 'mass' or 'uniform'")
    ot = OrderAwareOptimalTransport(
        epsilon=eps,
        lambda_pos=lam,
        rho=(math.inf if rho is None else rho),
        iterations=iters,
    )
    result = ot(F_frames.float(), T_c.float())
    if weight == "mass":
        score = result.score
    else:
        # 等权：S = mean_s cos(t_s, f̂_s)（不平衡质量不参与加权）
        text = F.normalize(T_c.float(), dim=-1)
        stage_similarity = torch.einsum("ckd,qckd->qck", text, result.fhat)
        score = stage_similarity.mean(dim=-1)
    return score, result.plan, result.mass


def ot_plan_stats(plan: Tensor) -> dict:
    """从传输计划 π 计算数值健康统计（**全程 detach，不入梯度/不改分数/不占计算图**）。

    Args:
        plan: [..., T, K] 传输计划（ot_stage_scores 的第二个返回值）。

    Returns:
        逐 (前导维) 张量字典：
          pi_entropy_norm  归一化熵 H(π)/log(TK) ∈[0,1]，趋 1=π 趋均匀（ε 过大）；
          row_residual     ‖π·1_K − a‖₁（帧侧硬边际残差，a=1/T，应≈0）；
          col_residual     ‖π·1_T − b‖₁（阶段侧边际残差，b=1/K；平衡档应≈0，不平衡为诊断量）；
          stage_mass_min   min_s m_s（阶段最小质量，识别"某阶段从不被分帧"的退化）；
          total_mass       Σπ（应≈1）。
    """
    p = plan.detach()
    num_frames, num_stages = p.shape[-2], p.shape[-1]
    a = 1.0 / num_frames
    b = 1.0 / num_stages
    total = p.sum(dim=(-2, -1))
    pn = p / total.clamp_min(1e-12).unsqueeze(-1).unsqueeze(-1)
    ent = -(pn.clamp_min(1e-12) * pn.clamp_min(1e-12).log()).sum(dim=(-2, -1))
    ent_norm = ent / math.log(num_frames * num_stages) if num_frames * num_stages > 1 else ent * 0.0
    stage_mass = p.sum(dim=-2)                       # [..., K]
    return {
        "pi_entropy_norm": ent_norm,                 # [...]
        "row_residual": (p.sum(dim=-1) - a).abs().sum(dim=-1),   # [...]
        "col_residual": (stage_mass - b).abs().sum(dim=-1),      # [...]
        "stage_mass_min": stage_mass.min(dim=-1).values,         # [...]
        "total_mass": total,                         # [...]
    }


__all__ = ["build_D", "sinkhorn_log", "ot_stage_scores", "ot_plan_stats"]
