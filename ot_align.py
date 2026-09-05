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


def ot_position_only_plan(
    num_frames: int,
    num_stages: int,
    lam: float,
    eps: float,
    rho: Optional[float] = None,
    iters: int = 30,
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    """位置-only 传输计划 π_pos = OT(λ·D)（**内容成本置零**），返回 [T,K]（fp32）。

    与 `ot_stage_scores` 同解算器、同 (ε, ρ, iters)，但成本只含 λ·位置先验 D，无内容项。
    因不依赖 (q,c)，对所有查询/类**恒相同**——作为"纯几何软窗口"对照基线：
    内容自适应量 = ‖π_full − π_pos‖₁（见 `plan_l1`）。λ=0 时 π_pos 即均匀耦合。
    """
    cost = lam * build_D(num_frames, num_stages, device=device)   # [T,K]，无内容
    return sinkhorn_log(cost, eps=eps, rho=rho, iters=iters)      # [T,K]


def row_conditional_entropy(plan: Tensor) -> Tensor:
    """逐帧条件分配熵 H(π_i,·/行和)/log K，对帧轴取均值，返回 [...]∈[0,1]（detach）。

    1=每帧在阶段上均匀分配（无锐度）；0=硬分配。比全局 `pi_entropy_norm` 干净：
    平衡/单侧不平衡下帧边际恒被硬约束为均匀，全局 H(π)/log(TK) 有下限
    ≈logT/(logT+logK)（帧轴永远贡献 logT），会把"逐帧其实近均匀"误读成"熵未满"。
    本量除以 logK 隔离 K 路分配锐度，λ↑ 应下降。
    """
    p = plan.detach()
    num_stages = p.shape[-1]
    row = p.sum(dim=-1, keepdim=True).clamp_min(1e-12)            # [...,T,1] 帧侧硬边际
    cond = p / row                                               # 每帧对阶段的条件分布
    ent = -(cond.clamp_min(1e-12) * cond.clamp_min(1e-12).log()).sum(dim=-1)  # [...,T]
    ent_norm = ent / math.log(num_stages) if num_stages > 1 else ent * 0.0
    return ent_norm.mean(dim=-1)                                 # [...]


def band_mass(plan: Tensor, bandwidth: int = 0) -> Tensor:
    """位置先验最近对角带的行归一化质量占比，对帧轴取均值，返回 [...]∈[0,1]（detach）。

    每帧取 D(i,·) 最近阶段（bandwidth=0）或其 ±bandwidth 邻域为"带"，量 = 带内质量/行和。
    λ↑（位置先验增强）应上升；用于确认 λ 把 plan 拉向带状对角（对照单测 U5b）。
    """
    p = plan.detach()
    num_frames, num_stages = p.shape[-2], p.shape[-1]
    nearest = build_D(num_frames, num_stages, device=p.device).argmin(dim=1)  # [T]
    stages = torch.arange(num_stages, device=p.device)
    mask = (stages[None, :] - nearest[:, None]).abs() <= bandwidth            # [T,K] bool
    row = p.sum(dim=-1).clamp_min(1e-12)                         # [...,T]
    band = (p * mask.to(p.dtype)).sum(dim=-1)                   # [...,T]
    return (band / row).mean(dim=-1)                            # [...]


def plan_l1(pi_full: Tensor, pi_pos: Tensor) -> Tensor:
    """内容自适应量：逐前导维 ‖π_full − π_pos‖₁（π_pos [T,K] 广播），返回 [...]（detach）。

    →0：π_full 退化为纯几何软窗口（内容不塑形传输，创新点 1 失去意义）；
    显著>数值噪声：内容参与塑形。是步 B 放行的核心门控量之一。
    """
    pf = pi_full.detach()
    pp = pi_pos.detach()
    while pp.ndim < pf.ndim:
        pp = pp.unsqueeze(0)
    return (pf - pp).abs().sum(dim=(-2, -1))                    # [...] 逐 (q,c)


def plan_pairwise_l1(plan: Tensor, dim: int) -> Tensor:
    """沿 `dim` 轴的平均成对 L1（度量 plan 随该轴内容变化的幅度），detach。

    plan [NQ,C,T,K]：dim=1（跨类，同 query 不同类）→[NQ]；dim=0（跨 query，同类不同
    query）→[C]。π_pos 对二者恒为 0（纯几何无内容）——故显著>0 即证内容自适应。
    """
    p = plan.detach()
    n = p.shape[dim]
    if n < 2:
        return p.new_zeros(p.shape[2:-2] if p.ndim > 4 else ())
    pm = p.movedim(dim, 0)                                      # [n, ..., T, K]
    diffs = [ (pm[i] - pm[j]).abs().sum(dim=(-2, -1))
              for i in range(n) for j in range(i + 1, n) ]
    return torch.stack(diffs, dim=0).mean(dim=0)               # [...]


__all__ = [
    "build_D", "sinkhorn_log", "ot_stage_scores", "ot_plan_stats",
    "ot_position_only_plan", "row_conditional_entropy", "band_mass",
    "plan_l1", "plan_pairwise_l1",
]
