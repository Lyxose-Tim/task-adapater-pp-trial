"""Order-aware, one-sided unbalanced optimal transport for stage matching.

The transport problems are deliberately tiny (normally ``7 x K``), so the
solver keeps the Sinkhorn iterations in the autograd graph.  All numerical
work is performed in float32 even when the surrounding CLIP model runs in
mixed precision.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import ContextManager

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass(frozen=True)
class OTAlignment:
    """Outputs for every query/class transport problem.

    Attributes:
        plan: Transport plans with shape ``[Q,C,T,K]``.
        mass: Per-stage transported mass with shape ``[Q,C,K]``.
        fhat: Transported, normalized visual stages ``[Q,C,K,D]``.
        score: Mass-weighted semantic similarity ``[Q,C]``.
        cost: Content plus positional cost ``[Q,C,T,K]``.
        transport_cost: Cost averaged using each plan, shape ``[Q,C]``.
        row_residual: Maximum hard frame-marginal error for each pair.
        column_residual: Maximum error from the uniform stage target.  For
            finite ``rho`` this is diagnostic only because that marginal is
            intentionally soft.
        balanced: Whether the explicit ``rho=inf`` branch was used.
    """

    plan: Tensor
    mass: Tensor
    fhat: Tensor
    score: Tensor
    cost: Tensor
    transport_cost: Tensor
    row_residual: Tensor
    column_residual: Tensor
    balanced: bool


def position_distance(
    num_frames: int,
    num_stages: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Return the squared normalized temporal distance ``[T,K]``."""

    if num_frames < 1 or num_stages < 1:
        raise ValueError("num_frames and num_stages must be positive")
    frame_position = torch.linspace(0.0, 1.0, num_frames, device=device, dtype=torch.float32)
    stage_position = torch.linspace(0.0, 1.0, num_stages, device=device, dtype=torch.float32)
    return (frame_position[:, None] - stage_position[None, :]).square()


def _autocast_disabled(device: torch.device) -> ContextManager[object]:
    # CUDA autocast can otherwise cast a float32 einsum back to fp16.  Older
    # torch releases do not expose autocast for every accelerator backend.
    if device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def sinkhorn_log(
    cost: Tensor,
    *,
    epsilon: float = 0.05,
    rho: float = math.inf,
    iterations: int = 30,
) -> Tensor:
    """Solve batched one-sided unbalanced OT in the log domain.

    ``cost`` may have arbitrary batch dimensions followed by ``[T,K]``.
    The frame marginal is hard and uniform.  The uniform stage marginal is
    hard only when ``rho=inf``; otherwise its dual update is damped by
    ``tau=rho/(rho+epsilon)``.  The returned plan is float32 and its rows are
    explicitly projected once more after the final stage update.
    """

    if cost.ndim < 2:
        raise ValueError("cost must end in [T,K]")
    if not cost.is_floating_point():
        raise TypeError("cost must be floating point")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not math.isinf(rho) and (rho <= 0 or not math.isfinite(rho)):
        raise ValueError("rho must be positive or infinity")

    cost32 = cost.float()
    num_frames, num_stages = cost32.shape[-2:]
    if min(num_frames, num_stages) < 1:
        raise ValueError("transport axes cannot be empty")

    log_a = cost32.new_full((num_frames,), -math.log(num_frames))
    log_b = cost32.new_full((num_stages,), -math.log(num_stages))

    # Entropic reference is a (uniform) a x b product measure.
    log_kernel = (
        -cost32 / float(epsilon)
        + log_a.view(*([1] * (cost32.ndim - 2)), num_frames, 1)
        + log_b.view(*([1] * (cost32.ndim - 2)), 1, num_stages)
    )
    log_u = torch.zeros_like(cost32[..., :, 0])
    log_v = torch.zeros_like(cost32[..., 0, :])
    balanced = math.isinf(rho)
    tau = 1.0 if balanced else float(rho) / (float(rho) + float(epsilon))

    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        column_log_mass = torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
        # tau=1 is the conventional balanced Sinkhorn update.  A finite rho
        # relaxes only the stage marginal while leaving the row update hard.
        log_v = tau * (log_b - column_log_mass)

    # The last operation in the loop updates v.  Recompute u so the returned
    # plan satisfies its promised hard frame marginal to numerical precision.
    log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
    log_plan = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    return torch.exp(log_plan)


class OrderAwareOptimalTransport(nn.Module):
    """Semantic-conditioned temporal matching with a soft stage marginal."""

    def __init__(
        self,
        *,
        epsilon: float = 0.05,
        lambda_pos: float = 0.0,
        rho: float = math.inf,
        iterations: int = 30,
        mass_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if lambda_pos < 0 or not math.isfinite(lambda_pos):
            raise ValueError("lambda_pos must be finite and non-negative")
        if mass_epsilon <= 0 or not math.isfinite(mass_epsilon):
            raise ValueError("mass_epsilon must be finite and positive")
        # Validate the remaining scalars once through the public solver rules.
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be finite and positive")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if not math.isinf(rho) and (rho <= 0 or not math.isfinite(rho)):
            raise ValueError("rho must be positive or infinity")
        self.epsilon = float(epsilon)
        self.lambda_pos = float(lambda_pos)
        self.rho = float(rho)
        self.iterations = int(iterations)
        self.mass_epsilon = float(mass_epsilon)

    @property
    def balanced(self) -> bool:
        """Whether both uniform marginals are enforced."""

        return math.isinf(self.rho)

    def forward(self, aligned_frames: Tensor, stages: Tensor) -> OTAlignment:
        """Align ``[Q,T,D]`` frames to ``[C,K,D]`` stage descriptions."""

        if aligned_frames.ndim != 3:
            raise ValueError("aligned_frames must have shape [Q,T,D]")
        if stages.ndim != 3:
            raise ValueError("stages must have shape [C,K,D]")
        if aligned_frames.shape[-1] != stages.shape[-1]:
            raise ValueError("frame and stage embedding dimensions must match")
        if aligned_frames.device != stages.device:
            raise ValueError("aligned_frames and stages must be on the same device")
        if not aligned_frames.is_floating_point() or not stages.is_floating_point():
            raise TypeError("aligned_frames and stages must be floating point")

        with _autocast_disabled(aligned_frames.device):
            frames = F.normalize(aligned_frames.float(), dim=-1)
            text = F.normalize(stages.float(), dim=-1)
            content_cost = 1.0 - torch.einsum("qtd,ckd->qctk", frames, text)
            positional = position_distance(
                frames.shape[1], text.shape[1], device=frames.device
            )
            cost = content_cost + self.lambda_pos * positional[None, None]
            plan = sinkhorn_log(
                cost,
                epsilon=self.epsilon,
                rho=self.rho,
                iterations=self.iterations,
            )

            mass = plan.sum(dim=-2)
            numerator = torch.einsum("qctk,qtd->qckd", plan, frames)
            fhat = F.normalize(
                numerator / mass.unsqueeze(-1).clamp_min(self.mass_epsilon),
                dim=-1,
            )
            stage_similarity = torch.einsum("ckd,qckd->qck", text, fhat)
            weights = mass / mass.sum(dim=-1, keepdim=True).clamp_min(self.mass_epsilon)
            score = (weights * stage_similarity).sum(dim=-1)
            transport_cost = (plan * cost).sum(dim=(-2, -1))

            target_a = plan.new_full((frames.shape[1],), 1.0 / frames.shape[1])
            target_b = plan.new_full((text.shape[1],), 1.0 / text.shape[1])
            row_residual = (plan.sum(dim=-1) - target_a).abs().amax(dim=-1)
            column_residual = (mass - target_b).abs().amax(dim=-1)

        return OTAlignment(
            plan=plan,
            mass=mass,
            fhat=fhat,
            score=score,
            cost=cost,
            transport_cost=transport_cost,
            row_residual=row_residual,
            column_residual=column_residual,
            balanced=self.balanced,
        )


__all__ = [
    "OTAlignment",
    "OrderAwareOptimalTransport",
    "position_distance",
    "sinkhorn_log",
]
