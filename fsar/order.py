"""Order-aware scoring, regularisation, and episode perturbations.

The original Task-Adapter++ semantic branch assigns three text stages to the
fixed aligned-query windows ``[0:3]``, ``[2:5]``, and ``[4:7]``.  This module
keeps that assignment explicit: permuting text stages never permutes the
visual windows along with them.  That distinction is essential for measuring
and training order sensitivity.
"""

from __future__ import annotations

from contextlib import nullcontext
from itertools import permutations
from typing import ContextManager, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Permutation = Tuple[int, ...]

IDENTITY_PERMUTATION: Permutation = (0, 1, 2)
STAGE_PERMUTATIONS: Tuple[Permutation, ...] = tuple(permutations(range(3)))
NON_IDENTITY_STAGE_PERMUTATIONS: Tuple[Permutation, ...] = tuple(
    permutation
    for permutation in STAGE_PERMUTATIONS
    if permutation != IDENTITY_PERMUTATION
)


def _autocast_disabled(device: torch.device) -> ContextManager[object]:
    """Keep fixed-window cosine scoring in fp32 when its caller uses AMP."""

    if device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


def stage_permutations(
    num_stages: int = 3,
    include_identity: bool = True,
) -> Tuple[Permutation, ...]:
    """Return stage permutations in deterministic lexicographic order.

    For the three-stage protocol this returns the six experimental
    permutations, with ``(0, 1, 2)`` first.
    """

    if isinstance(num_stages, bool) or not isinstance(num_stages, int):
        raise TypeError("num_stages must be an integer")
    if num_stages < 1:
        raise ValueError("num_stages must be positive")

    result = tuple(permutations(range(num_stages)))
    if include_identity:
        return result
    identity = tuple(range(num_stages))
    return tuple(permutation for permutation in result if permutation != identity)


def _validated_permutation(
    permutation: Sequence[int],
    size: int,
    name: str = "permutation",
) -> Permutation:
    try:
        result = tuple(int(index) for index in permutation)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a sequence of integer indices") from exc
    if len(result) != size or sorted(result) != list(range(size)):
        raise ValueError(f"{name} must contain every index in [0, {size}) exactly once")
    return result


def permute_stage_text(
    stage_text: torch.Tensor,
    permutation: Sequence[int],
) -> torch.Tensor:
    """Permute the stage axis of text features shaped ``[K, C, D]``."""

    if not isinstance(stage_text, torch.Tensor):
        raise TypeError("stage_text must be a torch.Tensor")
    if stage_text.ndim != 3:
        raise ValueError("stage_text must have shape [K, C, D]")
    order = _validated_permutation(permutation, stage_text.shape[0])
    index = torch.tensor(order, dtype=torch.long, device=stage_text.device)
    return stage_text.index_select(0, index)


def fixed_stage_semantic_scores(
    aligned_query: torch.Tensor,
    stage_text: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Score fixed stage-to-window semantic alignment.

    Args:
        aligned_query: Aligned visual features shaped ``[T, Q, D]``.
        stage_text: Stage/class text features shaped ``[K, C, D]``.
        eps: Epsilon used by cosine normalisation.

    Returns:
        Query-by-class scores shaped ``[Q, C]``.

    ``T`` must equal ``2*K + 1``.  Stage ``k`` is compared with the three
    aligned-query positions beginning at ``2*k``.  The final score is the mean
    of all ``3*K`` cosine similarities, matching the three-stage, seven-step
    Task-Adapter++ equation (division by nine).
    """

    if not isinstance(aligned_query, torch.Tensor):
        raise TypeError("aligned_query must be a torch.Tensor")
    if not isinstance(stage_text, torch.Tensor):
        raise TypeError("stage_text must be a torch.Tensor")
    if aligned_query.ndim != 3:
        raise ValueError("aligned_query must have shape [T, Q, D]")
    if stage_text.ndim != 3:
        raise ValueError("stage_text must have shape [K, C, D]")
    if aligned_query.shape[-1] != stage_text.shape[-1]:
        raise ValueError("aligned_query and stage_text feature dimensions must match")
    if aligned_query.device != stage_text.device:
        raise ValueError("aligned_query and stage_text must be on the same device")
    if not aligned_query.is_floating_point() or not stage_text.is_floating_point():
        raise TypeError("aligned_query and stage_text must have floating-point dtypes")
    if eps <= 0:
        raise ValueError("eps must be positive")

    num_steps, num_queries, _ = aligned_query.shape
    num_stages, num_classes, _ = stage_text.shape
    if num_stages < 1 or num_classes < 1 or num_queries < 1:
        raise ValueError("stage_text and aligned_query dimensions must be non-empty")
    expected_steps = 2 * num_stages + 1
    if num_steps != expected_steps:
        raise ValueError(
            f"aligned_query has T={num_steps}; fixed {num_stages}-stage scoring "
            f"requires T={expected_steps}"
        )

    # CLIP's visual path commonly emits fp16 while the text adapter may emit
    # fp32.  Cosine scoring is a small numerical island, so promote both sides
    # explicitly rather than relying on einsum dtype coercion (which it does
    # not perform on CUDA).
    with _autocast_disabled(aligned_query.device):
        query = F.normalize(aligned_query.float(), dim=-1, eps=eps)
        text = F.normalize(stage_text.float(), dim=-1, eps=eps)
        score = query.new_zeros((num_queries, num_classes))
        for stage_index in range(num_stages):
            window = query[2 * stage_index : 2 * stage_index + 3]
            score = score + torch.einsum(
                "tqd,cd->qc", window, text[stage_index]
            )
        return score / float(3 * num_stages)


def all_permutation_scores(
    aligned_query: torch.Tensor,
    stage_text: torch.Tensor,
    orders: Optional[Iterable[Sequence[int]]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return fixed-window scores for every requested text-stage order.

    The result has shape ``[P, Q, C]``.  With three stages and the default
    ``orders``, ``P`` is six and index zero is the identity order.
    """

    if stage_text.ndim != 3:
        raise ValueError("stage_text must have shape [K, C, D]")
    selected_orders = (
        stage_permutations(stage_text.shape[0]) if orders is None else tuple(orders)
    )
    if not selected_orders:
        raise ValueError("orders must contain at least one permutation")
    scores = [
        fixed_stage_semantic_scores(
            aligned_query,
            permute_stage_text(stage_text, order),
            eps=eps,
        )
        for order in selected_orders
    ]
    return torch.stack(scores, dim=0)


def correct_class_order_scores(
    permutation_scores: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Select correct-class scores and return them as ``[Q, P]``.

    Args:
        permutation_scores: Scores shaped ``[P, Q, C]``.
        targets: Integer class indices shaped ``[Q]``.
    """

    if permutation_scores.ndim != 3:
        raise ValueError("permutation_scores must have shape [P, Q, C]")
    if targets.ndim != 1 or targets.shape[0] != permutation_scores.shape[1]:
        raise ValueError("targets must have shape [Q]")
    if targets.dtype == torch.bool or targets.is_floating_point():
        raise TypeError("targets must contain integer class indices")
    targets = targets.to(device=permutation_scores.device, dtype=torch.long)
    if bool(((targets < 0) | (targets >= permutation_scores.shape[2])).any()):
        raise ValueError("targets contain an out-of-range class index")
    gather_index = targets.view(1, -1, 1).expand(
        permutation_scores.shape[0], -1, 1
    )
    return permutation_scores.gather(2, gather_index).squeeze(2).transpose(0, 1)


class OrderContrastiveLoss(nn.Module):
    """Contrast the identity-order score against non-identity orders.

    ``positive_scores`` may have any shape ``S`` and ``negative_scores`` must
    have shape ``S + [M]``, where the final dimension contains negative text
    permutations.  Both ``margin`` and ``infonce`` modes return one scalar by
    default.  With ``reduction='none'`` they return one loss per element of
    ``S`` (the margin loss is averaged over its negative orders first).
    """

    def __init__(
        self,
        mode: str = "margin",
        margin: float = 0.1,
        temperature: float = 0.07,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        normalised_mode = str(mode).lower().replace("_", "")
        if normalised_mode not in {"margin", "infonce"}:
            raise ValueError("mode must be 'margin' or 'infonce'")
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.mode = normalised_mode
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.reduction = reduction

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()

    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(positive_scores, torch.Tensor) or not isinstance(
            negative_scores, torch.Tensor
        ):
            raise TypeError("positive_scores and negative_scores must be tensors")
        if negative_scores.ndim != positive_scores.ndim + 1:
            raise ValueError(
                "negative_scores must append a negative-order dimension to "
                "positive_scores"
            )
        if negative_scores.shape[:-1] != positive_scores.shape:
            raise ValueError(
                "negative_scores must have shape positive_scores.shape + [M]"
            )
        if negative_scores.shape[-1] < 1:
            raise ValueError("at least one negative order is required")
        if positive_scores.device != negative_scores.device:
            raise ValueError("positive_scores and negative_scores must share a device")
        if not positive_scores.is_floating_point() or not negative_scores.is_floating_point():
            raise TypeError("positive_scores and negative_scores must be floating point")

        if self.mode == "margin":
            per_negative = F.relu(
                self.margin - positive_scores.unsqueeze(-1) + negative_scores
            )
            per_item = per_negative.mean(dim=-1)
        else:
            logits = torch.cat(
                (positive_scores.unsqueeze(-1), negative_scores), dim=-1
            ) / self.temperature
            per_item = torch.logsumexp(logits, dim=-1) - logits[..., 0]
        return self._reduce(per_item)

    def from_permutation_scores(
        self,
        permutation_scores: torch.Tensor,
        targets: torch.Tensor,
        identity_index: int = 0,
    ) -> torch.Tensor:
        """Compute loss directly from ``[P, Q, C]`` permutation scores."""

        selected = correct_class_order_scores(permutation_scores, targets)
        num_orders = selected.shape[-1]
        if not 0 <= identity_index < num_orders:
            raise ValueError("identity_index is out of range")
        negative_indices = [
            index for index in range(num_orders) if index != identity_index
        ]
        if not negative_indices:
            raise ValueError("at least two permutation orders are required")
        return self(
            selected[..., identity_index],
            selected[..., negative_indices],
        )


def make_frame_permutation(
    num_frames: int,
    mode: str = "identity",
    seed: Optional[int] = None,
) -> Permutation:
    """Create a reproducible identity, reverse, or random frame order."""

    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        raise TypeError("num_frames must be an integer")
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    mode = str(mode).lower()
    if mode == "identity":
        return tuple(range(num_frames))
    if mode == "reverse":
        return tuple(range(num_frames - 1, -1, -1))
    if mode != "random":
        raise ValueError("mode must be 'identity', 'reverse', or 'random'")
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    return tuple(torch.randperm(num_frames, generator=generator).tolist())


def permute_query_frames(
    episode: torch.Tensor,
    permutation: Sequence[int],
    n_support: int,
    sample_dim: int = 1,
    time_dim: int = 2,
) -> torch.Tensor:
    """Apply one synchronous frame permutation to query clips only.

    The typical input layout is ``[way, support+query, time, ...]``.  The first
    ``n_support`` items on ``sample_dim`` are returned unchanged, while every
    remaining query clip uses exactly the same frame indices.  ``time_dim``
    can be set to 3 for the repository's ``[way, sample, C, T, H, W]`` layout.
    The operation is functional (not in-place) and preserves autograd.
    """

    if not isinstance(episode, torch.Tensor):
        raise TypeError("episode must be a torch.Tensor")
    if episode.ndim < 3:
        raise ValueError("episode must have at least three dimensions")
    sample_dim = sample_dim % episode.ndim
    time_dim = time_dim % episode.ndim
    if sample_dim == time_dim:
        raise ValueError("sample_dim and time_dim must be different")
    if isinstance(n_support, bool) or not isinstance(n_support, int):
        raise TypeError("n_support must be an integer")
    num_samples = episode.shape[sample_dim]
    if not 0 <= n_support <= num_samples:
        raise ValueError("n_support must be between zero and the sample count")
    order = _validated_permutation(
        permutation, episode.shape[time_dim], name="frame permutation"
    )
    index = torch.tensor(order, dtype=torch.long, device=episode.device)

    support = episode.narrow(sample_dim, 0, n_support)
    query = episode.narrow(sample_dim, n_support, num_samples - n_support)
    query = query.index_select(time_dim, index)
    return torch.cat((support, query), dim=sample_dim)


# Descriptive aliases retained for experiment scripts.
score_fixed_stage_semantics = fixed_stage_semantic_scores
score_stage_permutations = all_permutation_scores
synchronous_query_frame_permutation = permute_query_frames


__all__ = [
    "IDENTITY_PERMUTATION",
    "NON_IDENTITY_STAGE_PERMUTATIONS",
    "OrderContrastiveLoss",
    "STAGE_PERMUTATIONS",
    "all_permutation_scores",
    "correct_class_order_scores",
    "fixed_stage_semantic_scores",
    "make_frame_permutation",
    "permute_query_frames",
    "permute_stage_text",
    "score_fixed_stage_semantics",
    "score_stage_permutations",
    "stage_permutations",
    "synchronous_query_frame_permutation",
]
