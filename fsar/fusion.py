"""Centralised score fusion and cached-score pseudo validation.

The released Task-Adapter++ code multiplies two cosine-score matrices directly.
This module keeps that operation as an exact, explicit baseline and adds the two
alpha parameterisations from innovation 4a.  Pseudo validation intentionally
accepts *scores*, not a model or images: the alpha sweep therefore cannot cause
an accidental model forward for every grid point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Sequence, Tuple

import torch
from torch import Tensor


FusionMode = Literal["legacy_product", "probability_product", "logit_convex"]


def _validate_branch_scores(visual_scores: Tensor, semantic_scores: Tensor) -> None:
    if not torch.is_tensor(visual_scores) or not torch.is_tensor(semantic_scores):
        raise TypeError("visual_scores and semantic_scores must be tensors")
    if visual_scores.shape != semantic_scores.shape:
        raise ValueError(
            "visual and semantic scores must have the same shape, got "
            f"{tuple(visual_scores.shape)} and {tuple(semantic_scores.shape)}"
        )
    if visual_scores.ndim < 2:
        raise ValueError("branch scores must have a class dimension and at least one sample dimension")
    if visual_scores.shape[-1] < 2:
        raise ValueError("fusion requires at least two classes")
    if not visual_scores.is_floating_point() or not semantic_scores.is_floating_point():
        raise TypeError("branch scores must be floating-point tensors")
    if not bool(torch.isfinite(visual_scores).all()) or not bool(torch.isfinite(semantic_scores).all()):
        raise ValueError("branch scores must contain only finite values")


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {value}")
    return value


def _validate_temperature(value: float, name: str) -> float:
    value = float(value)
    if not value > 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def fuse_scores(
    visual_scores: Tensor,
    semantic_scores: Tensor,
    *,
    mode: FusionMode = "legacy_product",
    alpha: float = 0.5,
    visual_temperature: float = 1.0,
    semantic_temperature: float = 1.0,
    normalize_probability: bool = True,
) -> Tensor:
    """Fuse cached ``[..., classes]`` branch scores.

    ``legacy_product`` is the paper/released-code baseline and is bit-for-bit
    the expression ``visual_scores * semantic_scores``.  It deliberately does
    not use alpha or temperatures.

    ``probability_product`` computes

    ``p_vis ** (2*alpha) * p_sem ** (2*(1-alpha))``.

    By default the proportional scores are row-normalised.  Passing
    ``normalize_probability=False`` returns the raw product; normalisation is a
    positive common factor per sample and therefore cannot change argmax.

    ``logit_convex`` is the additive reference
    ``2*alpha*visual + 2*(1-alpha)*semantic``.
    """

    _validate_branch_scores(visual_scores, semantic_scores)
    if mode == "legacy_product":
        # Preserve this literal operation: it is the compatibility contract.
        return visual_scores * semantic_scores

    alpha = _validate_alpha(alpha)
    if mode == "logit_convex":
        return 2.0 * alpha * visual_scores + 2.0 * (1.0 - alpha) * semantic_scores
    if mode != "probability_product":
        raise ValueError(f"unknown fusion mode: {mode!r}")

    visual_temperature = _validate_temperature(visual_temperature, "visual_temperature")
    semantic_temperature = _validate_temperature(semantic_temperature, "semantic_temperature")

    # Work in log space so endpoints (an exponent of zero) and sharply peaked
    # distributions stay finite.  softmax below is exactly the normalisation of
    # the proportional probability product.
    log_visual = torch.log_softmax(visual_scores / visual_temperature, dim=-1)
    log_semantic = torch.log_softmax(semantic_scores / semantic_temperature, dim=-1)
    log_fused = 2.0 * alpha * log_visual + 2.0 * (1.0 - alpha) * log_semantic
    if normalize_probability:
        return torch.softmax(log_fused, dim=-1)
    return torch.exp(log_fused)


def fuse_global_alpha(
    visual_scores: Tensor,
    semantic_scores: Tensor,
    alpha: float,
    *,
    mode: FusionMode = "probability_product",
    visual_temperature: float = 1.0,
    semantic_temperature: float = 1.0,
) -> Tensor:
    """Apply a dataset-level validation-selected alpha to cached scores."""

    if mode == "legacy_product":
        raise ValueError("global alpha is undefined for legacy_product")
    return fuse_scores(
        visual_scores,
        semantic_scores,
        mode=mode,
        alpha=alpha,
        visual_temperature=visual_temperature,
        semantic_temperature=semantic_temperature,
    )


def _correct_class_margin(scores: Tensor, targets: Tensor) -> Tensor:
    correct = scores.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    competitors = scores.masked_fill(
        torch.nn.functional.one_hot(targets, scores.shape[-1]).to(dtype=torch.bool),
        float("-inf"),
    )
    return correct - competitors.max(dim=-1).values


@dataclass(frozen=True)
class AlphaCandidate:
    """One cached-score alpha-grid measurement."""

    alpha: float
    accuracy: float
    margin: float

    def as_dict(self) -> Dict[str, float]:
        return {"alpha": self.alpha, "accuracy": self.accuracy, "margin": self.margin}


@dataclass(frozen=True)
class AlphaSelection:
    """Selection result plus evidence needed for experiment ledgers."""

    alpha: float
    used_fallback: bool
    reason: str
    global_alpha: float
    mode: str
    candidates: Tuple[AlphaCandidate, ...]
    samples: int
    model_forward_calls_during_grid: int = 0
    used_cached_scores: bool = True

    @property
    def best_accuracy(self) -> float:
        selected = min(self.candidates, key=lambda item: abs(item.alpha - self.alpha))
        return selected.accuracy

    @property
    def best_margin(self) -> float:
        selected = min(self.candidates, key=lambda item: abs(item.alpha - self.alpha))
        return selected.margin

    def as_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "used_fallback": self.used_fallback,
            "reason": self.reason,
            "global_alpha": self.global_alpha,
            "mode": self.mode,
            "samples": self.samples,
            "model_forward_calls_during_grid": self.model_forward_calls_during_grid,
            "used_cached_scores": self.used_cached_scores,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def _alpha_grid(values: Iterable[float]) -> Tuple[float, ...]:
    grid = tuple(_validate_alpha(value) for value in values)
    if not grid:
        raise ValueError("alpha_grid must not be empty")
    if len(set(grid)) != len(grid):
        raise ValueError("alpha_grid must not contain duplicates")
    return tuple(sorted(grid))


def select_alpha_from_cached_scores(
    visual_scores: Tensor,
    semantic_scores: Tensor,
    targets: Tensor,
    *,
    alpha_grid: Sequence[float],
    global_alpha: float,
    mode: FusionMode = "probability_product",
    visual_temperature: float = 1.0,
    semantic_temperature: float = 1.0,
    flat_tolerance: float = 1e-12,
) -> AlphaSelection:
    """Select an episode alpha from cached augmented-support-view scores.

    Inputs may be ``[views, support, classes]`` or any other shape whose final
    axis is class.  Targets must match all leading score dimensions.  The
    selection maximises pseudo accuracy, then labelled correct-class margin.
    Any residual tie is resolved deterministically by proximity to the global
    alpha and then by smaller alpha.  If *both* accuracy and margin curves are
    flat, the prescribed global-alpha fallback is used.

    The tensor-only signature is an intentional cost/safety property: no model,
    callable, image, or augmentation is accepted, so the grid performs zero
    model forwards.
    """

    _validate_branch_scores(visual_scores, semantic_scores)
    if mode == "legacy_product":
        raise ValueError("alpha selection requires an alpha-dependent fusion mode")
    if targets.shape != visual_scores.shape[:-1]:
        raise ValueError(
            f"targets shape must be {tuple(visual_scores.shape[:-1])}, got {tuple(targets.shape)}"
        )
    if not targets.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("targets must use an integer dtype")
    targets = targets.to(device=visual_scores.device, dtype=torch.long)
    if targets.numel() == 0:
        raise ValueError("pseudo validation requires at least one cached sample")
    if int(targets.min()) < 0 or int(targets.max()) >= visual_scores.shape[-1]:
        raise ValueError("targets contain a class index outside the score matrix")
    if float(flat_tolerance) < 0:
        raise ValueError("flat_tolerance must be non-negative")

    grid = _alpha_grid(alpha_grid)
    global_alpha = _validate_alpha(global_alpha)
    candidate_rows = []
    with torch.no_grad():
        for alpha in grid:
            scores = fuse_scores(
                visual_scores,
                semantic_scores,
                mode=mode,
                alpha=alpha,
                visual_temperature=visual_temperature,
                semantic_temperature=semantic_temperature,
            )
            accuracy = (scores.argmax(dim=-1) == targets).float().mean()
            margin = _correct_class_margin(scores, targets).mean()
            candidate_rows.append(AlphaCandidate(alpha, float(accuracy), float(margin)))
    candidates = tuple(candidate_rows)

    accuracies = [candidate.accuracy for candidate in candidates]
    margins = [candidate.margin for candidate in candidates]
    is_flat = (
        max(accuracies) - min(accuracies) <= flat_tolerance
        and max(margins) - min(margins) <= flat_tolerance
    )
    if is_flat:
        chosen = global_alpha
        fallback = True
        reason = "flat_curve_global_fallback"
    else:
        best_accuracy = max(accuracies)
        accuracy_ties = [
            candidate
            for candidate in candidates
            if abs(candidate.accuracy - best_accuracy) <= flat_tolerance
        ]
        best_margin = max(candidate.margin for candidate in accuracy_ties)
        margin_ties = [
            candidate
            for candidate in accuracy_ties
            if abs(candidate.margin - best_margin) <= flat_tolerance
        ]
        chosen = min(margin_ties, key=lambda item: (abs(item.alpha - global_alpha), item.alpha)).alpha
        fallback = False
        reason = "max_accuracy_then_margin"

    return AlphaSelection(
        alpha=chosen,
        used_fallback=fallback,
        reason=reason,
        global_alpha=global_alpha,
        mode=mode,
        candidates=candidates,
        samples=targets.numel(),
    )


@dataclass(frozen=True)
class KIPSelection:
    """Per-sample hard branch gate and its stability evidence."""

    scores: Tensor
    choose_visual: Tensor
    visual_stability: Tensor
    semantic_stability: Tensor

    @property
    def visual_selection_rate(self) -> float:
        return float(self.choose_visual.float().mean())


def _probability_stability(probabilities: Tensor, metric: str) -> Tensor:
    if metric == "margin":
        top_two = probabilities.topk(2, dim=-1).values
        return top_two[..., 0] - top_two[..., 1]
    if metric == "negative_entropy":
        eps = torch.finfo(probabilities.dtype).tiny
        return (probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
    raise ValueError("stability metric must be 'margin' or 'negative_entropy'")


def kip_hard_gate(
    visual_scores: Tensor,
    semantic_scores: Tensor,
    *,
    visual_temperature: float = 1.0,
    semantic_temperature: float = 1.0,
    stability: Literal["margin", "negative_entropy"] = "margin",
) -> KIPSelection:
    """Training-free KIP-style baseline: choose the stabler branch per sample.

    Stability is measured on calibrated class probabilities.  A deterministic
    equality tie selects the visual branch.  Returned scores are probabilities,
    which makes the two branch scales comparable before the hard selection.
    """

    _validate_branch_scores(visual_scores, semantic_scores)
    visual_temperature = _validate_temperature(visual_temperature, "visual_temperature")
    semantic_temperature = _validate_temperature(semantic_temperature, "semantic_temperature")
    visual_probability = torch.softmax(visual_scores / visual_temperature, dim=-1)
    semantic_probability = torch.softmax(semantic_scores / semantic_temperature, dim=-1)
    visual_stability = _probability_stability(visual_probability, stability)
    semantic_stability = _probability_stability(semantic_probability, stability)
    choose_visual = visual_stability >= semantic_stability
    scores = torch.where(choose_visual.unsqueeze(-1), visual_probability, semantic_probability)
    return KIPSelection(scores, choose_visual, visual_stability, semantic_stability)


__all__ = [
    "AlphaCandidate",
    "AlphaSelection",
    "FusionMode",
    "KIPSelection",
    "fuse_global_alpha",
    "fuse_scores",
    "kip_hard_gate",
    "select_alpha_from_cached_scores",
]
