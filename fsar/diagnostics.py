"""Statistical accumulation for paired order-sensitivity diagnostics."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


CI95_Z = 1.959963984540054


@dataclass(frozen=True)
class MeanCI:
    """A sample mean and its normal-approximation 95% confidence interval."""

    mean: float
    ci95: float
    low: float
    high: float
    n: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "ci95": self.ci95,
            "ci95_low": self.low,
            "ci95_high": self.high,
            "n": self.n,
        }


def _finite_1d(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    else:
        array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def summarize_mean_ci95(values: Iterable[float]) -> MeanCI:
    """Summarize a sample with mean and two-sided 95% CI.

    The half-width is ``1.96 * sample_std / sqrt(n)``.  A singleton sample has
    zero half-width because its sample variance is undefined.
    """

    array = _finite_1d(values, "values")
    mean = float(array.mean())
    half_width = 0.0
    if array.size > 1:
        half_width = float(CI95_Z * array.std(ddof=1) / np.sqrt(array.size))
    return MeanCI(
        mean=mean,
        ci95=half_width,
        low=mean - half_width,
        high=mean + half_width,
        n=int(array.size),
    )


def mean_ci95(values: Iterable[float]) -> Tuple[float, float]:
    """Return ``(mean, 95% CI half-width)`` for compatibility with logs."""

    result = summarize_mean_ci95(values)
    return result.mean, result.ci95


def summarize_paired_mean_ci95(
    reference: Iterable[float],
    comparison: Iterable[float],
) -> MeanCI:
    """Summarize paired differences ``reference - comparison``."""

    reference_array = _finite_1d(reference, "reference")
    comparison_array = _finite_1d(comparison, "comparison")
    if reference_array.shape != comparison_array.shape:
        raise ValueError("reference and comparison must contain equally many pairs")
    return summarize_mean_ci95(reference_array - comparison_array)


def paired_mean_ci95(
    reference: Iterable[float],
    comparison: Iterable[float],
) -> Tuple[float, float]:
    """Return paired ``reference - comparison`` mean and 95% CI half-width."""

    result = summarize_paired_mean_ci95(reference, comparison)
    return result.mean, result.ci95


# A concise alias for callers that already state the confidence level.
paired_mean_ci = paired_mean_ci95


class ConditionAccumulator:
    """Accumulate episode-aligned values for diagnostic conditions.

    ``add_episode`` is the preferred API because it guarantees that all
    conditions remain paired.  ``add`` is provided for streaming one condition
    at a time; paired statistics then validate equal lengths before use.
    """

    def __init__(self, conditions: Optional[Sequence[str]] = None) -> None:
        self._values: "OrderedDict[str, list[float]]" = OrderedDict()
        self._fixed_conditions = conditions is not None
        if conditions is not None:
            for condition in conditions:
                self._create_condition(condition)

    @staticmethod
    def _validate_condition(condition: str) -> str:
        if not isinstance(condition, str) or not condition:
            raise ValueError("condition names must be non-empty strings")
        return condition

    def _create_condition(self, condition: str) -> None:
        condition = self._validate_condition(condition)
        if condition in self._values:
            raise ValueError(f"duplicate condition: {condition}")
        self._values[condition] = []

    @staticmethod
    def _scalar(value: float) -> float:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 0:
            raise ValueError("each accumulated result must be a scalar")
        result = float(array)
        if not np.isfinite(result):
            raise ValueError("accumulated results must be finite")
        return result

    @property
    def conditions(self) -> Tuple[str, ...]:
        return tuple(self._values.keys())

    def __len__(self) -> int:
        if not self._values:
            return 0
        lengths = {len(values) for values in self._values.values()}
        return lengths.pop() if len(lengths) == 1 else min(lengths)

    def add(self, condition: str, value: float) -> None:
        """Append one scalar to a condition."""

        condition = self._validate_condition(condition)
        if condition not in self._values:
            if self._fixed_conditions:
                raise KeyError(f"unknown condition: {condition}")
            self._values[condition] = []
        self._values[condition].append(self._scalar(value))

    record = add

    def add_episode(self, results: Mapping[str, float]) -> None:
        """Append one complete, paired episode of condition results."""

        if not isinstance(results, Mapping) or not results:
            raise ValueError("results must be a non-empty condition mapping")
        result_keys = tuple(results.keys())
        for condition in result_keys:
            self._validate_condition(condition)

        if not self._values:
            for condition in result_keys:
                self._values[condition] = []
        elif set(result_keys) != set(self._values):
            missing = sorted(set(self._values) - set(result_keys))
            extra = sorted(set(result_keys) - set(self._values))
            raise ValueError(
                f"episode conditions do not match; missing={missing}, extra={extra}"
            )
        lengths = {len(values) for values in self._values.values()}
        if len(lengths) > 1:
            raise ValueError("conditions are already unpaired; cannot add an episode")

        converted = {
            condition: self._scalar(results[condition]) for condition in self._values
        }
        for condition, value in converted.items():
            self._values[condition].append(value)

    update = add_episode

    def values(self, condition: str) -> np.ndarray:
        """Return a copy of one condition's values."""

        if condition not in self._values:
            raise KeyError(f"unknown condition: {condition}")
        return np.asarray(self._values[condition], dtype=np.float64).copy()

    def summary(self, condition: str) -> MeanCI:
        return summarize_mean_ci95(self.values(condition))

    def summaries(self) -> Dict[str, Dict[str, float]]:
        return {
            condition: self.summary(condition).as_dict()
            for condition in self._values
        }

    def paired_difference(self, reference: str, comparison: str) -> MeanCI:
        """Return paired ``reference - comparison`` statistics."""

        return summarize_paired_mean_ci95(
            self.values(reference), self.values(comparison)
        )

    def order_sensitivity(
        self,
        reference: str,
        permutation_conditions: Sequence[str],
    ) -> MeanCI:
        """Compute per-episode ``reference - mean(permuted)`` with paired CI."""

        if not permutation_conditions:
            raise ValueError("permutation_conditions must not be empty")
        reference_values = self.values(reference)
        permuted_values = [self.values(name) for name in permutation_conditions]
        if any(values.shape != reference_values.shape for values in permuted_values):
            raise ValueError("all conditions must contain equally many paired episodes")
        comparison = np.stack(permuted_values, axis=0).mean(axis=0)
        return summarize_paired_mean_ci95(reference_values, comparison)

    def clear(self) -> None:
        for values in self._values.values():
            values.clear()


__all__ = [
    "CI95_Z",
    "ConditionAccumulator",
    "MeanCI",
    "mean_ci95",
    "paired_mean_ci",
    "paired_mean_ci95",
    "summarize_mean_ci95",
    "summarize_paired_mean_ci95",
]

