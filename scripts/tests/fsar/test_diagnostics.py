"""Tests for paired innovation-3 diagnostics."""

import numpy as np
import pytest

from fsar.diagnostics import (
    CI95_Z,
    ConditionAccumulator,
    mean_ci95,
    paired_mean_ci95,
    summarize_mean_ci95,
)


def test_mean_and_ci95_use_sample_standard_deviation():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    result = summarize_mean_ci95(values)
    expected_half_width = CI95_Z * values.std(ddof=1) / np.sqrt(values.size)

    assert result.mean == pytest.approx(2.5)
    assert result.ci95 == pytest.approx(expected_half_width)
    assert result.low == pytest.approx(result.mean - expected_half_width)
    assert result.high == pytest.approx(result.mean + expected_half_width)
    assert result.n == 4
    assert mean_ci95(values) == pytest.approx((2.5, expected_half_width))


def test_paired_ci_is_computed_from_episode_differences():
    reference = [0.9, 0.7, 0.8]
    comparison = [0.8, 0.6, 0.7]
    mean, half_width = paired_mean_ci95(reference, comparison)
    assert mean == pytest.approx(0.1)
    assert half_width == pytest.approx(0.0, abs=1e-15)


def test_condition_accumulator_summaries_pairing_and_order_sensitivity():
    accumulator = ConditionAccumulator(["C0", "P1", "P2"])
    accumulator.add_episode({"C0": 1.0, "P1": 0.0, "P2": 1.0})
    accumulator.update({"C0": 0.0, "P1": 0.0, "P2": 0.0})
    accumulator.add_episode({"C0": 1.0, "P1": 0.0, "P2": 0.0})

    assert len(accumulator) == 3
    assert accumulator.conditions == ("C0", "P1", "P2")
    assert accumulator.summary("C0").mean == pytest.approx(2.0 / 3.0)
    assert accumulator.paired_difference("C0", "P1").mean == pytest.approx(
        2.0 / 3.0
    )

    # Per-episode values: [0.5, 0.0, 1.0].  The CI is therefore genuinely
    # paired instead of being built from independently aggregated accuracies.
    os_result = accumulator.order_sensitivity("C0", ["P1", "P2"])
    assert os_result.mean == pytest.approx(0.5)
    assert os_result.n == 3


def test_condition_accumulator_rejects_unpaired_episodes():
    accumulator = ConditionAccumulator(["C0", "C1"])
    with pytest.raises(ValueError, match="do not match"):
        accumulator.add_episode({"C0": 1.0})

    accumulator.add("C0", 1.0)
    with pytest.raises(ValueError, match="already unpaired"):
        accumulator.add_episode({"C0": 1.0, "C1": 0.0})


def test_statistics_reject_nonfinite_or_unpaired_values():
    with pytest.raises(ValueError, match="finite"):
        summarize_mean_ci95([1.0, np.nan])
    with pytest.raises(ValueError, match="equally many"):
        paired_mean_ci95([1.0, 2.0], [1.0])

