import math

import pytest
import torch

from fsar.ot import OrderAwareOptimalTransport, position_distance, sinkhorn_log


def _features(seed: int = 7, *, q: int = 2, c: int = 3, t: int = 7, k: int = 3, d: int = 8):
    generator = torch.Generator().manual_seed(seed)
    frames = torch.randn(q, t, d, generator=generator)
    stages = torch.randn(c, k, d, generator=generator)
    return frames, stages


def test_lambda_zero_is_invariant_to_stage_permutation():
    frames, stages = _features()
    model = OrderAwareOptimalTransport(epsilon=0.08, lambda_pos=0.0, rho=0.3, iterations=50)
    original = model(frames, stages)
    permutation = torch.tensor([2, 0, 1])
    permuted = model(frames, stages[:, permutation])

    torch.testing.assert_close(original.score, permuted.score, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        original.plan[..., permutation], permuted.plan, atol=2e-6, rtol=2e-6
    )


def test_large_position_weight_concentrates_plan_on_diagonal():
    # Constant embeddings make position the only discriminating cost.
    frames = torch.ones(1, 4, 5)
    stages = torch.ones(1, 4, 5)
    model = OrderAwareOptimalTransport(
        epsilon=0.01, lambda_pos=20.0, rho=math.inf, iterations=80
    )
    result = model(frames, stages)
    plan = result.plan[0, 0]

    assert plan.diagonal().sum() > 0.999
    assert result.row_residual.max() < 1e-6
    assert result.column_residual.max() < 1e-6


def test_finite_rho_has_finite_gradients_and_hard_frame_marginal():
    frames, stages = _features()
    frames.requires_grad_()
    stages.requires_grad_()
    result = OrderAwareOptimalTransport(
        epsilon=0.05, lambda_pos=0.7, rho=0.2, iterations=30
    )(frames, stages)

    loss = result.score.square().mean() + 0.05 * result.transport_cost.mean()
    loss.backward()
    assert torch.isfinite(result.plan).all()
    assert torch.isfinite(result.fhat).all()
    assert torch.isfinite(frames.grad).all()
    assert torch.isfinite(stages.grad).all()
    assert result.row_residual.max() < 2e-6


def test_fp16_inputs_leave_ot_island_as_float32():
    frames, stages = _features(q=1, c=2)
    result = OrderAwareOptimalTransport(rho=0.5)(frames.half(), stages.half())

    for value in (
        result.plan,
        result.mass,
        result.fhat,
        result.score,
        result.cost,
        result.transport_cost,
    ):
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()
    assert result.plan.shape == (1, 2, 7, 3)
    assert result.mass.shape == (1, 2, 3)
    assert result.fhat.shape == (1, 2, 3, 8)
    assert result.score.shape == (1, 2)


def test_infinite_rho_uses_explicit_balanced_branch():
    frames, stages = _features(q=1, c=1, t=5, k=4)
    result = OrderAwareOptimalTransport(
        epsilon=0.1, lambda_pos=0.0, rho=math.inf, iterations=100
    )(frames, stages)

    assert result.balanced is True
    torch.testing.assert_close(
        result.plan.sum(dim=-1), torch.full((1, 1, 5), 0.2), atol=2e-6, rtol=0
    )
    torch.testing.assert_close(
        result.mass, torch.full((1, 1, 4), 0.25), atol=2e-6, rtol=0
    )


def test_position_distance_matches_squared_normalized_definition():
    expected = torch.tensor([[0.0, 1.0], [0.25, 0.25], [1.0, 0.0]])
    torch.testing.assert_close(position_distance(3, 2), expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epsilon": 0.0},
        {"rho": 0.0},
        {"iterations": 0},
    ],
)
def test_sinkhorn_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        sinkhorn_log(torch.zeros(2, 3), **kwargs)
