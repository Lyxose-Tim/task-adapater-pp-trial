"""Tests for the innovation-3 order-aware core."""

import math

import pytest
import torch

from fsar.order import (
    IDENTITY_PERMUTATION,
    NON_IDENTITY_STAGE_PERMUTATIONS,
    OrderContrastiveLoss,
    STAGE_PERMUTATIONS,
    all_permutation_scores,
    correct_class_order_scores,
    fixed_stage_semantic_scores,
    make_frame_permutation,
    permute_query_frames,
    permute_stage_text,
    stage_permutations,
)


def test_three_stages_have_six_deterministic_permutations():
    assert len(STAGE_PERMUTATIONS) == math.factorial(3)
    assert len(set(STAGE_PERMUTATIONS)) == 6
    assert STAGE_PERMUTATIONS[0] == IDENTITY_PERMUTATION
    assert len(NON_IDENTITY_STAGE_PERMUTATIONS) == 5
    assert IDENTITY_PERMUTATION not in NON_IDENTITY_STAGE_PERMUTATIONS
    assert stage_permutations(3, include_identity=False) == (
        NON_IDENTITY_STAGE_PERMUTATIONS
    )


def test_fixed_stage_scoring_keeps_visual_windows_fixed():
    aligned = torch.zeros(7, 2, 3)
    aligned[0:3, :, 0] = 1.0
    stage_text = torch.zeros(3, 1, 3)
    stage_text[0, 0, 0] = 1.0

    identity = fixed_stage_semantic_scores(aligned, stage_text)
    swapped_text = permute_stage_text(stage_text, (1, 0, 2))
    swapped = fixed_stage_semantic_scores(aligned, swapped_text)

    assert identity.shape == (2, 1)
    assert torch.allclose(identity, torch.full((2, 1), 1.0 / 3.0))
    # Original stage zero is now compared with fixed window [2:5].  Only
    # aligned position 2 overlaps the visual signal.
    assert torch.allclose(swapped, torch.full((2, 1), 1.0 / 9.0))


def test_all_permutation_scores_and_correct_class_selection():
    torch.manual_seed(4)
    aligned = torch.randn(7, 4, 5, requires_grad=True)
    stage_text = torch.randn(3, 3, 5, requires_grad=True)
    scores = all_permutation_scores(aligned, stage_text)
    selected = correct_class_order_scores(scores, torch.tensor([0, 2, 1, 0]))

    assert scores.shape == (6, 4, 3)
    assert selected.shape == (4, 6)
    assert torch.allclose(selected[:, 0], scores[0, range(4), [0, 2, 1, 0]])
    selected.sum().backward()
    assert aligned.grad is not None
    assert stage_text.grad is not None


def test_fixed_stage_scoring_rejects_wrong_temporal_length():
    with pytest.raises(ValueError, match="requires T=7"):
        fixed_stage_semantic_scores(torch.randn(6, 2, 4), torch.randn(3, 3, 4))


def test_margin_order_contrastive_loss_numeric_and_gradient():
    positive = torch.tensor([0.8, 0.5], requires_grad=True)
    negatives = torch.tensor([[0.7, 0.4], [0.6, 0.55]], requires_grad=True)
    loss_fn = OrderContrastiveLoss(
        mode="margin", margin=0.2, reduction="none"
    )
    loss = loss_fn(positive, negatives)

    assert torch.allclose(loss, torch.tensor([0.05, 0.275]))
    loss.mean().backward()
    assert positive.grad is not None
    assert negatives.grad is not None


def test_infonce_order_contrastive_loss_and_direct_score_api():
    positive = torch.tensor([2.0, 1.0])
    negatives = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    loss_fn = OrderContrastiveLoss(
        mode="infonce", temperature=1.0, reduction="none"
    )
    expected = -torch.log_softmax(
        torch.cat((positive[:, None], negatives), dim=-1), dim=-1
    )[:, 0]
    assert torch.allclose(loss_fn(positive, negatives), expected)

    permutation_scores = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ]
    )
    direct = loss_fn.from_permutation_scores(
        permutation_scores, torch.tensor([0, 1])
    )
    assert direct.shape == (2,)
    assert direct[0] < direct[1]


def test_query_frame_permutation_preserves_support_and_is_synchronous():
    # [way, support+query, time, channel]
    episode = torch.arange(2 * 3 * 4 * 1).reshape(2, 3, 4, 1)
    order = make_frame_permutation(4, mode="reverse")
    result = permute_query_frames(episode, order, n_support=1)

    assert torch.equal(result[:, :1], episode[:, :1])
    assert torch.equal(result[:, 1:], episode[:, 1:].flip(2))
    assert order == (3, 2, 1, 0)
    assert make_frame_permutation(8, mode="random", seed=916) == (
        make_frame_permutation(8, mode="random", seed=916)
    )


def test_query_frame_permutation_supports_repository_layout():
    episode = torch.arange(1 * 2 * 1 * 3 * 1 * 1).reshape(1, 2, 1, 3, 1, 1)
    result = permute_query_frames(
        episode, (2, 0, 1), n_support=1, sample_dim=1, time_dim=3
    )
    assert torch.equal(result[:, :1], episode[:, :1])
    assert torch.equal(result[:, 1:, :, :, :, :], episode[:, 1:, :, [2, 0, 1], :, :])

