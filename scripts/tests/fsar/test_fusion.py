from __future__ import annotations

import pytest
import torch

from fsar.fusion import (
    fuse_global_alpha,
    fuse_scores,
    kip_hard_gate,
    select_alpha_from_cached_scores,
)


def _branches():
    visual = torch.tensor([[2.0, -1.0, 0.3], [-0.2, 0.1, 1.7]])
    semantic = torch.tensor([[0.5, 1.2, -0.3], [1.1, -0.4, 0.2]])
    return visual, semantic


def test_legacy_product_is_exact_and_default():
    visual, semantic = _branches()
    expected = visual * semantic
    assert torch.equal(fuse_scores(visual, semantic), expected)
    assert torch.equal(fuse_scores(visual, semantic, mode="legacy_product", alpha=0.0), expected)


def test_probability_product_endpoints_are_single_branch_probabilities():
    visual, semantic = _branches()
    at_visual = fuse_scores(visual, semantic, mode="probability_product", alpha=1.0)
    at_semantic = fuse_scores(visual, semantic, mode="probability_product", alpha=0.0)
    # Endpoint exponent is two, so the selected probability is squared and
    # renormalised rather than merely copied.
    expected_visual = torch.softmax(2.0 * torch.log_softmax(visual, -1), -1)
    expected_semantic = torch.softmax(2.0 * torch.log_softmax(semantic, -1), -1)
    torch.testing.assert_close(at_visual, expected_visual)
    torch.testing.assert_close(at_semantic, expected_semantic)


def test_alpha_half_is_probability_product_and_normalisation_preserves_prediction():
    visual, semantic = _branches()
    pv = torch.softmax(visual, -1)
    ps = torch.softmax(semantic, -1)
    raw = fuse_scores(
        visual,
        semantic,
        mode="probability_product",
        alpha=0.5,
        normalize_probability=False,
    )
    normalised = fuse_scores(visual, semantic, mode="probability_product", alpha=0.5)
    torch.testing.assert_close(raw, pv * ps)
    torch.testing.assert_close(normalised, raw / raw.sum(-1, keepdim=True))
    assert torch.equal(raw.argmax(-1), normalised.argmax(-1))
    # Probability product is intentionally not the raw-cosine legacy product.
    assert not torch.allclose(normalised, visual * semantic)


def test_logit_convex_endpoints_and_half():
    visual, semantic = _branches()
    torch.testing.assert_close(
        fuse_scores(visual, semantic, mode="logit_convex", alpha=1.0), 2 * visual
    )
    torch.testing.assert_close(
        fuse_scores(visual, semantic, mode="logit_convex", alpha=0.0), 2 * semantic
    )
    torch.testing.assert_close(
        fuse_scores(visual, semantic, mode="logit_convex", alpha=0.5), visual + semantic
    )


def test_global_alpha_wrapper_and_validation():
    visual, semantic = _branches()
    expected = fuse_scores(visual, semantic, mode="probability_product", alpha=0.7)
    torch.testing.assert_close(fuse_global_alpha(visual, semantic, 0.7), expected)
    with pytest.raises(ValueError, match="undefined"):
        fuse_global_alpha(visual, semantic, 0.5, mode="legacy_product")


def test_pseudo_validation_uses_accuracy_then_margin_and_reports_zero_forwards():
    # Alpha=1 and alpha=0 are both perfectly accurate; the visual branch has
    # the wider correct-class margins, so the secondary rule selects alpha=1.
    visual = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    semantic = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    targets = torch.tensor([[0, 1]])
    selected = select_alpha_from_cached_scores(
        visual,
        semantic,
        targets,
        alpha_grid=[0.0, 1.0],
        global_alpha=0.5,
        mode="logit_convex",
    )
    assert selected.alpha == 1.0
    assert selected.reason == "max_accuracy_then_margin"
    assert not selected.used_fallback
    assert selected.model_forward_calls_during_grid == 0
    assert selected.used_cached_scores
    assert selected.samples == 2
    assert selected.as_dict()["model_forward_calls_during_grid"] == 0


def test_residual_tie_is_stable_by_global_proximity_then_smaller_alpha():
    # The two samples exchange branch strengths.  Alpha=.25 and .75 therefore
    # have identical perfect accuracy and identical mean margin, while alpha=0
    # is worse and keeps the overall curve non-flat.
    visual = torch.tensor([[4.0, 0.0], [-1.0, 0.0]])
    semantic = torch.tensor([[-1.0, 0.0], [4.0, 0.0]])
    targets = torch.tensor([0, 0])
    selected = select_alpha_from_cached_scores(
        visual,
        semantic,
        targets,
        alpha_grid=[0.0, 0.25, 0.75],
        global_alpha=0.5,
        mode="logit_convex",
    )
    # Both candidates have the same accuracy and correct-class margin.
    assert selected.alpha == 0.25


def test_flat_curve_falls_back_to_global_alpha_even_if_not_on_grid():
    visual = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    semantic = visual.clone()
    targets = torch.tensor([[0, 1]])
    selected = select_alpha_from_cached_scores(
        visual,
        semantic,
        targets,
        alpha_grid=[0.0, 0.5, 1.0],
        global_alpha=0.35,
    )
    assert selected.alpha == 0.35
    assert selected.used_fallback
    assert selected.reason == "flat_curve_global_fallback"
    assert len(selected.candidates) == 3


def test_accuracy_tie_with_nonflat_margin_is_not_a_flat_fallback():
    visual = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    semantic = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    targets = torch.tensor([0, 1])
    selected = select_alpha_from_cached_scores(
        visual,
        semantic,
        targets,
        alpha_grid=[0.0, 1.0],
        global_alpha=0.0,
        mode="logit_convex",
    )
    assert selected.alpha == 1.0
    assert not selected.used_fallback


def test_kip_gate_selects_more_stable_branch_and_visual_on_exact_tie():
    visual = torch.tensor([[5.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    semantic = torch.tensor([[1.0, 0.0], [0.0, 6.0], [3.0, 0.0]])
    result = kip_hard_gate(visual, semantic)
    assert result.choose_visual.tolist() == [True, False, True]
    torch.testing.assert_close(result.scores.sum(-1), torch.ones(3))
    assert result.visual_selection_rate == pytest.approx(2 / 3)


def test_probability_fusion_is_finite_for_extreme_scores_and_has_gradients():
    visual = torch.tensor([[1000.0, -1000.0]], requires_grad=True)
    semantic = torch.tensor([[-1000.0, 1000.0]], requires_grad=True)
    scores = fuse_scores(visual, semantic, mode="probability_product", alpha=0.37)
    assert torch.isfinite(scores).all()
    scores.square().sum().backward()
    assert visual.grad is not None and torch.isfinite(visual.grad).all()
    assert semantic.grad is not None and torch.isfinite(semantic.grad).all()


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_invalid_alpha_is_rejected(alpha):
    visual, semantic = _branches()
    with pytest.raises(ValueError, match="alpha"):
        fuse_scores(visual, semantic, mode="probability_product", alpha=alpha)
