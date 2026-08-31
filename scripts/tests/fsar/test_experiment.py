from __future__ import annotations

import torch
import torch.nn as nn

from fsar.experiment import (
    calibrate_global_alpha,
    compute_episode_loss,
    diagnose_order,
    episode_targets,
    pseudo_validate_episode_alpha,
    save_transport_heatmaps,
)
from fsar.model import EpisodeFeatures, EpisodicTaskAdapter
from fsar.ot import OrderAwareOptimalTransport


def test_compute_episode_loss_includes_order_gradient():
    torch.manual_seed(4)
    aligned = torch.randn(7, 4, 8, requires_grad=True)
    stages = torch.randn(3, 2, 8, requires_grad=True)
    visual = torch.rand(4, 2, requires_grad=True)
    features = EpisodeFeatures(
        visual,
        torch.empty(2, 1, 8, 8),
        torch.empty(4, 8, 8),
        aligned,
        stages,
        torch.tensor([0, 1]),
    )
    semantic = torch.rand(4, 2, requires_grad=True)
    output = {
        "visual_logits": visual,
        "semantic_logits": semantic,
        "final_logits": visual * semantic,
        "features": features,
    }
    targets = torch.tensor([0, 0, 1, 1])
    losses = compute_episode_loss(output, targets, order_weight=0.3, order_margin=0.5)
    losses["loss"].backward()
    assert losses["loss_order"].item() >= 0
    assert aligned.grad is not None and torch.isfinite(aligned.grad).all()
    assert stages.grad is not None and torch.isfinite(stages.grad).all()


class _Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, videos):
        base = videos.float().mean(dim=(1, 3, 4)).unsqueeze(-1)
        return base * torch.arange(1, 9, device=videos.device).float() + self.anchor


class _Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, tokens):
        return tokens.float() * torch.arange(1, 9, device=tokens.device).float() + self.anchor


def _tokenizer(prompts):
    return torch.tensor([[sum(value.encode()) % 71 + 1] for value in prompts])


def test_diagnosis_runs_paired_condition_matrix():
    corpus = {
        "a": {"sub_act_en_li": ["a0", "a1", "a2"]},
        "b": {"sub_act_en_li": ["b0", "b1", "b2"]},
    }
    model = EpisodicTaskAdapter(
        2,
        1,
        1,
        corpus,
        embed_dim=8,
        visual_encoder=_Visual(),
        text_encoder=_Text(),
        tokenizer=_tokenizer,
    )
    images = torch.randn(2, 2, 8, 3, 4, 4)
    labels = torch.tensor([[0, 0], [1, 1]])
    result = diagnose_order([(images, labels), (images + 0.1, labels)], model, torch.device("cpu"))
    assert result["protocol"]["episodes"] == 2
    assert result["summary"]["fused"]["order_sensitivity"]["n"] == 2
    assert len(result["episodes"]) == 2
    visual = result["summary"]["visual"]
    assert visual["perm_123"]["mean"] == visual["perm_321"]["mean"]


def test_episode_targets_are_class_major():
    targets = episode_targets(3, 2, torch.device("cpu"))
    assert targets.tolist() == [0, 0, 1, 1, 2, 2]


class _CountingPseudoModel(nn.Module):
    n_way = 2
    n_support = 1
    n_query = 1
    fusion_mode = "probability_product"
    visual_temperature = 1.0
    semantic_temperature = 1.0

    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, images, labels, return_aux=True):
        self.forward_calls += 1
        query_mean = images[:, 1:].mean(dim=(1, 2, 3, 4, 5))
        visual = torch.stack((query_mean, -query_mean), dim=1)
        semantic = torch.tensor([[0.8, 0.2], [0.2, 0.8]], device=images.device)
        return {
            "visual_logits": visual,
            "semantic_logits": semantic,
            "final_logits": visual * semantic,
        }


def test_pseudo_alpha_uses_support_only_and_zero_grid_forwards():
    model = _CountingPseudoModel()
    images = torch.randn(2, 2, 4, 3, 6, 6)
    labels = torch.tensor([[0, 0], [1, 1]])
    first = pseudo_validate_episode_alpha(
        model,
        images,
        labels,
        alpha_grid=(0.0, 0.5, 1.0),
        global_alpha=0.5,
        views=3,
        seed=11,
    )
    changed_query = images.clone()
    changed_query[:, 1:] = 1000.0
    second = pseudo_validate_episode_alpha(
        model,
        changed_query,
        labels,
        alpha_grid=(0.0, 0.5, 1.0),
        global_alpha=0.5,
        views=3,
        seed=11,
    )
    assert model.forward_calls == 6
    assert first.selection.alpha == second.selection.alpha
    assert first.selection.model_forward_calls_during_grid == 0
    assert first.view_forward_calls == 3
    torch.testing.assert_close(first.visual_scores, second.visual_scores)


def test_transport_heatmap_saves_png_and_raw_plan(tmp_path):
    transport = OrderAwareOptimalTransport(iterations=3)(
        torch.randn(1, 7, 4), torch.randn(2, 3, 4)
    )
    saved = save_transport_heatmaps(
        transport, tmp_path, episode_index=2, max_query_class_pairs=2
    )
    assert len(saved) == 2
    assert all(path.is_file() for path in saved)
    episode_dir = tmp_path / "episode_00002"
    assert (episode_dir / "metadata.json").is_file()
    assert len(list(episode_dir.glob("*.npy"))) == 2


def test_global_alpha_calibration_reuses_one_forward_per_validation_episode():
    model = _CountingPseudoModel()
    batch = (
        torch.randn(2, 2, 4, 3, 4, 4),
        torch.tensor([[0, 0], [1, 1]]),
    )
    selection = calibrate_global_alpha(
        [batch, batch],
        model,
        torch.device("cpu"),
        alpha_grid=(0.0, 0.5, 1.0),
    )
    assert model.forward_calls == 2
    assert selection.model_forward_calls_during_grid == 0
    assert selection.samples == 4
