from __future__ import annotations

import torch
import torch.nn as nn

from fsar.corpus_v2 import ClassCandidates, CorpusV2, StageCandidate
from fsar.model import EpisodicTaskAdapter, fixed_window_semantic_scores
from module_sem_adapter import ResidualAttentionBlock


class FakeVisualEncoder(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.dim = dim

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        # videos [B,C,T,H,W]
        base = videos.float().mean(dim=(1, 3, 4)).unsqueeze(-1)
        scales = torch.arange(1, self.dim + 1, device=videos.device).float()
        return base * scales + self.anchor


class FakeTokenizer:
    def __call__(self, prompts):
        return torch.tensor([[sum(text.encode("utf-8")) % 97] for text in prompts], dtype=torch.long)


class FakeTextEncoder(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.dim = dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scales = torch.arange(1, self.dim + 1, device=tokens.device).float()
        return tokens.float() * scales + self.anchor


def _corpus():
    return {
        "a": {"sub_act_en_li": ["a0", "a1", "a2"]},
        "b": {"sub_act_en_li": ["b0", "b1", "b2"]},
        "c": {"sub_act_en_li": ["c0", "c1", "c2"]},
    }


def test_branch_orientation_is_query_by_class():
    model = EpisodicTaskAdapter(
        3,
        1,
        2,
        _corpus(),
        num_frames=8,
        embed_dim=8,
        visual_encoder=FakeVisualEncoder(),
        text_encoder=FakeTextEncoder(),
        tokenizer=FakeTokenizer(),
    )
    images = torch.randn(3, 3, 8, 3, 4, 4)
    labels = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    visual, semantic = model(images, labels)
    assert visual.shape == (6, 3)
    assert semantic.shape == (6, 3)


def test_fixed_window_scores_have_query_rows_and_class_columns():
    stage_text = torch.eye(3).unsqueeze(0).repeat(3, 1, 1)  # [K,C,D]
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    aligned = torch.eye(3)[targets].unsqueeze(0).repeat(7, 1, 1)
    scores = fixed_window_semantic_scores(aligned, stage_text)
    assert scores.shape == (6, 3)
    assert torch.equal(scores.argmax(dim=1), targets)


class CrossBatchMixingTextEncoder(nn.Module):
    """Couples every prompt in one forward call across the batch axis.

    The real semantic encoder's O-MSA branch attends across the prompt/batch
    dimension, so any implementation that batches prompts from different
    classes into one call leaks information between classes.  This encoder
    makes such leakage detectable with synthetic tensors.
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.dim = dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scales = torch.arange(1, self.dim + 1, device=tokens.device).float()
        base = tokens.float() * scales
        return base + base.mean(dim=0, keepdim=True) + self.anchor


def test_text_stage_encoding_is_isolated_per_class():
    # 落实 R-02：episode 文本编码不得让不同类的子动作互相影响。
    def build(n_way):
        return EpisodicTaskAdapter(
            n_way,
            1,
            1,
            _corpus(),
            num_frames=8,
            embed_dim=8,
            visual_encoder=FakeVisualEncoder(),
            text_encoder=CrossBatchMixingTextEncoder(),
            tokenizer=FakeTokenizer(),
        )

    full_episode = build(3).encode_text_stages(torch.tensor([0, 1, 2]))
    for class_position, class_index in enumerate((0, 1, 2)):
        solo = build(1).encode_text_stages(torch.tensor([class_index]))
        torch.testing.assert_close(full_episode[:, class_position], solo[:, 0])


def test_released_order_msa_is_permutation_equivariant():
    torch.manual_seed(7)
    mask = torch.empty(5, 5).fill_(float("-inf")).triu_(1)
    block = ResidualAttentionBlock(16, 4, True, mask).eval()
    x = torch.randn(5, 3, 16)
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        expected = block(x)[:, permutation]
        actual = block(x[:, permutation])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_model_integrates_order_aware_ot_with_query_class_orientation():
    model = EpisodicTaskAdapter(
        3,
        1,
        2,
        _corpus(),
        num_frames=8,
        embed_dim=8,
        visual_encoder=FakeVisualEncoder(),
        text_encoder=FakeTextEncoder(),
        tokenizer=FakeTokenizer(),
        semantic_backend="ot",
        ot_options={"epsilon": 0.1, "lambda_pos": 0.7, "rho": 0.5, "iterations": 10},
    )
    images = torch.randn(3, 3, 8, 3, 4, 4)
    labels = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    output = model(images, labels, return_aux=True)
    assert output["semantic_logits"].shape == (6, 3)
    assert output["transport"].plan.shape == (6, 3, 7, 3)
    assert output["transport"].balanced is False
    torch.testing.assert_close(
        output["final_logits"], output["visual_logits"] * output["semantic_logits"]
    )


def test_model_probability_fusion_is_selected_centrally():
    model = EpisodicTaskAdapter(
        3,
        1,
        1,
        _corpus(),
        num_frames=8,
        embed_dim=8,
        visual_encoder=FakeVisualEncoder(),
        text_encoder=FakeTextEncoder(),
        tokenizer=FakeTokenizer(),
        fusion_mode="probability_product",
        fusion_alpha=0.3,
    )
    images = torch.randn(3, 2, 8, 3, 4, 4)
    labels = torch.tensor([[0, 0], [1, 1], [2, 2]])
    output = model(images, labels, return_aux=True)
    assert output["fusion_mode"] == "probability_product"
    torch.testing.assert_close(output["final_logits"].sum(dim=-1), torch.ones(3))


def test_model_selects_ragged_v2_candidates_from_support_only():
    def entry(prefix):
        return ClassCandidates(
            [
                StageCandidate(
                    k=k,
                    subs=tuple(f"{prefix}{k}-{index}" for index in range(k)),
                    gen_id=f"{prefix}-k{k}",
                )
                for k in (2, 3, 4, 5)
            ]
        )

    corpus = CorpusV2(dataset="toy", classes={"a": entry("a"), "b": entry("b")})
    model = EpisodicTaskAdapter(
        2,
        1,
        1,
        corpus,
        num_frames=8,
        embed_dim=8,
        visual_encoder=FakeVisualEncoder(),
        text_encoder=FakeTextEncoder(),
        tokenizer=FakeTokenizer(),
        semantic_backend="ot",
        ot_options={"epsilon": 0.1, "lambda_pos": 0.0, "rho": 0.5, "iterations": 5},
    )
    images = torch.randn(2, 2, 8, 3, 4, 4)
    labels = torch.tensor([[0, 0], [1, 1]])
    output = model(images, labels, return_aux=True)
    features = output["features"]
    assert isinstance(features.stage_text, tuple)
    assert len(features.stage_text) == 2
    assert {row["K"] for row in features.selected_candidates}.issubset({2, 3, 4, 5})
    assert output["semantic_logits"].shape == (2, 2)
    assert isinstance(output["transport"], tuple)
