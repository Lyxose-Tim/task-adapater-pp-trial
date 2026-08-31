"""Canonical episodic Task-Adapter++ model used by the innovation experiments.

This module isolates the paper implementation from the unrelated ``models/``
ZSL/continual-learning package.  All score matrices follow one convention:
rows are query videos and columns are episode classes.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fsar.fusion import fuse_scores
from fsar.corpus_v2 import CorpusV2, StageCandidate, select_candidate_from_support
from fsar.order import fixed_stage_semantic_scores
from fsar.ot import OTAlignment, OrderAwareOptimalTransport


Tensor = torch.Tensor


def _autocast_disabled(device: torch.device) -> ContextManager[object]:
    """Keep the small cosine-scoring island in fp32 under CUDA AMP."""

    if device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


@dataclass
class EpisodeFeatures:
    """Reusable features for paired order diagnostics."""

    visual_scores: Tensor          # [Q, C]
    support_frames: Tensor         # [C, shot, T, D]
    query_frames: Tensor           # [Q, T, D]
    aligned_query: Tensor          # [T-1, Q, D]
    stage_text: Union[Tensor, Tuple[Tensor, ...]]  # [K,C,D] or ragged C * [Kc,D]
    class_indices: Tensor          # [C]
    selected_candidates: Optional[Tuple[Mapping[str, Any], ...]] = None


def fixed_window_semantic_scores(aligned_query: Tensor, stage_text: Tensor) -> Tensor:
    """Paper Eq. (16) with overlapping windows, returned as ``[Q, C]``.

    With seven aligned frames and three stages, the windows are exactly
    ``[0,1,2]``, ``[2,3,4]`` and ``[4,5,6]``.  The generalized form keeps
    evenly-spaced, overlapping windows for other valid ``T``/``K`` values.
    """

    return fixed_stage_semantic_scores(aligned_query, stage_text)


class QueryCrossAttention(nn.Module):
    """The parameter-free cross-query attention used by the released code."""

    def __init__(self, dim: int = 512, residual: str = "current") -> None:
        super().__init__()
        if residual not in {"current", "previous"}:
            raise ValueError("residual must be 'current' (released code) or 'previous' (paper Eq.13)")
        self.scale = dim ** -0.5
        self.residual = residual

    def forward(self, previous: Tensor, current: Tensor) -> Tensor:
        if previous.shape != current.shape or previous.ndim != 2:
            raise ValueError("adjacent frame tensors must share [Q,D] shape")
        weights = torch.softmax(previous @ current.transpose(0, 1) * self.scale, dim=-1)
        attended = weights @ current
        residual = current if self.residual == "current" else previous
        return attended + residual


class EpisodicTaskAdapter(nn.Module):
    """Paper-faithful episodic support/query model with innovation hooks."""

    def __init__(
        self,
        n_way: int,
        n_support: int,
        n_query: int,
        corpus: Union[Mapping[str, Mapping[str, Any]], CorpusV2],
        *,
        visual_depth: int = 6,
        text_depth: int = 2,
        num_frames: int = 8,
        embed_dim: int = 512,
        checkpoint_path: Optional[str] = None,
        visual_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        tokenizer: Optional[Callable[[Sequence[str]], Tensor]] = None,
        cross_attention_residual: str = "current",
        semantic_backend: str = "fixed_window",
        ot_options: Optional[Mapping[str, Any]] = None,
        ot_frame_source: str = "aligned",
        fusion_mode: str = "legacy_product",
        fusion_alpha: float = 0.5,
        visual_temperature: float = 1.0,
        semantic_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if min(n_way, n_support, n_query, num_frames) < 1:
            raise ValueError("episode sizes and num_frames must be positive")
        self.n_way = int(n_way)
        self.n_support = int(n_support)
        self.n_query = int(n_query)
        self.num_frames = int(num_frames)
        self.embed_dim = int(embed_dim)
        self.corpus_v2 = corpus if isinstance(corpus, CorpusV2) else None
        self.corpus = {} if self.corpus_v2 is not None else dict(corpus)
        self.class_names = (
            list(self.corpus_v2.classes.keys())
            if self.corpus_v2 is not None
            else list(self.corpus.keys())
        )
        self.tokenizer = tokenizer
        if semantic_backend not in {"fixed_window", "ot"}:
            raise ValueError("semantic_backend must be 'fixed_window' or 'ot'")
        if ot_frame_source not in {"aligned", "raw"}:
            raise ValueError("ot_frame_source must be 'aligned' or 'raw'")
        self.semantic_backend = semantic_backend
        self.ot_frame_source = ot_frame_source
        self.ot = (
            OrderAwareOptimalTransport(**dict(ot_options or {}))
            if semantic_backend == "ot"
            else None
        )
        # Innovation 2 uses the same OT machinery for support-only candidate
        # selection even when the requested scoring ablation is fixed-window.
        self.candidate_selector_ot = (
            self.ot
            if self.ot is not None
            else OrderAwareOptimalTransport(**dict(ot_options or {}))
        )
        self.fusion_mode = str(fusion_mode)
        self.fusion_alpha = float(fusion_alpha)
        self.visual_temperature = float(visual_temperature)
        self.semantic_temperature = float(semantic_temperature)

        if visual_encoder is None or text_encoder is None:
            from fsar.utils import get_clip_tokenizer, resolve_checkpoint

            resolved = resolve_checkpoint(checkpoint_path)
            if visual_encoder is None:
                from module_adapter import clip_vit_base_patch16_adapter

                visual_encoder = clip_vit_base_patch16_adapter(
                    embed_dim=embed_dim,
                    adapter_layers=visual_depth,
                    checkpoint_path=resolved,
                )
            if text_encoder is None:
                from module_sem_adapter import clip_encode_text_adapter

                text_encoder = clip_encode_text_adapter(text_depth, checkpoint_path=resolved)
            if self.tokenizer is None:
                self.tokenizer = get_clip_tokenizer()

        if self.tokenizer is None:
            raise ValueError("tokenizer is required when encoders are injected")
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.cross_attention = QueryCrossAttention(embed_dim, residual=cross_attention_residual)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _episode_tensor(self, images: Tensor) -> Tensor:
        """Normalize accepted inputs to ``[C,SQ,T,3,H,W]``."""

        sq = self.n_support + self.n_query
        if images.ndim == 6:
            expected = (self.n_way, sq, self.num_frames)
            if tuple(images.shape[:3]) != expected:
                raise ValueError(f"episode prefix must be {expected}, got {tuple(images.shape[:3])}")
            return images
        if images.ndim == 4:
            expected = self.n_way * sq * self.num_frames
            if images.shape[0] != expected:
                raise ValueError(f"expected {expected} flattened frames, got {images.shape[0]}")
            return images.reshape(self.n_way, sq, self.num_frames, *images.shape[1:])
        raise ValueError("images must have [C,SQ,T,3,H,W] or flattened [N,3,H,W] shape")

    def _class_indices(self, labels: Tensor) -> Tensor:
        if labels.ndim == 1:
            indices = labels
        elif labels.ndim >= 2:
            indices = labels[:, 0]
        else:
            raise ValueError("labels must identify one class per episode row")
        indices = indices.to(dtype=torch.long, device="cpu")
        if indices.numel() != self.n_way:
            raise ValueError(f"expected {self.n_way} class ids, got {indices.numel()}")
        if indices.min().item() < 0 or indices.max().item() >= len(self.class_names):
            raise IndexError("episode class id is outside the corpus order")
        return indices

    def _encode_visual(self, videos: Tensor) -> Tensor:
        # Backbone contract: [B,C,T,H,W] -> [B,T,D].
        features = self.visual_encoder(videos)
        if features.ndim != 3 or features.shape[1] != self.num_frames:
            raise ValueError("visual encoder must return [B,T,D]")
        return features

    def encode_visual_episode(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        episode = self._episode_tensor(images)
        support = episode[:, :self.n_support]
        query = episode[:, self.n_support:]
        channels, height, width = episode.shape[3:]
        support_in = support.reshape(self.n_way * self.n_support, self.num_frames, channels, height, width)
        query_in = query.reshape(self.n_way * self.n_query, self.num_frames, channels, height, width)
        support_in = support_in.permute(0, 2, 1, 3, 4)
        query_in = query_in.permute(0, 2, 1, 3, 4)

        # Separate calls are an information-isolation invariant of the paper.
        support_frames = self._encode_visual(support_in).reshape(
            self.n_way, self.n_support, self.num_frames, -1
        )
        query_frames = self._encode_visual(query_in).reshape(
            self.n_way * self.n_query, self.num_frames, -1
        )
        prototypes = support_frames.mean(dim=1)
        with _autocast_disabled(query_frames.device):
            query_video = F.normalize(query_frames.mean(dim=1).float(), dim=-1)
            support_video = F.normalize(prototypes.mean(dim=1).float(), dim=-1)
            visual_scores = query_video @ support_video.transpose(0, 1)

        aligned = self._align_sequences(query_frames)
        return visual_scores, support_frames, query_frames, aligned

    def _align_sequences(self, frames: Tensor) -> Tensor:
        """Apply the released adjacent-frame cross-query alignment.

        Args:
            frames: ``[B,T,D]`` sequences.  The returned orientation is kept
                compatible with the released implementation: ``[T-1,B,D]``.
        """

        if frames.ndim != 3 or frames.shape[1] != self.num_frames:
            raise ValueError("frames must have [B,num_frames,D] shape")
        query_by_time = frames.permute(1, 0, 2)
        aligned = [
            self.cross_attention(query_by_time[i], query_by_time[i + 1])
            for i in range(self.num_frames - 1)
        ]
        return torch.stack(aligned, dim=0)

    def _sub_actions(self, class_name: str) -> Sequence[str]:
        entry = self.corpus[class_name]
        for key in ("sub_act_en_li", "sub_actions", "subs"):
            if key in entry:
                values = entry[key]
                break
        else:
            raise KeyError(f"corpus entry {class_name!r} has no sub-action list")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            raise ValueError(f"corpus entry {class_name!r} has invalid sub-actions")
        if any(not str(value).strip() for value in values):
            raise ValueError(f"corpus entry {class_name!r} contains an empty sub-action")
        return [str(value) for value in values]

    def encode_text_stages(
        self,
        class_indices: Tensor,
        permutation: Optional[Sequence[int]] = None,
    ) -> Tensor:
        # One text-transformer call per class, exactly like the released
        # implementation.  This is a semantic requirement rather than a
        # performance choice: the O-MSA branch of the semantic encoder attends
        # across the prompt/batch axis, so batching the whole episode would
        # let sub-actions of different classes attend to each other.  (R-02)
        per_class = []
        expected_k: Optional[int] = None
        for raw_index in class_indices.tolist():
            name = self.class_names[int(raw_index)]
            subs = list(self._sub_actions(name))
            if permutation is not None:
                if sorted(permutation) != list(range(len(subs))):
                    raise ValueError("permutation must cover every stage exactly once")
                subs = [subs[i] for i in permutation]
            if expected_k is None:
                expected_k = len(subs)
            if len(subs) != expected_k:
                raise ValueError("fixed-window baseline requires equal K within an episode")
            per_class.append(self._encode_stage_list(name, subs))
        if expected_k is None:
            raise ValueError("episode contains no text classes")
        # C * [K,D] -> [C,K,D] -> [K,C,D]
        return torch.stack(per_class, dim=0).permute(1, 0, 2)

    def _encode_stage_list(self, class_name: str, stages: Sequence[str]) -> Tensor:
        prompts = [f"A video of action about {class_name}: {stage}" for stage in stages]
        tokens = self.tokenizer(prompts)
        if not torch.is_tensor(tokens):
            tokens = torch.as_tensor(tokens)
        embedding = self.text_encoder(tokens.to(self.device))
        if embedding.ndim != 2 or embedding.shape[0] != len(stages):
            raise ValueError("text encoder must return one [D] row per stage")
        return embedding

    def select_v2_text_stages(
        self,
        class_indices: Tensor,
        support_frames: Tensor,
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Mapping[str, Any], ...]]:
        """Select one variable-K candidate per class using support videos only.

        The public signature deliberately has no query features.  Each support
        video is aligned independently, candidate fitting uses OT transport
        cost, and exact ties follow the v2 contract by preferring smaller K.
        """

        if self.corpus_v2 is None:
            raise RuntimeError("v2 stage selection requires a CorpusV2")
        if support_frames.shape[:2] != (self.n_way, self.n_support):
            raise ValueError("support_frames must have [C,shot,T,D] shape")
        selected_embeddings = []
        diagnostics = []
        for episode_class, raw_index in enumerate(class_indices.tolist()):
            class_name = self.class_names[int(raw_index)]
            candidates = self.corpus_v2.classes[class_name].active_candidates
            if not candidates:
                raise ValueError(f"v2 class {class_name!r} has no active candidates")
            encoded = {
                candidate.gen_id: self._encode_stage_list(class_name, candidate.subs)
                for candidate in candidates
            }
            aligned_support = self._align_sequences(support_frames[episode_class]).permute(1, 0, 2)

            def support_cost(item: Tensor, candidate: StageCandidate) -> Tensor:
                result = self.candidate_selector_ot(
                    item.unsqueeze(0), encoded[candidate.gen_id].unsqueeze(0)
                )
                return result.transport_cost[0, 0]

            selection = select_candidate_from_support(
                class_name,
                candidates,
                tuple(aligned_support),
                support_cost,
            )
            selected_embeddings.append(encoded[selection.candidate.gen_id])
            diagnostics.append(
                {
                    "class_name": class_name,
                    "gen_id": selection.candidate.gen_id,
                    "K": selection.candidate.k,
                    "mean_support_cost": selection.mean_support_cost,
                    "support_count": selection.support_count,
                }
            )
        return tuple(selected_embeddings), tuple(diagnostics)

    def extract_episode_features(self, images: Tensor, labels: Tensor) -> EpisodeFeatures:
        class_indices = self._class_indices(labels)
        visual, support, query, aligned = self.encode_visual_episode(images)
        if self.corpus_v2 is None:
            stage_text = self.encode_text_stages(class_indices)
            selections = None
        else:
            stage_text, selections = self.select_v2_text_stages(class_indices, support)
        return EpisodeFeatures(
            visual, support, query, aligned, stage_text, class_indices, selections
        )

    def score_episode(
        self,
        features: EpisodeFeatures,
        stage_permutation: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        stage_text = features.stage_text
        if stage_permutation is not None:
            if torch.is_tensor(stage_text):
                if sorted(stage_permutation) != list(range(stage_text.shape[0])):
                    raise ValueError("stage_permutation is invalid")
                stage_text = stage_text[list(stage_permutation)]
            else:
                permuted = []
                for stages in stage_text:
                    if sorted(stage_permutation) != list(range(stages.shape[0])):
                        raise ValueError(
                            "one shared stage_permutation cannot cover ragged K"
                        )
                    permuted.append(stages[list(stage_permutation)])
                stage_text = tuple(permuted)
        transport: Optional[Union[OTAlignment, Tuple[OTAlignment, ...]]] = None
        if self.semantic_backend == "fixed_window":
            if torch.is_tensor(stage_text):
                semantic = fixed_window_semantic_scores(features.aligned_query, stage_text)
            else:
                semantic = torch.cat(
                    [
                        fixed_window_semantic_scores(
                            features.aligned_query, stages.unsqueeze(1)
                        )
                        for stages in stage_text
                    ],
                    dim=1,
                )
        else:
            if self.ot is None:  # defensive guard for mutated/deserialised modules
                raise RuntimeError("OT semantic backend has no solver")
            frames = (
                features.aligned_query.permute(1, 0, 2)
                if self.ot_frame_source == "aligned"
                else features.query_frames
            )
            if torch.is_tensor(stage_text):
                transport = self.ot(frames, stage_text.permute(1, 0, 2))
                semantic = transport.score
            else:
                per_class = tuple(
                    self.ot(frames, stages.unsqueeze(0)) for stages in stage_text
                )
                transport = per_class
                semantic = torch.cat([item.score for item in per_class], dim=1)
        if semantic.shape != features.visual_scores.shape:
            raise AssertionError(
                f"branch shapes must both be [Q,C], got {features.visual_scores.shape} and {semantic.shape}"
            )
        fused = fuse_scores(
            features.visual_scores,
            semantic,
            mode=self.fusion_mode,
            alpha=self.fusion_alpha,
            visual_temperature=self.visual_temperature,
            semantic_temperature=self.semantic_temperature,
        )
        output: Dict[str, Any] = {
            "visual_logits": features.visual_scores,
            "semantic_logits": semantic,
            "final_logits": fused,
            "semantic_backend": self.semantic_backend,
            "fusion_mode": self.fusion_mode,
        }
        if transport is not None:
            output["transport"] = transport
        return output

    def forward(
        self,
        images: Tensor,
        labels: Tensor,
        *,
        stage_permutation: Optional[Sequence[int]] = None,
        return_aux: bool = False,
    ):
        features = self.extract_episode_features(images, labels)
        output = self.score_episode(features, stage_permutation=stage_permutation)
        if return_aux:
            output["features"] = features
            return output
        # P1/P2 fixed: visual first, semantic second, both [Q,C].
        return output["visual_logits"], output["semantic_logits"]


# Backward-compatible name used by the released run.py.
TaskAdapter = EpisodicTaskAdapter
