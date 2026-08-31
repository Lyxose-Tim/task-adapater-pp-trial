"""Deterministic class-major episodic sampling for direct-video datasets."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Sampler


class EpisodicBatchSampler(Sampler[List[int]]):
    """Yield ``n_way * (n_support+n_query)`` indices in class-major order."""

    def __init__(
        self,
        labels: Sequence[int],
        n_way: int,
        n_support: int,
        n_query: int,
        episodes: int,
        *,
        seed: int = 916,
    ) -> None:
        self.n_way = int(n_way)
        self.samples_per_class = int(n_support) + int(n_query)
        self.episodes = int(episodes)
        self.seed = int(seed)
        self.epoch = 0
        if min(self.n_way, self.samples_per_class, self.episodes) < 1:
            raise ValueError("n_way, samples per class, and episodes must be positive")
        grouped: Dict[int, List[int]] = defaultdict(list)
        for index, label in enumerate(labels):
            grouped[int(label)].append(index)
        self.by_class = {
            label: indices for label, indices in grouped.items()
            if len(indices) >= self.samples_per_class
        }
        self.classes = sorted(self.by_class)
        if len(self.classes) < self.n_way:
            raise ValueError(
                f"need {self.n_way} eligible classes with at least {self.samples_per_class} videos; "
                f"found {len(self.classes)}"
            )

    def __len__(self) -> int:
        return self.episodes

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic, epoch-specific episode stream."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
        class_tensor = torch.tensor(self.classes, dtype=torch.long)
        for _ in range(self.episodes):
            selected_positions = torch.randperm(len(self.classes), generator=generator)[:self.n_way]
            selected_classes = class_tensor[selected_positions].tolist()
            batch: List[int] = []
            for label in selected_classes:
                candidates = self.by_class[int(label)]
                positions = torch.randperm(len(candidates), generator=generator)[:self.samples_per_class]
                batch.extend(candidates[position] for position in positions.tolist())
            yield batch


def collate_episode(batch: Sequence[Any], n_way: int, samples_per_class: int):
    if len(batch) != n_way * samples_per_class:
        raise ValueError("episode batch size does not match n_way * samples_per_class")
    if isinstance(batch[0], Mapping):
        videos = torch.stack([item["video"] for item in batch])
        labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    else:
        videos = torch.stack([item[0] for item in batch])
        labels = torch.tensor([int(item[1]) for item in batch], dtype=torch.long)
    videos = videos.reshape(n_way, samples_per_class, *videos.shape[1:])
    labels = labels.reshape(n_way, samples_per_class)
    if not bool((labels == labels[:, :1]).all()):
        raise AssertionError("sampler/collator violated class-major episode layout")
    return videos, labels


def build_episode_loader(
    dataset: Any,
    *,
    n_way: int,
    n_support: int,
    n_query: int,
    episodes: int,
    num_workers: int = 8,
    seed: int = 916,
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    if not hasattr(dataset, "records"):
        raise TypeError("dataset must expose annotation records with labels")
    sampler = EpisodicBatchSampler(
        [record.label for record in dataset.records],
        n_way,
        n_support,
        n_query,
        episodes,
        seed=seed,
    )
    samples_per_class = n_support + n_query
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=partial(collate_episode, n_way=n_way, samples_per_class=samples_per_class),
    )


__all__ = ["EpisodicBatchSampler", "build_episode_loader", "collate_episode"]
