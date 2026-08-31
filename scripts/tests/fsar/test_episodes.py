from dataclasses import dataclass

import torch

from fsar.episodes import EpisodicBatchSampler, collate_episode


def test_sampler_is_class_major_and_reproducible():
    labels = [0] * 4 + [1] * 4 + [2] * 4
    left = list(EpisodicBatchSampler(labels, 2, 1, 1, 3, seed=11))
    right = list(EpisodicBatchSampler(labels, 2, 1, 1, 3, seed=11))
    assert left == right
    for batch in left:
        chosen = [labels[index] for index in batch]
        assert chosen[0] == chosen[1]
        assert chosen[2] == chosen[3]
        assert chosen[0] != chosen[2]


def test_sampler_is_reproducible_but_changes_between_epochs():
    labels = [0] * 8 + [1] * 8 + [2] * 8 + [3] * 8
    sampler = EpisodicBatchSampler(labels, 3, 1, 1, 5, seed=11)
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)
    replay = EpisodicBatchSampler(labels, 3, 1, 1, 5, seed=11)
    replay.set_epoch(1)

    assert epoch_zero != epoch_one
    assert epoch_one == list(replay)


def test_collate_episode_shapes_and_labels():
    batch = [
        (torch.full((8, 3, 2, 2), float(label)), label)
        for label in (4, 4, 9, 9)
    ]
    videos, labels = collate_episode(batch, 2, 2)
    assert videos.shape == (2, 2, 8, 3, 2, 2)
    assert labels.tolist() == [[4, 4], [9, 9]]
