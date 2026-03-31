"""Shared data utilities used across DASVDD datasets."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor


def get_target_label_idx(labels: Sequence[int], targets: Sequence[int]) -> list[int]:
    """Return indices whose labels belong to the target set."""

    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: Tensor, scale: str = "l2") -> Tensor:
    """Apply global contrast normalization to an image tensor."""

    n_features = int(np.prod(x.shape))
    normalized = x - torch.mean(x)

    if scale == "l1":
        x_scale = torch.mean(torch.abs(normalized))
    elif scale == "l2":
        x_scale = torch.sqrt(torch.sum(normalized ** 2)) / n_features
    else:
        raise ValueError(f"Unsupported scale '{scale}'. Expected 'l1' or 'l2'.")

    return normalized / x_scale


def build_one_class_split(
    train_dataset: Iterable[tuple[Tensor, int]],
    test_dataset: Iterable[tuple[Tensor, int]],
    target_class: int,
) -> tuple[list[Tensor], list[Tensor], list[int]]:
    """Build one-class train/test splits with binary anomaly labels."""

    train_samples = [sample for sample, label in train_dataset if label == target_class]

    test_samples: list[Tensor] = []
    labels: list[int] = []
    for sample, label in test_dataset:
        test_samples.append(sample)
        labels.append(0 if label == target_class else 1)

    return train_samples, test_samples, labels


def OneClass(
    train_dataset: Iterable[tuple[Tensor, int]],
    test_dataset: Iterable[tuple[Tensor, int]],
    Class: int,
) -> tuple[list[Tensor], list[Tensor], list[int]]:
    """Backward-compatible wrapper for the original helper name."""

    return build_one_class_split(train_dataset, test_dataset, target_class=Class)
