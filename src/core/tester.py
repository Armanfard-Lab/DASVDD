"""Evaluation utilities for DASVDD."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from sklearn import metrics
from torch import Tensor, nn
from torch.utils.data import DataLoader


def _unwrap_batch(batch: Tensor | list[Tensor] | tuple[Tensor, ...]) -> Tensor:
    """Extract the feature tensor from a data-loader batch."""

    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def evaluate_dasvdd(
    model: nn.Module,
    center: Tensor,
    input_shape: int,
    gamma: float,
    test_loader: DataLoader,
    labels: Sequence[int],
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> float:
    """Evaluate DASVDD and return the ROC-AUC score as a percentage."""

    with torch.no_grad():
        anomaly_scores: list[float] = []

        for batch in test_loader:
            batch_features = _unwrap_batch(batch).view(-1, input_shape).to(device)
            reconstructed_batch, encoded_batch = model(batch_features)
            reconstruction_error = torch.mean((reconstructed_batch - batch_features) ** 2, dim=1)
            svdd_distance = torch.sum((encoded_batch - center) ** 2, dim=1)
            scores = reconstruction_error + gamma * svdd_distance
            anomaly_scores.extend(scores.detach().cpu().tolist())

    return float(metrics.roc_auc_score(labels, anomaly_scores) * 100)


def DASVDD_test(
    model: nn.Module,
    C: Tensor,
    in_shape: int,
    Gamma: float,
    test_loader: DataLoader,
    labels: Sequence[int],
    criterion: nn.Module,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> float:
    """Backward-compatible wrapper for older notebook-style imports."""

    return evaluate_dasvdd(
        model=model,
        center=C,
        input_shape=in_shape,
        gamma=Gamma,
        test_loader=test_loader,
        labels=labels,
        device=device,
    )
