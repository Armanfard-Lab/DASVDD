"""Gamma tuning utilities for DASVDD."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def _unwrap_batch(batch: Tensor | list[Tensor] | tuple[Tensor, ...]) -> Tensor:
    """Extract the feature tensor from a data-loader batch."""

    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def tune_gamma(
    autoencoder_cls: type[nn.Module],
    input_shape: int,
    criterion: nn.Module,
    train_loader: DataLoader,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    T: int = 10,
) -> float:
    """Estimate the gamma weighting used in the DASVDD objective."""

    gamma = torch.tensor(0.0, device=device)

    for _ in range(T):
        model = autoencoder_cls(input_shape=input_shape).to(device)
        mean_radius = torch.tensor(0.0, device=device)
        reconstruction_error = torch.tensor(0.0, device=device)

        for batch in train_loader:
            batch_features = _unwrap_batch(batch).view(-1, input_shape).to(device)
            reconstructed_batch, encoded_batch = model(batch_features)
            mean_radius += torch.sum(encoded_batch ** 2, dim=1).mean()
            reconstruction_error += criterion(reconstructed_batch, batch_features)

        mean_radius = mean_radius / len(train_loader)
        reconstruction_error = reconstruction_error / len(train_loader)
        gamma += reconstruction_error / mean_radius

    gamma = gamma / T
    return float(gamma.detach().item())
