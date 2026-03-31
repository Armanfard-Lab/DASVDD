"""Training utilities for DASVDD."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def _unwrap_batch(batch: Tensor | list[Tensor] | tuple[Tensor, ...]) -> Tensor:
    """Extract the feature tensor from a batch produced by a data loader."""

    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def _mean_squared_distance(encoded_batch: Tensor, center: Tensor) -> Tensor:
    """Compute the mean squared distance of encoded samples to the SVDD center."""

    return torch.sum((encoded_batch - center) ** 2, dim=1).mean()


def train_dasvdd(
    model: nn.Module,
    input_shape: int,
    center: Tensor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    center_optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    gamma: float,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_epochs: int = 300,
    support_fraction: float = 0.9,
    verbosity: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train DASVDD and return loss traces for optional analysis."""

    total_losses = np.zeros(num_epochs)
    reconstruction_losses = np.zeros(num_epochs)
    svdd_losses = np.zeros(num_epochs)
    center_history = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_reconstruction_loss = 0.0
        epoch_svdd_loss = 0.0

        for batch in train_loader:
            batch_features = _unwrap_batch(batch).view(-1, input_shape).to(device)
            support_batch_size = max(1, int(np.ceil(support_fraction * batch_features.size(0))))
            support_batch = batch_features[:support_batch_size, :]
            center_batch = batch_features[support_batch_size:, :]

            if center_batch.size(0) == 0:
                center_batch = support_batch

            optimizer.zero_grad()
            reconstructed_batch, encoded_batch = model(support_batch)
            svdd_radius = _mean_squared_distance(encoded_batch, center)
            reconstruction_term = criterion(reconstructed_batch, support_batch)
            train_loss = reconstruction_term + gamma * svdd_radius
            train_loss.backward()
            optimizer.step()

            epoch_total_loss += float(train_loss.item())
            epoch_reconstruction_loss += float(reconstruction_term.item())
            epoch_svdd_loss += float(svdd_radius.item())

            center_optimizer.zero_grad()
            _, center_codes = model(center_batch)
            target_center = torch.mean(center_codes, dim=0)
            center_loss = criterion(center, target_center)
            center_loss.backward()
            center_optimizer.step()

            center_history[epoch] += float(center[0].detach().cpu())

        num_batches = len(train_loader)
        total_losses[epoch] = epoch_total_loss / num_batches
        reconstruction_losses[epoch] = epoch_reconstruction_loss / num_batches
        svdd_losses[epoch] = epoch_svdd_loss / num_batches
        center_history[epoch] = center_history[epoch] / num_batches

        if verbosity == 1:
            print(f"epoch : {epoch + 1}/{num_epochs}, loss = {total_losses[epoch]:.6f}")

    return total_losses, reconstruction_losses, svdd_losses, center_history


def DASVDD_trainer(
    model: nn.Module,
    in_shape: int,
    code_size: int,
    C: Tensor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    update_center: torch.optim.Optimizer,
    criterion: nn.Module,
    Gamma: float,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_epochs: int = 300,
    K: float = 0.9,
    verbosity: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible wrapper for older notebook-style imports."""

    return train_dasvdd(
        model=model,
        input_shape=in_shape,
        center=C,
        train_loader=train_loader,
        optimizer=optimizer,
        center_optimizer=update_center,
        criterion=criterion,
        gamma=Gamma,
        device=device,
        num_epochs=num_epochs,
        support_fraction=K,
        verbosity=verbosity,
    )
