"""Command-line entry point for running DASVDD experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.gamma_tune import tune_gamma
from src.core.tester import evaluate_dasvdd
from src.core.trainer import train_dasvdd
from src.data.dataset_loader import (
    load_cifar,
    load_fashion_mnist,
    load_mnist,
    load_pima,
    load_speech,
)
from src.models.cifar_net import CifarAutoencoder
from src.models.mnist_net import MnistAutoencoder
from src.models.pima_net import PimaAutoencoder
from src.models.speech_net import SpeechAutoencoder


@dataclass(frozen=True)
class DatasetSpec:
    """Describes how to prepare one supported DASVDD dataset."""

    loader_factory: Callable[[int, int, int], tuple[DataLoader, DataLoader, Sequence[int]]]
    model_factory: type[nn.Module]
    input_shape: int
    uses_target_class: bool = True


@dataclass(frozen=True)
class ExperimentSetup:
    """Prepared objects needed to train and evaluate DASVDD."""

    train_loader: DataLoader
    test_loader: DataLoader
    labels: Sequence[int]
    model: nn.Module
    input_shape: int
    code_size: int


DATASET_SPECS: dict[str, DatasetSpec] = {
    "MNIST": DatasetSpec(load_mnist, MnistAutoencoder, 28 * 28),
    "FMNIST": DatasetSpec(load_fashion_mnist, MnistAutoencoder, 28 * 28),
    "CIFAR": DatasetSpec(load_cifar, CifarAutoencoder, 32 * 32 * 3),
    "Speech": DatasetSpec(load_speech, SpeechAutoencoder, 400, uses_target_class=False),
    "PIMA": DatasetSpec(load_pima, PimaAutoencoder, 8, uses_target_class=False),
}


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the DASVDD CLI parser."""

    parser = argparse.ArgumentParser(description="Run DASVDD on one of the supported datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DATASET_SPECS.keys()),
        help="Dataset to use for training and evaluation.",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=0,
        help="Normal class label for one-class image datasets. Ignored for Speech and PIMA.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size used for training.")
    return parser


def load_experiment_setup(dataset_name: str, batch_size: int, target_class: int) -> ExperimentSetup:
    """Instantiate dataset loaders and the matching autoencoder model."""

    dataset_spec = DATASET_SPECS[dataset_name]
    class_value = target_class if dataset_spec.uses_target_class else 0
    train_loader, test_loader, labels = dataset_spec.loader_factory(batch_size, 1, class_value)
    model = dataset_spec.model_factory(input_shape=dataset_spec.input_shape)
    code_size = int(getattr(model, "code_size"))
    return ExperimentSetup(
        train_loader=train_loader,
        test_loader=test_loader,
        labels=labels,
        model=model,
        input_shape=dataset_spec.input_shape,
        code_size=code_size,
    )


def main() -> None:
    """Run DASVDD training and evaluation from the command line."""

    args = build_argument_parser().parse_args()

    print(f"\nStarting DASVDD on the {args.dataset} dataset...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup = load_experiment_setup(args.dataset, args.batch_size, args.target_class)

    model = setup.model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    center = torch.randn(setup.code_size, device=device, requires_grad=True)
    center_optimizer = torch.optim.Adagrad([center], lr=1, lr_decay=0.01)

    print("Tuning gamma...")
    gamma = tune_gamma(model.__class__, setup.input_shape, criterion, setup.train_loader, device=device, T=10)

    print("Training DASVDD...")
    train_dasvdd(
        model=model,
        input_shape=setup.input_shape,
        center=center,
        train_loader=setup.train_loader,
        optimizer=optimizer,
        center_optimizer=center_optimizer,
        criterion=criterion,
        gamma=gamma,
        device=device,
        num_epochs=args.epochs,
        support_fraction=0.9,
        verbosity=1,
    )

    print("\nEvaluating...")
    auc = evaluate_dasvdd(
        model=model,
        center=center,
        input_shape=setup.input_shape,
        gamma=gamma,
        test_loader=setup.test_loader,
        labels=setup.labels,
        device=device,
    )
    print(f"ROC-AUC: {auc:.2f}")


if __name__ == "__main__":
    main()
