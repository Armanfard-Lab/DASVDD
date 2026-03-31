"""Dataset loading utilities for DASVDD.

Tabular datasets are shuffled with a fixed random seed before splitting to
avoid order-dependent evaluation artifacts while keeping runs reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from src.data.utils import build_one_class_split, global_contrast_normalization


IMAGE_DATASET_ROOT = "~/torch_datasets"
TABULAR_DATASET_ROOT = "data"
TABULAR_SPLIT_SEED = 42


def _shuffle_frame(frame: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Return a shuffled copy of a tabular dataset."""

    return frame.sample(frac=1, random_state=random_state).reset_index(drop=True)


def _load_one_class_image_dataset(
    dataset_class: type[torchvision.datasets.VisionDataset],
    train_batch: int,
    test_batch: int,
    target_class: int,
    transform: transforms.Compose,
) -> tuple[DataLoader, DataLoader, list[int]]:
    """Create one-class train and test loaders for an image dataset."""

    train_dataset = dataset_class(root=IMAGE_DATASET_ROOT, train=True, transform=transform, download=True)
    test_dataset = dataset_class(root=IMAGE_DATASET_ROOT, train=False, transform=transform, download=True)

    train_samples, test_samples, labels = build_one_class_split(train_dataset, test_dataset, target_class)
    train_loader = DataLoader(
        train_samples,
        batch_size=train_batch,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(test_samples, batch_size=test_batch, shuffle=False, num_workers=2)
    return train_loader, test_loader, labels


def _split_tabular_dataset(
    data_path: str,
    label_column: int,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, Sequence[int]]:
    """Load, shuffle, and split a tabular dataset into train and test partitions."""

    data = _shuffle_frame(pd.read_csv(data_path, header=None), random_state=TABULAR_SPLIT_SEED)
    labels = data[label_column].copy()
    features = data.drop(columns=label_column)

    split_index = int(len(data) * test_fraction)
    x_test = features[:split_index]
    x_train = features[split_index:]
    y_test = labels[:split_index]
    return x_train, x_test, y_test.tolist()


def _build_tabular_loaders(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: Sequence[int],
    train_batch: int,
    test_batch: int,
) -> tuple[DataLoader, DataLoader, Sequence[int]]:
    """Wrap tabular splits in PyTorch data loaders."""

    train_dataset = TensorDataset(torch.tensor(x_train.values, dtype=torch.float32))
    test_dataset = TensorDataset(
        torch.tensor(x_test.values, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False)
    return train_loader, test_loader, y_test


def load_mnist(train_batch: int, test_batch: int, target_class: int) -> tuple[DataLoader, DataLoader, list[int]]:
    """Load MNIST in one-class format."""

    transform = transforms.Compose([transforms.ToTensor()])
    return _load_one_class_image_dataset(
        torchvision.datasets.MNIST,
        train_batch,
        test_batch,
        target_class,
        transform,
    )


def load_fashion_mnist(
    train_batch: int,
    test_batch: int,
    target_class: int,
) -> tuple[DataLoader, DataLoader, list[int]]:
    """Load Fashion-MNIST in one-class format."""

    transform = transforms.Compose([transforms.ToTensor()])
    return _load_one_class_image_dataset(
        torchvision.datasets.FashionMNIST,
        train_batch,
        test_batch,
        target_class,
        transform,
    )


def load_cifar(train_batch: int, test_batch: int, target_class: int) -> tuple[DataLoader, DataLoader, list[int]]:
    """Load CIFAR-10 in one-class format with contrast normalization."""

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: global_contrast_normalization(x, scale="l2")),
        ]
    )
    return _load_one_class_image_dataset(
        torchvision.datasets.CIFAR10,
        train_batch,
        test_batch,
        target_class,
        transform,
    )


def load_speech(
    train_batch: int,
    test_batch: int,
    target_class: int = 0,
) -> tuple[DataLoader, DataLoader, Sequence[int]]:
    """Load the Speech dataset from the local data directory."""

    del target_class
    x_train, x_test, y_test = _split_tabular_dataset(f"{TABULAR_DATASET_ROOT}/speech.csv", label_column=400, test_fraction=0.1)
    return _build_tabular_loaders(x_train, x_test, y_test, train_batch, test_batch)


def load_pima(
    train_batch: int,
    test_batch: int,
    target_class: int = 0,
) -> tuple[DataLoader, DataLoader, Sequence[int]]:
    """Load the PIMA dataset from the local data directory."""

    del target_class
    x_train, x_test, y_test = _split_tabular_dataset(f"{TABULAR_DATASET_ROOT}/pima.csv", label_column=8, test_fraction=0.4)
    return _build_tabular_loaders(x_train, x_test, y_test, train_batch, test_batch)


def MNIST_loader(train_batch: int, test_batch: int, Class: int) -> tuple[DataLoader, DataLoader, list[int]]:
    """Backward-compatible wrapper for the original loader name."""

    return load_mnist(train_batch, test_batch, Class)


def FMNIST_loader(train_batch: int, test_batch: int, Class: int) -> tuple[DataLoader, DataLoader, list[int]]:
    """Backward-compatible wrapper for the original loader name."""

    return load_fashion_mnist(train_batch, test_batch, Class)


def CIFAR_loader(train_batch: int, test_batch: int, Class: int) -> tuple[DataLoader, DataLoader, list[int]]:
    """Backward-compatible wrapper for the original loader name."""

    return load_cifar(train_batch, test_batch, Class)


def Speech_loader(train_batch: int, test_batch: int) -> tuple[DataLoader, DataLoader, Sequence[int]]:
    """Backward-compatible wrapper for the original loader name."""

    return load_speech(train_batch, test_batch)


def PIMA_loader(train_batch: int, test_batch: int) -> tuple[DataLoader, DataLoader, Sequence[int]]:
    """Backward-compatible wrapper for the original loader name."""

    return load_pima(train_batch, test_batch)
