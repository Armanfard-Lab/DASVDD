# DASVDD

Implementation of **DASVDD (Deep Autoencoding Support Vector Data Descriptor)** for anomaly detection and one-class classification.

The project combines dataset-specific autoencoders with an SVDD-inspired objective to learn compact representations of normal data and score anomalies through reconstruction quality and distance to a learned center.

## Highlights

- Supports `MNIST`, `FMNIST`, `CIFAR`, `Speech`, and `PIMA`
- Includes dataset-specific autoencoder architectures
- Tunes the SVDD weighting term before training
- Uses deterministic preprocessing for reproducible runs
- Provides a simple command-line interface for running experiments

## Repository layout

```text
.
├── main.py               # CLI entry point
├── src/
│   ├── core/             # Training, evaluation, and gamma tuning
│   ├── data/             # Dataset loaders and preprocessing helpers
│   └── models/           # Dataset-specific autoencoders
├── data/                 # Local tabular datasets
└── requirements.txt      # Python dependencies
```

## Installation

```bash
git clone https://github.com/Armanfard-Lab/DASVDD.git
cd DASVDD

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run DASVDD on any supported dataset:

```bash
python3 main.py --dataset MNIST --target_class 0 --epochs 30 --batch_size 128
```

Example runs:

```bash
python3 main.py --dataset FMNIST --target_class 3 --epochs 20
python3 main.py --dataset CIFAR --target_class 1 --epochs 10
python3 main.py --dataset PIMA --epochs 30
python3 main.py --dataset Speech --epochs 30
```

For `Speech` and `PIMA`, `--target_class` is ignored because the datasets already include anomaly labels.

## Command-line arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--dataset` | Dataset to use for training and evaluation | Required |
| `--target_class` | Normal class for one-class image datasets | `0` |
| `--epochs` | Number of training epochs | `30` |
| `--batch_size` | Training mini-batch size | `128` |

## Datasets

- `MNIST` and `FMNIST` are downloaded automatically through `torchvision`.
- `CIFAR` is downloaded automatically and normalized with global contrast normalization.
- `Speech` and `PIMA` are loaded from the local `data/` directory.
- Tabular datasets are shuffled with a fixed seed before splitting so evaluation is reproducible without depending on CSV row order.

## Development notes

- The refactored modules expose clearer APIs under `src.core`, `src.data`, and `src.models`.
- Backward-compatible aliases such as `DASVDD_trainer`, `DASVDD_test`, and `AE_MNIST` are still available for older notebooks or scripts.

## Output

The CLI prints progress for:

- gamma tuning
- DASVDD training
- final `ROC-AUC` evaluation

## Citation

If you use this repository in academic work, please cite:

```bibtex
@ARTICLE{DASVDD,
  author={Hojjati, Hadi and Armanfard, Narges},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  title={DASVDD: Deep Autoencoding Support Vector Data Descriptor for Anomaly Detection},
  year={2024},
  volume={36},
  number={8},
  pages={3739-3750},
  keywords={Anomaly detection;Training;Task analysis;Support vector machines;Image reconstruction;Data models;Benchmark testing;Anomaly detection;deep autoencoder;deep learning;support vector data descriptor},
  doi={10.1109/TKDE.2023.3328882}
}
```

Paper link: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10314785)
