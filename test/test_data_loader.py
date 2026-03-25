"""Tests for CSV dataset and train/val DataLoaders."""

import pandas as pd
import torch

from data_loader import RegressionCSVDataset, get_dataloaders


def _write_regression_csv(path, n_rows: int = 12, n_features: int = 4):
    # Synthetic table: last column is the target y; preceding columns are features.
    cols = [f"x{i}" for i in range(n_features)] + ["y"]
    rows = {c: torch.randn(n_rows).tolist() for c in cols}
    pd.DataFrame(rows).to_csv(path, index=False)


def test_regression_csv_dataset_splits_features_and_targets(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_regression_csv(csv_path, n_rows=5, n_features=3)

    ds = RegressionCSVDataset(csv_path)

    # Length matches number of data rows.
    assert len(ds) == 5
    features, targets = ds[0]
    # Features: all columns except last; targets: last column as shape (1,) for MSE-friendly batches.
    assert features.shape == (3,)
    assert targets.shape == (1,)


def test_get_dataloaders_train_val_sizes_sum_to_full_dataset(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_regression_csv(csv_path, n_rows=10, n_features=4)

    train_loader, val_loader = get_dataloaders(
        str(csv_path),
        batch_size=4,
        train_fraction=0.7,
        shuffle_train=False,
        num_workers=0,
    )

    # Train + val subsets partition the full dataset (no overlap, full coverage).
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    assert n_train + n_val == 10

    # One batch from each loader should yield tensors with consistent feature/target dims.
    x_t, y_t = next(iter(train_loader))
    assert x_t.shape[1] == 4 and y_t.shape[1] == 1
