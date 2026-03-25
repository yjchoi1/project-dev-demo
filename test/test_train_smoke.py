"""Smoke test: full train() runs for one epoch on tiny data (checks end-to-end wiring)."""

import json

import pandas as pd
import torch

from main import load_config
from train import train


def test_train_one_epoch_completes(tmp_path):
    # Tiny CSV so the loop is fast; same layout as the real project (features + y).
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "x1": [0.0, 1.0, 0.5, -1.0, 2.0],
            "x2": [1.0, 0.0, 0.5, 1.0, -0.5],
            "x3": [0.0, 0.0, 1.0, 1.0, 0.0],
            "x4": [-1.0, 1.0, 0.0, 0.5, 0.5],
            "y": [0.1, 0.2, 0.15, 0.05, 0.25],
        }
    ).to_csv(csv_path, index=False)

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "data": {
                    "csv_path": str(csv_path),
                    "train_fraction": 0.8,
                    "batch_size": 2,
                    "shuffle_train": False,
                    "num_workers": 0,
                },
                "model": {"hidden_sizes": [8], "activation": "ReLU"},
                "training": {"epochs": 1, "lr": 0.01, "device": "cpu"},
            }
        )
    )

    config = load_config(str(cfg_path))
    model = train(config)

    # If we get here, optimizer stepped without shape/device errors; model is a trained module.
    assert isinstance(model, torch.nn.Module)
