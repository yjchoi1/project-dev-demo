"""Tests that JSON config loads and non-JSON fields (e.g. activation) are resolved."""

import json

import torch.nn as nn

from main import load_config


def test_load_config_maps_activation_string_to_module_class(tmp_path):
    # Minimal valid config matching the nested keys expected by train().
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "data": {
                    "csv_path": "dummy.csv",
                    "train_fraction": 0.8,
                    "batch_size": 8,
                    "shuffle_train": False,
                    "num_workers": 0,
                },
                "model": {"hidden_sizes": [8], "activation": "GELU"},
                "training": {"epochs": 1, "lr": 0.01, "device": "cpu"},
            }
        )
    )

    config = load_config(str(cfg_path))

    # After load_config, activation must be a class PyTorch can instantiate (e.g. nn.GELU), not a string.
    assert config["model"]["activation"] is nn.GELU
