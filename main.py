import torch.nn as nn

from train import train

config = {
    "data": {
        "csv_path": "data/dataset.csv",
        "train_fraction": 0.8,
        "batch_size": 32,
        "shuffle_train": True,
        "num_workers": 0,
    },
    "model": {
        "hidden_sizes": [64, 64],
        "activation": nn.ReLU,
    },
    "training": {
        "epochs": 50,
        "lr": 1e-3,
        "device": "cpu",
    },
}

if __name__ == "__main__":
    model = train(config)