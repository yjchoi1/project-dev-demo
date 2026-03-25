import json
import torch.nn as nn

from train import train

ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
    "GELU": nn.GELU,
    "SiLU": nn.SiLU,
    "ELU": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
    "Mish": nn.Mish,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        config = json.load(f)
    config["model"]["activation"] = ACTIVATIONS[config["model"]["activation"]]
    return config


if __name__ == "__main__":
    config = load_config("config.json")
    model = train(config)