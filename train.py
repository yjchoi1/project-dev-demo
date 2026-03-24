import torch
from torch import nn

from data_loader import get_dataloaders
from model import MLP


def train(config):
    train_loader, val_loader = get_dataloaders(
        config["csv_path"],
        config["batch_size"],
        config["train_fraction"],
        config["shuffle_train"],
        config["num_workers"],
    )

    # Get feature dim (e.g. 4) and target dim (e.g. 1) to build the model.
    example_features, example_targets = train_loader.dataset[0]
    model = MLP(
        example_features.shape[0],
        example_targets.shape[0],
        config["hidden_sizes"],
        config["activation"],
    )
    device = torch.device(config["device"])
    model = model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Training looop
    for _ in range(config["epochs"]):
        model.train()
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            predictions = model(batch_features)  # Forward pass
            loss = loss_fn(predictions, batch_targets)  # Compute loss
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()  # Backprop: compute d(loss)/d(weights).
            optimizer.step()  # Apply the update for this batch.

    return model
