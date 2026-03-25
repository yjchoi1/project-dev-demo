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

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            predictions = model(batch_features)
            loss = loss_fn(predictions, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_features)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                predictions = model(batch_features)
                loss = loss_fn(predictions, batch_targets)
                val_loss += loss.item() * len(batch_features)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{config['epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    return model