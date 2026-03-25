import torch
from torch import nn

from data_loader import get_dataloaders
from model import MLP


def train(config):
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    train_loader, val_loader = get_dataloaders(
        data_cfg["csv_path"],
        data_cfg["batch_size"],
        data_cfg["train_fraction"],
        data_cfg["shuffle_train"],
        data_cfg["num_workers"],
    )

    # Get feature dim (e.g. 4) and target dim (e.g. 1) to build the model.
    example_features, example_targets = train_loader.dataset[0]
    model = MLP(
        example_features.shape[0],
        example_targets.shape[0],
        model_cfg["hidden_sizes"],
        model_cfg["activation"],
    )
    device = torch.device(train_cfg["device"])
    model = model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    checkpoint_path = train_cfg.get("checkpoint_path", "best_model.pt")
    best_val_loss = float("inf")

    for epoch in range(train_cfg["epochs"]):
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

        print(f"Epoch {epoch + 1}/{train_cfg['epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> saved checkpoint (val_loss={val_loss:.4f})")

    return model