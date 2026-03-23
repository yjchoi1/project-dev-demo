import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class RegressionCSVDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        data = torch.tensor(df.values, dtype=torch.float32)
        self.features = data[:, :-1]
        self.targets = data[:, -1:]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.targets[i]


def get_dataloaders(csv_path, batch_size, train_fraction, shuffle_train, num_workers):
    full = RegressionCSVDataset(csv_path)
    n_train = int(train_fraction * len(full))
    n_val = len(full) - n_train
    train_ds, val_ds = random_split(full, [n_train, n_val])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
