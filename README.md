# MLP Regression

A minimal PyTorch MLP for regression on tabular CSV data.

## Project Structure

```
├── config.json       # Hyperparameters and paths
├── main.py           # Entry point
├── train.py          # Training and validation loop
├── model.py          # MLP architecture
├── data_loader.py    # Dataset and DataLoader
└── data/
    └── dataset.csv   # Input data (features + target in last column)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Edit `config.json` to adjust hyperparameters, then run:

```bash
python main.py
```

## Config

| Section    | Key              | Description                              |
|------------|------------------|------------------------------------------|
| `data`     | `csv_path`       | Path to CSV file                         |
|            | `train_fraction` | Fraction of data used for training       |
|            | `batch_size`     | Mini-batch size                          |
| `model`    | `hidden_sizes`   | List of hidden layer widths              |
|            | `activation`     | Activation function (`ReLU`, `Tanh`, …)  |
| `training` | `epochs`         | Number of training epochs                |
|            | `lr`             | Adam learning rate                       |
|            | `device`         | `cpu` or `cuda`                          |
