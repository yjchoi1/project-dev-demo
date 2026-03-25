"""Tests for the MLP: wiring and tensor shapes (no training)."""

import torch
import torch.nn as nn

from model import MLP


def test_mlp_forward_output_shape():
    # A batch of 8 samples, each with 4 input features; model predicts 1 scalar per sample.
    batch_size, input_dim, output_dim = 8, 4, 1
    hidden_sizes = [16, 8]
    model = MLP(input_dim, output_dim, hidden_sizes, nn.ReLU)

    x = torch.randn(batch_size, input_dim)
    y = model(x)

    # Forward must preserve batch size and match declared output dimensionality.
    assert y.shape == (batch_size, output_dim)
