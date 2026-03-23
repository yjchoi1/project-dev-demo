import torch.nn as nn


class MLP(nn.Module):
    """Fully-connected network for regression or classification.

    Stacks linear layers with the same activation after each hidden layer.
    The final layer has no activation.
    """

    def __init__(self, input_dim, output_dim, hidden_sizes, activation):
        """Builds the MLP.

        Args:
            input_dim: Size of the input feature vector.
            output_dim: Size of the output (e.g. 1 for scalar regression).
            hidden_sizes: Iterable of hidden layer widths, in order.
            activation: Callable with no arguments that returns an activation
                module instance, e.g. ``nn.ReLU``, ``nn.Tanh``, ``nn.GELU``,
                ``nn.SiLU``, ``nn.LeakyReLU``, ``nn.ELU``, ``nn.Mish``.
        """
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_d, h))
            layers.append(activation())
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Runs the forward pass.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Output tensor of shape ``(batch, output_dim)``.
        """
        return self.net(x)
