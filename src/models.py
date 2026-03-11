import torch
import torch.nn as nn


class QuadraticMLP(nn.Module):


    """A simple feedforward neural network with quadratic interactions between the layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list[int],
        output_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        layer_dims = [input_dim, *hidden_dim, output_dim]
        modules = []
        for index in range(len(layer_dims) - 1):
            modules.append(nn.Linear(layer_dims[index], layer_dims[index + 1]))
            if index < len(layer_dims) - 2:
                modules.append(nn.Tanh())

        self.network = nn.Sequential(*modules)

    def forward(self, 
                x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)
    

class LogisticRegression(nn.Module):


    """A simple logistic regression model."""
  
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.linear(x))
    
class SVM(nn.Module):


    """A simple class implementing Support Vector Machine as a Pytotch module."""

    pass