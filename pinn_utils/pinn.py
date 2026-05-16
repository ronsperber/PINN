"""
module to create PINN for solving an ODE
and function to use that PINN to solution
"""
import torch
import torch.nn as nn
from typing import List, Callable

class PINN(nn.Module):
    """
    feed forward network intended to be used for solving DEs
    """
    def __init__(self,
                 num_hidden_layers:int = 2,
                 layer_width:int = 64,
                 input_activation: Callable[[torch.Tensor], torch.Tensor] = nn.Tanh(),
                 hidden_activation: Callable[[torch.Tensor], torch.Tensor] | List[Callable[[torch.Tensor], torch.Tensor]] = nn.Tanh(),
                 output_activation: Callable[[torch.Tensor], torch.Tensor] = nn.Identity(),
                 num_inputs:int = 1,
                 num_outputs:int = 1
    ) -> None:
        """
        Parameters
        num_hidden_layers: int
            number of layers between output layer and input layer
        layer_width: int
            number of neurons in each layer
        input_activation : Callable 
            activation function to be used on input layer
        hidden_activation: Callable of List of Callables
            Activation function(s) for hidden layers
            If a single function is provided it is applied to all layers
            If a list is provided, it should have length equal to num_hidden_layers
        output_activation: Callable
            activation function for output layer
        num_inputs
            number of independent variables in DE
        num_outputs
            number of equations in a system of DEs
        """
        super(PINN, self).__init__()
        if not isinstance (hidden_activation, list):
            hidden_activation = [hidden_activation] * num_hidden_layers
        if len(hidden_activation) != num_hidden_layers:
            raise ValueError(
                f"Length of hidden_activation ({len(hidden_activation)}) "
                f"must match num_hidden_layers ({num_hidden_layers}). "
                f"Either pass a single activation or a list of {num_hidden_layers} activations."
                )
        self.input_activation = input_activation
        self.hidden_activation = hidden_activation
        self.input_layer = nn.Linear(num_inputs, layer_width)
        self.output_layer = nn.Linear(layer_width, num_outputs)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(layer_width, layer_width) for _ in range(num_hidden_layers)]
        )
        self.output_activation = output_activation
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through input, hidden, and output layers."""
        x = self.input_activation(self.input_layer(x))
        for i,layer in enumerate(self.hidden_layers):
            x = self.hidden_activation[i](layer(x))
        x = self.output_layer(x)
        return self.output_activation(x)
