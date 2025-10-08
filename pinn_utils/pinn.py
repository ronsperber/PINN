import torch
import torch.nn as nn
from typing import List, Callable

def factorials_up_to_n(n :int) -> torch.Tensor:
    """
    returns a tensor of factorials
    from 0! to n!
    """
    if n < 0:
        raise ValueError("Factorials only have non-negative arguments")
    facs = torch.ones(n + 1, dtype=torch.float32)
    for k in range(1, n + 1):
        facs[k] = facs[k-1] * k
    return facs

def taylor_polynomial(a: float, ics: List[float]):
    """
    Generate Taylor polynomial centered at a
    of degree len(ics) - 1 using the ics to get coefficients
    """
    n = len(ics)
    facs = factorials_up_to_n(n)

    def g(x):
        result = torch.zeros_like(x)
        for k, yk in enumerate(ics):
            result += yk / facs[k] * (x - a)**k
        return result
    return g

def get_y_trial(a: float, ics: List[float], NN: nn.Module):
    """
    get a function y_trial that isolates NN(x)
    from the initial conditions
    """
    n = len(ics)
    poly = taylor_polynomial(a, ics)
    def y_trial(x):
        return poly(x) + (x-a)**n * NN(x)
    
    return y_trial

class PINN(nn.Module):
    """
    feed forward network intended to be used for solving DEs
    """
    def __init__(self,
                 num_hidden_layers:int = 2,
                 layer_width:int = 64,
                 activation: Callable[[torch.Tensor], torch.Tensor] = nn.Tanh(),
                 output_activation: Callable[[torch.Tensor], torch.Tensor] = nn.Identity(),
                 num_inputs:int = 1,
                 num_outputs:int = 1
    ):
        """
        Parameters
        num_hidden_layers: int
            number of layers between output layer and input layer
        layer_width: int
            number of neurons in each layer
        activation : Callable
            activation function to be used on layers
            other than the output layer
        output_activation: Callable
            activation function for output layer
        num_inputs
            number of independent variables in DE
        num_outputs
            number of equations in a system of DEs
        """
        super(PINN, self).__init__()
        self.activation = activation
        self.input_layer = nn.Linear(num_inputs, layer_width)
        self.output_layer = nn.Linear(layer_width, num_outputs)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(layer_width, layer_width) for _ in range(num_hidden_layers)]
        )
        self.output_activation = output_activation
        

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return self.output_activation(x)

def derivatives(y:torch.Tensor, x:torch.Tensor, order:int) -> List[torch.Tensor]:
    """
    function to return [y,y',y'', ... y^(order)]
    to be used in a differential equation F(x,y,y'..) = 0
    """
    derivs = [y]  # 0th derivative
    for _ in range(order):
        dy_dx = torch.autograd.grad(
            outputs=derivs[-1],
            inputs=x,
            grad_outputs=torch.ones_like(derivs[-1]),
            create_graph=True
        )[0]
        derivs.append(dy_dx)
    return derivs  # [y, y', y'', ..., y^(order)]

def get_loss(a: float, ics: List[float], NN:nn.Module, F:Callable) ->Callable:
    """
    generating the loss function 
    based on F(x,y,y',...y^n) = 0
    """
    y_trial_fn = get_y_trial(a, ics, NN)
    def loss(x):
        y0 = y_trial_fn(x)
        derivs = derivatives(y0, x, len(ics))
        residual = F(x, *derivs)
        return torch.mean(residual**2)
    
    return loss

def solve(F: Callable,
          a : float,
          ics: List[float],
          NN: PINN,
          x_train: torch.Tensor,
          epochs: int = 5000,
          lr: float = 1e-3
          ):
    loss_fn = get_loss(a, ics, NN, F)
    optimizer = torch.optim.Adam(params=NN.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(x_train)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    return get_y_trial(a, ics, NN)
    



