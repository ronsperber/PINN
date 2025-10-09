import torch
import torch.nn as nn
from typing import List, Callable, Union, Optional

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
          x: torch.Tensor,
          epochs: int = 5000,
          lr: float = 1e-3,
          batch_size: int = None,
          print_every: int = 500,
          val_size: Union[float, int] = 0.2,
          early_stopping: Optional[dict] = None
          ):
    loss_fn = get_loss(a, ics, NN, F)
    n_points = x.shape[0]
    if isinstance(val_size, int):
        if val_size <=0 or val_size >= n_points:
            raise ValueError("Not a valid validation size")
    elif isinstance(val_size, float):
        if val_size <=0 or val_size >= 1:
            raise ValueError("Not a valid validation size")
        else:
            val_size = int(n_points * val_size)
    else:
        raise TypeError("Validation size must be int or float")
    if x.requires_grad:
        x = x.detach().clone().requires_grad_(False)
    perm = torch.randperm(n_points)
    x_val = x[perm[:val_size]].requires_grad_(True)
    x_train = x[perm[val_size:]].requires_grad_(True)


    n_train = x_train.shape[0]
    if early_stopping is not None:
        min_epochs = early_stopping.get('min_epochs', 20)
        patience = early_stopping.get('patience', 10)
        best_val_loss = float('inf')
        epochs_since_best = 0
        best_weights = None
    optimizer = torch.optim.Adam(params=NN.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        NN.train()
        if batch_size is None:
            optimizer.zero_grad()
            loss = loss_fn(x_train)
            loss.backward()
            epoch_loss = loss.item()
            optimizer.step()
        else:
            # shuffle every epoch to get new batches
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            for i in range(0, n_train, batch_size):
                optimizer.zero_grad()
                end = min(i+batch_size, n_train)
                idx = perm[i:end]
                x_batch = x_train[idx]
                loss = loss_fn(x_batch)
                loss.backward()
                epoch_loss += loss.item() * len(idx)
                optimizer.step()
            epoch_loss /= n_train
        NN.eval()
        val_loss = loss_fn(x_val).item()
        if early_stopping is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_best = 0
                best_weights = {k: v.clone().detach() for k, v in NN.state_dict().items()}
            else:
                epochs_since_best += 1
            if epochs_since_best >= patience and epoch >= min_epochs:
                print(f"Early stopping at epoch {epoch}, validation loss did not improve for {early_stopping['patience']} epochs")
                NN.load_state_dict(best_weights)
                break
        NN.train()
        if epoch % print_every == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}")
    print(f"Final Epoch {epoch}, Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}")
    NN.eval()
    y_trial_grad = get_y_trial(a, ics, NN)

    def y_trial(x):
        with torch.no_grad():
            return y_trial_grad(x)
    return y_trial



