"""
module with functions to compute
derivatives, both ordinary and partial
and the jacobian if desired
"""
import torch
from itertools import combinations_with_replacement
from typing import List, Callable

def derivatives(y: torch.Tensor, x: torch.Tensor, order: int) -> List[torch.Tensor]:
    """
    function to return [y, y', y'', ... y^(order)]
    to be used in a differential equation F(x, y, y'..) = 0
    """
    derivs = [y]  # 0th derivative
    
    for _ in range(order):
        current = derivs[-1]
        
        # Handle multi-dimensional output
        if current.ndim > 1 and current.shape[1] > 1:
            # Compute derivative of each output component separately
            dy_dx_components = []
            for i in range(current.shape[1]):
                # compute dy_i/dx
                grad = torch.autograd.grad(
                    outputs=current[:, i],
                    inputs=x,
                    grad_outputs=torch.ones(current.shape[0], device=current.device),
                    create_graph=True,
                    retain_graph=True
                )[0]
                dy_dx_components.append(grad)
            # turn dy/dx into a tensor 
            dy_dx = torch.cat(dy_dx_components, dim=1)
        else:
            # Scalar case
            dy_dx = torch.autograd.grad(
                outputs=current,
                inputs=x,
                grad_outputs=torch.ones_like(current),
                create_graph=True
            )[0]
        
        derivs.append(dy_dx)
    
    return derivs  # [y, y', y'', ..., y^(order)]


def compute_unique_derivatives(f, x, order=2):
    """
    Compute all unique derivatives of f output up to a given order.
    Mixed partials are treated as equal (∂²/∂x∂y = ∂²/∂y∂x).

    Parameters
    ----------
    f: Callable | nn.Module
        function for which the derivative will be computed
    x: torch.Tensor of shape (batch_size, num_inputs), requires_grad=True
        tensor at which the derivatives of f are to be evaluated
    order: int
        maximum order of derivatives to compute

    Returns
    -------
    derivs_per_output: list of dicts, one per output
        Each dict maps derivative name -> tensor (batch_size,)
    """
    _ , num_inputs = x.shape
    y = f(x)
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    num_outputs = y.shape[1]

    derivs_per_output = []

    for j in range(num_outputs):
        yj = y[:, j]
        deriv_dict = {'y': yj}

        # Compute and store first-order derivatives
        first_order = [torch.autograd.grad(
            yj, x, grad_outputs=torch.ones_like(yj),
            create_graph=True, retain_graph=True
        )[0][:, i] for i in range(num_inputs)]
        for i, grad in enumerate(first_order):
            deriv_dict[f"x{i}"] = grad

        # Store for recursive construction
        prev_derivs = { (i,): first_order[i] for i in range(num_inputs) }

        # Higher-order derivatives
        for ord in range(2, order + 1):
            new_derivs = {}
            for combo in combinations_with_replacement(range(num_inputs), ord):
                shorter = combo[:-1]
                last = combo[-1]
                grad_prev = prev_derivs[shorter]
                grad = torch.autograd.grad(
                    grad_prev, x, grad_outputs=torch.ones_like(grad_prev),
                    create_graph=True, retain_graph=True
                )[0][:, last]
                name = "_".join(f"x{i}" for i in combo)
                deriv_dict[name] = grad
                new_derivs[combo] = grad
            prev_derivs = new_derivs

        derivs_per_output.append(deriv_dict)

    return derivs_per_output


def jacobian(u, X):
    """
    Computes the Jacobian J for a function u: R^m -> R^n evaluated at X.
    
    Parameters
    ----------
    u: callable
        returns shape (batch_size, n)
    X: tensor 
        inputs to u of shape (batch_size, m)
    
    Returns
    -------
    J: tensor of shape (batch_size, n, m)
        J[i,j,k] = d(u_j)/d(x_k) evaluated at batch element i
    """
    derivs_list = compute_unique_derivatives(u, X, order=1)  # list of dicts, length n
    n = len(derivs_list)
    m = len(derivs_list[0]) - 1  # number of input variables
    batch_size = X.shape[0]

    # initialize a tensor to hold the Jacobian
    J = torch.zeros(batch_size, n, m, device=X.device, dtype=X.dtype)

    # stack partial derivatives into the Jacobian
    for i, derivs in enumerate(derivs_list):      # over outputs
         for j, key in enumerate(sorted(k for k in derivs.keys() if k != 'y')):
            J[:, i, j] = derivs[key].squeeze(-1)  # make sure it's (batch_size,)

    return J