"""
module to solve PDEs 
or systems of PDEs
"""
import torch
from typing import List, Callable
from .pinn import PINN
from .training import train 

def get_pde_loss(NN, de_eq, ic_conditions=None, bc_conditions=None, DE_params=None):
    """
    Returns a loss function compatible with your `train()` loop.

    Parameters
    ----------
    NN : nn.Module
        The neural network model.
    de_eq : Callable[[nn.Module, torch.Tensor], torch.Tensor]
        PDE residual function (e.g. lambda NN, x: u_t - alpha * u_xx).
    ic_conditions : list of tuples | None
        Each tuple is (X_ic, ic_residual_fn, weight) where:
            X_ic : tensor of IC points
            ic_residual_fn : callable (NN, X) -> residual tensor (should be ≈ 0)
            weight : optional float (defaults to 1.0)
    bc_conditions : list of tuples | None
        Same structure for boundary conditions.
            Each tuple is (X_bc, bc_residual_fn, weight) with same semantics.
    DE_params : dict | None
        Optional PDE parameters (passed to `de_eq` if needed).

    Returns
    -------
    loss_fn : Callable
        A function that can be passed directly to `train()`.
    """

    # record what argument order this loss expects
    expected_keys = ["de"]
    if ic_conditions is not None:
        expected_keys.append("ic")
    if bc_conditions is not None:
        expected_keys.append("bc")

    def loss_fn(*X_sets):
        idx = 0
        total_loss = 0.0

        # PDE residual
        X_de = X_sets[idx]
        idx += 1
        de_residual = de_eq(NN, X_de) if DE_params is None else de_eq(NN, X_de, **DE_params)
        total_loss += torch.mean(de_residual**2)

        # Initial conditions
        if ic_conditions is not None:
            for (X_ic, ic_residual_fn, *maybe_weight) in ic_conditions:
                weight = maybe_weight[0] if maybe_weight else 1.0
                u_ic_eval = ic_residual_fn(NN, X_ic)
                total_loss += weight * torch.mean(u_ic_eval**2)
            idx += 1

        # Boundary conditions
        if bc_conditions is not None:
            for (X_bc, bc_residual_fn, *maybe_weight) in bc_conditions:
                weight = maybe_weight[0] if maybe_weight else 1.0
                u_bc_eval = bc_residual_fn(NN, X_bc)
                total_loss += weight * torch.mean(u_bc_eval**2)
            idx += 1

        return total_loss

    # attach expected_keys for consistency
    loss_fn.expected_keys = expected_keys
    return loss_fn

def pde_solve(
        DE : Callable,
        X_DE : torch.Tensor,
        NN : PINN,
        IC_list : List[tuple] | None = None,
        BC_list : List[tuple] | None = None,
        DE_params : dict | None = None,
        **train_params : dict
        
):
    """
    Given a differential equation, a region, a PINN, a list of ICs and list of BCS,
    and a dictionary of any parameters for the DE, train the PINN to get a solution
    to the PDE
    Parameters
    ----------
    DE : Callable
        partial differential equation that we're trying to solve
    X_DE : torch.Tensor
        region on which the differential equation should hold
    IC_list : List[tuple] | None, optional
        Each tuple is (X_ic, ic_residual_fn, weight) where:
            X_ic : tensor of IC points
            ic_residual_fn : Callable (NN, X) -> residual tensor (should be ≈ 0)
            weight : optional float (defaults to 1.0)
    BC_list : List[tuple] | None, optional
        Each tuple is (X_bc, bc_residual_fn, weight) with the same semantics as IC_list
    DE_params : dict | None, optional
        any additional arguments expected by the DE function (e.g. {"alpha": 0.1})
    train_params : dict
        any additional parameters to be passed to train()
    """
    # first get the loss function for this PDE
    loss_fn = get_pde_loss(
        NN=NN,
        de_eq=DE,
        ic_conditions=IC_list,
        bc_conditions=BC_list,
        DE_params=DE_params
    )
    # create the list of tensors for X to be passed to train()
    X = [X_DE] # start with the set for the DE
    for ic in IC_list:
        X.append(ic[0]) # add the IC training sets
    for bc in BC_list:
        X.append(bc[0]) # add the BC training sets
    # set batch mode to random unless set specifically to shuffle by the user
    batch_mode = train_params.get("batch_mode", "random")
    # train the neural network
    train(
        NN=NN,
        loss_fn=loss_fn,
        X=X,
        use_val=False,
        batch_mode=batch_mode,
        **train_params
    )
    return NN

