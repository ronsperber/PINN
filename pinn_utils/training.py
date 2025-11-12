"""
module to train a PyTorch neural network
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np, pandas as pd
import copy
from typing import List, Callable, TypeAlias

TensorLike: TypeAlias = torch.Tensor | np.ndarray | pd.DataFrame | pd.Series
TensorListLike: TypeAlias = TensorLike | list[TensorLike] | tuple[TensorLike, ...]

def process_training_data(X: TensorListLike) -> List[torch.Tensor]:
    """
    function to turn the arguments to train() to be a list of
    torch.Tensors
    Parameters
    ----------
    X : TensorListLike
        inputs to be processed
    Returns
    -------
    List of tensors
    """
    # if a single type was passed, convert to a list 
    if not isinstance(X, (list, tuple)):
        X = [X]
    # convert all elements of X to torch tensors
    processed_X = []
    for x in X:
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float32)
        elif isinstance(x, pd.Series):
            x = torch.tensor(x.values, dtype=torch.float32)
        elif isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif not torch.is_tensor(x):
            raise TypeError(f"Unsupported input type: {type(x)}")
        x = x.unsqueeze(-1) if x.ndim == 1 else x
        processed_X.append(x)
    return processed_X

def train(
          NN: nn.Module,
          X: TensorListLike,
          loss_fn : Callable,
          epochs: int = 5000,
          lr: float = 1e-3,
          batch_size: int | None= None,
          batch_mode: str = "shuffle",
          use_grad : bool = False,
          use_val: bool = True,
          val_size: float | int = 0.2,
          early_stopping: int | None = None,
          return_checkpoints: bool = False,
          checkpoint_every: int = 20,
          progress_callback: Callable[[int, float, float], None] |None = None,
          progress_every: int = 1
          ):
    """
    trains a neural network with training data and loss function provided
    Parameters
    ----------
    NN : nn.Module
        a neural network of type nn.Module.  
    X : TensorListLike
        the training data
    loss_fn : Callable
        loss function to use during training.
    epochs : int
        number of epochs to train
    lr : float
        learning rate
    batch_size : int | None
        when not None, the size of batches to use during training
        otherwise training is done on the whole data set at once each epoch
    batch_mode : str
        When the mode is "shuffle" the datasets are shuffled and broken into batch_size size subsets
        When the mode is "random" each epoch we randomly sample a bunch of batch_size each 'batch'
    use_grad : bool
        whether or not the data sets should require grad (typically used for PINN)
    use_val : bool
        whether or not to split into train/validation sets
    val_size: int | float
        what size to make the validation set. If this is an int,
        it is the absolute size of the validation set. If this is in (0,1),
        it is a relative size compared to the training set size
    early_stopping : dict | None
        When not None, keys for whether to stop early. Keys:
        min_epochs - minimum number of epochs before considering early stopping
        patience - how many epochs without improvement on val_loss needed before stopping
    return_checkpoints: bool
        whether or not to return intermediate solutions.
    checkpoint_every : int
        when return_checkpoints is true, how often to return a checkpoint of values
    progress_callback : Callable | None
        when not None, is a function used to send progress to whatever is calling solve
    progress_every : int
        when there is a progress_callback, how often to call it
    
    Returns
    -------
    checkpoints : List[Tuple(int,callable)] | None:
        when return_checkpoints is true a list of pairs (epoch, solution at epoch)
        otherwise returns None
    """
    
    # replace X with list of things that are all tensors
    X = process_training_data(X)
    # make sure all data sets have the same number of training points
    if use_val:
        batch_sizes = [x.shape[0] for x in X]
        if len(set(batch_sizes)) != 1:
            raise ValueError(f"All elements in X must have the same batch size, got {batch_sizes}")
        # get number of points being used
        data_size = X[0].shape[0]
        # make sure val_size is a valid input
        if isinstance(val_size, int):
            if val_size <=0 or val_size >= data_size:
                raise ValueError("Not a valid validation size")
        elif isinstance(val_size, float):
            if val_size <=0 or val_size >= 1:
                raise ValueError("Not a valid validation size")
            else:
                # when it's a float get the actual size of the validation set
                val_size = int(data_size * val_size)
        else:
            raise TypeError("Validation size must be int or float")
        # get rid of requires_grad on x if true so it can be split
        for i,x in enumerate(X):
            if x.requires_grad:
                X[i] = x.detach().clone().requires_grad_(False)
        # generate train/validation split based on val_size
        perm = torch.randperm(data_size)
        X_val = [x[perm[:val_size]].requires_grad_(use_grad) for x in X]
        X_train = [x[perm[val_size:]].requires_grad_(use_grad) for x in X]
    else:
        X_train =[x.requires_grad_(True) for x in X]
        X_val = None
    if return_checkpoints:
        checkpoints = []
    else:
        checkpoints = None
    train_size = X_train[0].shape[0]
    # when there's early stopping get min_epochs, patience
    # and set up best loss and epochs since best
    if early_stopping is not None:
        min_epochs = early_stopping.get('min_epochs', 20)
        patience = early_stopping.get('patience', 10)
        best_val_loss = float('inf')
        epochs_since_best = 0
        best_weights = None
    # initialize optimizer
    optimizer = torch.optim.Adam(params=NN.parameters(), lr=lr)
    # training loop
    with tqdm(total=epochs, desc="Training", unit="epoch") as pbar:
        for epoch in range(1, epochs + 1):
            # make sure NN is in training state
            NN.train()
            if batch_size is None:
                # when no batching exists reset optimizer
                # compute loss and do backpropogation
                optimizer.zero_grad()
                loss = loss_fn(*X_train)
                loss.backward()
                epoch_loss = loss.item()
                optimizer.step()
            elif batch_mode == "shuffle":
                # shuffle every epoch to get new batches
                perm = torch.randperm(train_size)
                epoch_loss = 0.0
                # loop over the batches and compute average loss
                # each batch run backpropogation
                for i in range(0, train_size, batch_size):
                    optimizer.zero_grad()
                    end = min(i+batch_size, train_size)
                    idx = perm[i:end]
                    X_batch = [x_train[idx] for x_train in X_train]
                    loss = loss_fn(*X_batch)
                    loss.backward()
                    epoch_loss += loss.item() * len(idx)
                    optimizer.step()
                epoch_loss /= train_size
            elif batch_mode == "random":
                # set epoch loss to zero
                epoch_loss = 0.0
                # get a number of batches based on largest training set and batch size
                num_batches = max([ x.shape[0] for x in X_train]) // batch_size
                for _ in range(num_batches):
                    # get a random sample of batch_size from each X in X_train
                    X_batch = [x[torch.randint(0, x.shape[0], (batch_size,))] for x in X_train]
                    optimizer.zero_grad()
                    loss = loss_fn(*X_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                epoch_loss /= num_batches
            else:
                raise ValueError(f"{batch_mode} is not a valid mode. Mode must be 'random' or 'shuffle'")
            # compute loss on validation set
            # set NN to eval for this
            NN.eval()
            if use_val:
                val_loss = loss_fn(*X_val).item()
            else:
                val_loss = float('nan')
            # Call progress callback if provided (epoch, train_loss, val_loss)
            if progress_callback is not None:
                try:
                    # Only invoke callback every `progress_every` epochs to reduce UI churn,
                    # but always call on the final epoch so the UI can show completion.
                    if progress_every <= 1 or (epoch % progress_every == 0) or (epoch == epochs):
                        progress_callback(epoch, float(epoch_loss), float(val_loss))
                except Exception:
                    # Ensure progress callback failures don't stop training
                    pass
            # check for early stopping conditions when they exist
            if use_val:
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
            # put back into training
            NN.train()
            pbar.set_postfix(loss=epoch_loss, val_loss=val_loss)
            pbar.update(1)
            # When we are returning checkpoints, create the checkpoint function and add to list
            if return_checkpoints and epoch % checkpoint_every == 0:
                checkpoint_state = {k: v.clone() for k, v in NN.state_dict().items()}
                # create a function that uses a fresh NN with these weights
                nn_copy = copy.deepcopy(NN)       # or create a new PINN instance
                nn_copy.load_state_dict(checkpoint_state)
                nn_copy.eval()
                checkpoints.append((epoch, nn_copy))
        # put into eval mode after training
    NN.eval()
    return checkpoints
