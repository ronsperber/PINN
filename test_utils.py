import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pytest
from torch.testing import assert_close
from pinn_utils.pinn import PINN
from pinn_utils.training import process_training_data
from pinn_utils.ode_solve import factorials_up_to_n, taylor_polynomial


# --- PINN constructor ---

def test_pinn_forward_shape():
    model = PINN(num_hidden_layers=2, layer_width=16, num_inputs=1, num_outputs=1)
    x = torch.rand(5, 1)
    assert model(x).shape == (5, 1)

def test_pinn_multi_io_shape():
    model = PINN(num_hidden_layers=2, layer_width=16, num_inputs=2, num_outputs=3)
    x = torch.rand(4, 2)
    assert model(x).shape == (4, 3)

def test_pinn_activation_list():
    model = PINN(num_hidden_layers=2, hidden_activation=[nn.Tanh(), nn.ReLU()])
    assert model(torch.rand(3, 1)).shape == (3, 1)

def test_pinn_mismatched_activation_list_raises():
    with pytest.raises(ValueError):
        PINN(num_hidden_layers=3, hidden_activation=[nn.Tanh(), nn.ReLU()])


# --- process_training_data ---

def test_process_tensor_1d_unsqueezed():
    result = process_training_data(torch.tensor([1.0, 2.0, 3.0]))
    assert len(result) == 1
    assert result[0].shape == (3, 1)

def test_process_tensor_2d_unchanged():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = process_training_data(x)
    assert result[0].shape == (2, 2)

def test_process_numpy_converted():
    result = process_training_data(np.array([1.0, 2.0, 3.0]))
    assert isinstance(result[0], torch.Tensor)
    assert result[0].shape == (3, 1)

def test_process_dataframe_converted():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = process_training_data(df)
    assert result[0].shape == (2, 2)

def test_process_series_converted():
    result = process_training_data(pd.Series([1.0, 2.0, 3.0]))
    assert result[0].shape == (3, 1)

def test_process_list_of_tensors():
    x1 = torch.tensor([[1.0], [2.0]])
    x2 = torch.tensor([[3.0], [4.0]])
    result = process_training_data([x1, x2])
    assert len(result) == 2
    assert result[0].shape == (2, 1)
    assert result[1].shape == (2, 1)

def test_process_invalid_type_raises():
    with pytest.raises(TypeError):
        process_training_data("not a tensor")  # type: ignore[arg-type]


# --- factorials_up_to_n ---

def test_factorials_base_case():
    assert_close(factorials_up_to_n(0), torch.tensor([1.0]))

def test_factorials_correctness():
    assert_close(factorials_up_to_n(5), torch.tensor([1.0, 1.0, 2.0, 6.0, 24.0, 120.0]))

def test_factorials_negative_raises():
    with pytest.raises(ValueError):
        factorials_up_to_n(-1)


# --- taylor_polynomial ---

def test_taylor_polynomial_constant():
    # ics=[c] → g(x) = c everywhere; use batch_size=2 to avoid internal squeeze
    g = taylor_polynomial(a=0.0, ics=[3.0])
    x = torch.tensor([[0.5], [-1.0]])
    assert_close(g(x), torch.tensor([[3.0], [3.0]]))

def test_taylor_polynomial_value_at_center():
    # g(a) must equal ics[0]
    a, ics = 1.5, [2.0, -1.0, 4.0]
    g = taylor_polynomial(a=a, ics=ics)
    x = torch.tensor([[a], [a]])
    assert_close(g(x), torch.tensor([[2.0], [2.0]]))

def test_taylor_polynomial_linear():
    # g(x) = 1 + 3*(x - 2): g(3) = 4, g(2) = 1
    g = taylor_polynomial(a=2.0, ics=[1.0, 3.0])
    x = torch.tensor([[3.0], [2.0]])
    assert_close(g(x), torch.tensor([[4.0], [1.0]]))
