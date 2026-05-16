
import numpy as np
import pytest
from pinn_utils.de_sols import (
    exp_sol,
    logistic_sol,
    bernoulli_sol,
    linear_nonhomogeneous_sol,
    cauchy_euler_sol,
    linear_homogeneous_sol,
    linear_2nd_nonhomogeneous_sol,
    nonlinear_2nd_example
)
def numerical_derivative(f, x, order=1, h=1e-5):
    if order == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    elif order == 2:
        return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)
    else:
        raise ValueError("Only first and second order supported")

# Each entry: (solver_function, residual_fn, param_dicts, order)
DE_TESTS = [
    # First-order DEs
    (
        exp_sol,
        lambda y, x, k: numerical_derivative(y, x) - k*y(x),
        [{"k": 1.0, "x0": 0.0, "y0": 1.0}, {"k": -0.5, "x0": 0.5, "y0": 2.0}],
        1
    ),
    (
        logistic_sol,
        lambda y, x, k: numerical_derivative(y, x) - k*y(x)*(1 - y(x)),
        [{"k": 1.0, "x0": 0.0, "y0": 0.1}],
        1
    ),
    (
        bernoulli_sol,
        lambda y, x, k: numerical_derivative(y, x) - k*y(x)**2,
        [{"k": 1.0, "x0": 0.0, "y0": 1.0},
         {"k": -1.0, "x0": 0.0, "y0": -1.0},
         {"k": 1.0, "x0": 0.0, "y0": 0.0}],
        1
    ),
    (
        linear_nonhomogeneous_sol,
        lambda y, x, k: numerical_derivative(y, x) - (k*y(x) + np.sin(x)),
        [{"k": 1.0, "x0": 0.0, "y0": 1.0}],
        1
    ),
    # Second-order DEs
    (
        cauchy_euler_sol,
        lambda y, x, b, c: x**2 * numerical_derivative(y, x, order=2) + b*x*numerical_derivative(y, x) + c*y(x),
        [{"b": 3, "c": 2, "x0": 1.5, "y0": 1.0, "yprime0": 0.0},
         {"b": 2, "c": 1, "x0": 1.5, "y0": 1.0, "yprime0": 0.0},
         {"b": 1, "c": 2, "x0": 1.5, "y0": 1.0, "yprime0": 0.0}],
        2
    ),
    (
        linear_homogeneous_sol,
        lambda y, x, b, c: numerical_derivative(y, x, order=2) + b*numerical_derivative(y, x) + c*y(x),
        [{"b": 3, "c": 2, "x0": 0.0, "y0": 1.0, "yprime0": 0.0},
         {"b": 2, "c": 1, "x0": 0.0, "y0": 1.0, "yprime0": 0.0},
         {"b": 0, "c": 4, "x0": 0.0, "y0": 1.0, "yprime0": 0.0}],
        2
    ),
    (
        linear_2nd_nonhomogeneous_sol,
        lambda y, x: numerical_derivative(y, x, order=2) - y(x) - np.exp(x),
        [{"x0": 0.0, "y0": 1.0, "yprime0": 1.0},
         {"x0": 1.0, "y0": 1.0, "yprime0": -1.0}],
         2
    ),
    (
        nonlinear_2nd_example,
        lambda y, x, k: numerical_derivative(y, x, order=2) - k * numerical_derivative(y, x)**2,
        [{"k": 1.0, "x0": 0.0, "y0": 1.0, "yprime0": 1.0},
         {"k": -0.5, "x0": 0.5, "y0": 2.0, "yprime0": -1.0},
         {"k": 2.0, "x0": 1.0, "y0": 0.0, "yprime0": 0.5}],
         2
    )
]

_parametrize_cases = [
    (solver, residual, params, order)
    for solver, residual, param_list, order in DE_TESTS
    for params in param_list
]

@pytest.mark.parametrize("solver,residual,param_dict,order", _parametrize_cases)
def test_all_DEs(solver, residual, param_dict, order, atol=5e-5):
    y_true = solver(**param_dict)
    x0 = param_dict.get("x0", 0.0)
    xs = np.linspace(x0 - 0.5, x0 + 0.5, 50)
    res = residual(y_true, xs, **{k: v for k,v in param_dict.items() if k not in ("x0","y0","yprime0")})
    assert np.max(np.abs(res)) < atol


@pytest.mark.parametrize("solver,param_dict,order", [
    (solver, params, order)
    for solver, residual, params, order in _parametrize_cases
])
def test_initial_conditions(solver, param_dict, order):
    y_true = solver(**param_dict)
    x0 = param_dict["x0"]
    assert np.abs(y_true(x0) - param_dict["y0"]) < 1e-10
    if order == 2:
        dy_x0 = numerical_derivative(y_true, x0)
        assert np.abs(dy_x0 - param_dict["yprime0"]) < 1e-5
