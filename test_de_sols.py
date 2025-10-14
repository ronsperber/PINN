import numpy as np
import pytest
from pinn_utils.de_sols import (
    exp_sol,
    logistic_sol,
    cauchy_euler_sol,
    linear_homogeneous_sol,
)

def numerical_derivative(f, x, order=1, h=1e-6):
    if order == 1:
        return (f(x + h) - f(x - h)) / (2*h)
    elif order == 2:
        return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)
    else:
        raise ValueError("Only order 1 and 2 supported")

def report_max_residual(lhs, desc=""):
    max_res = np.max(np.abs(lhs))
    print(f"{desc} max residual: {max_res:.5g}")
    return max_res

# --- 1st-order ODEs ---
@pytest.mark.parametrize("k", [1.0, -2.0, 0.5])
def test_exp_sol(k):
    x0, y0 = 0.0, 1.0
    y_true = exp_sol(k, x0, y0)
    xs = np.linspace(-1, 1, 50)
    y1 = numerical_derivative(y_true, xs)
    lhs = y1 - k*y_true(xs)
    max_res = report_max_residual(lhs, f"exp_sol k={k}")
    assert max_res < 1e-5

@pytest.mark.parametrize("k", [1.0, 2.0])
def test_logistic_sol(k):
    x0, y0 = 0.0, 0.5
    y_true = logistic_sol(k, x0, y0)
    xs = np.linspace(-2, 2, 50)
    y1 = numerical_derivative(y_true, xs)
    lhs = y1 - k*y_true(xs)*(1 - y_true(xs))
    max_res = report_max_residual(lhs, f"logistic_sol k={k}")
    assert max_res < 5e-4  # slightly relaxed due to finite differences

# --- 2nd-order ODEs ---
@pytest.mark.parametrize("b,c", [
    (3, 2),      # distinct real roots
    (2, 1),      # repeated root
    (1, 2),      # complex roots
])
def test_cauchy_euler_cases(b, c):
    x0, y0, yprime0 = 1.5, 1.0, 0.0
    y_true = cauchy_euler_sol(x0, y0, yprime0, b, c)
    xs = np.linspace(1.0, 3.0, 50)
    y1 = numerical_derivative(y_true, xs)
    y2 = numerical_derivative(y_true, xs, order=2)
    lhs = xs**2 * y2 + b * xs * y1 + c * y_true(xs)
    max_res = report_max_residual(lhs, f"Cauchy-Euler b={b}, c={c}")
    assert max_res < 5e-3  # relaxed due to numeric derivative noise

@pytest.mark.parametrize("b,c", [
    (3, 2),      # distinct real roots
    (2, 1),      # repeated root
    (0, 4),      # complex roots
])
def test_linear_homogeneous_cases(b, c):
    x0, y0, yprime0 = 0.0, 1.0, 0.0
    y_true = linear_homogeneous_sol(b, c, x0, y0, yprime0)
    xs = np.linspace(-1, 1, 50)
    y1 = numerical_derivative(y_true, xs)
    y2 = numerical_derivative(y_true, xs, order=2)
    lhs = y2 + b * y1 + c * y_true(xs)
    max_res = report_max_residual(lhs, f"Linear Homogeneous b={b}, c={c}")
    assert max_res < 5e-3
