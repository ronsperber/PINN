import numpy as np
import torch
from pinn_utils.de_sols import (
    bernoulli_sol,
    cauchy_euler_sol,
    exp_sol,
    logistic_sol,
    linear_nonhomogeneous_sol,
    linear_homogeneous_sol,
    linear_2nd_nonhomogeneous_sol,
    nonlinear_2nd_example,
)

__all__ = ["ODES"]

# ODE metadata mapping for the sidebar and solver logic
ODES = {
    "y' = k y": {
        "order": 1,
        "needs_k": True,
        "needs_b_c": False,
        "x0_positive": False,
        "F_factory": lambda k=None, **kw: (lambda x, y, dy: dy - k * y),
        "true_factory": lambda x0, y0, k=None, **kw: exp_sol(k=k, x0=x0, y0=y0),
        "ode_str": lambda k=None, **kw: f"dy/dx = {k} y",
    },
    "y' = k y (1 - y)": {
        "order": 1,
        "needs_k": True,
        "needs_b_c": False,
        "x0_positive": False,
        "F_factory": lambda k=None, **kw: (lambda x, y, dy: dy - k * y * (1 - y)),
        "true_factory": lambda x0, y0, k=None, **kw: (lambda x: y0 * np.ones_like(x)) if y0 in (0, 1) else logistic_sol(k=k, x0=x0, y0=y0),
        "ode_str": lambda k=None, **kw: f"dy/dx = {k} y (1 - y)",
    },
    "y' = k y + sin(x)": {
        "order": 1,
        "needs_k": True,
        "needs_b_c": False,
        "x0_positive": False,
        "F_factory": lambda k=None, **kw: (lambda x, y, dy: dy - k * y - torch.sin(x)),
        "true_factory": lambda x0, y0, k=None, **kw: linear_nonhomogeneous_sol(k=k, x0=x0, y0=y0),
        "ode_str": lambda k=None, **kw: f"dy/dx = {k} y + sin(x)",
    },
    "y' = k y²": {
        "order": 1,
        "needs_k": True,
        "needs_b_c": False,
        "x0_positive": False,
        "F_factory": lambda k=None, **kw: (lambda x, y, dy: dy - k * y ** 2),
        "true_factory": lambda x0, y0, k=None, **kw: bernoulli_sol(k=k, x0=x0, y0=y0),
        "ode_str": lambda k=None, **kw: f"dy/dx = {k} y²",
    },
    "y'' + by' + cy = 0": {
        "order": 2,
        "needs_k": False,
        "needs_b_c": True,
        "x0_positive": False,
        "F_factory": lambda b=None, c=None, **kw: (lambda x, y, dy, ddy: ddy + b * dy + c * y),
        "true_factory": lambda x0, y0, yprime0=None, b=None, c=None, **kw: linear_homogeneous_sol(x0=x0, y0=y0, yprime0=yprime0, b=b, c=c),
        "ode_str": lambda b=None, c=None, **kw: f"d²y/dx² + {b} dy/dx + {c} y = 0",
    },
    "x² y'' + b x y' + c y = 0": {
        "order": 2,
        "needs_k": False,
        "needs_b_c": True,
        "x0_positive": True,
        "F_factory": lambda b=None, c=None, **kw: (lambda x, y, dy, ddy: x ** 2 * ddy + b * x * dy + c * y),
        "true_factory": lambda x0, y0, yprime0=None, b=None, c=None, **kw: cauchy_euler_sol(x0=x0, y0=y0, yprime0=yprime0, b=b, c=c),
        "ode_str": lambda b=None, c=None, **kw: f"x² d²y/dx² + {b} x dy/dx + {c} y = 0",
    },
    "y'' - y = eˣ": {
        "order": 2,
        "needs_k": False,
        "needs_b_c": False,
        "x0_positive": False,
        "F_factory": lambda **kw: (lambda x, y, dy, ddy: ddy - y - torch.exp(x)),
        "true_factory": lambda x0, y0, yprime0=None, **kw: linear_2nd_nonhomogeneous_sol(x0=x0, y0=y0, yprime0=yprime0),
        "ode_str": lambda **kw: "d²y/dx² - y = eˣ",
    },
    "y'' = k (y')²": {
        "order": 2,
        "needs_k": True,
        "needs_b_c": False,
        "x0_positive": False,
        "F_factory": lambda k=None, **kw: (lambda x, y, dy, ddy: ddy - k * dy ** 2),
        "true_factory": lambda x0, y0, yprime0=None, k=None, **kw: nonlinear_2nd_example(k=k, x0=x0, y0=y0, yprime0=yprime0),
        "ode_str": lambda k=None, **kw: f"d²y/dx² = {k} (dy/dx)²",
    },
}
