import numpy as np
from typing import Callable
"""
Analytic solutions for selected differential equations.
These are used to benchmark or validate PINN-based solvers.
"""

def exp_sol(k: float, x0: float, y0: float) -> Callable:
    """
    returns the analytic solution for the ODE
    dy/dx = k*y
    y(x0) = y0
    """
    def y_analytic(x: np.ndarray) -> np.ndarray:
        return y0 * np.exp(k * (x - x0))
    return y_analytic

def cauchy_euler_sol(x0: float, y0: float, yprime0: float, b: float, c: float) -> Callable:
    """
    returns the analytic solution for the ODE
    x^2 y'' + b x y' + c y = 0
    y(x0) = y0
    y'(x0) = yprime0
    """
    # Characteristic equation: r^2 + (b-1) r + c = 0
    coeffs = [1, b-1, c]
    r1, r2 = np.roots(coeffs)

    if np.iscomplex(r1):  # complex roots
        alpha = r1.real
        beta = abs(r1.imag)
        cosb = np.cos(beta * np.log(x0))
        sinb = np.sin(beta * np.log(x0))
        A = np.array([
            [cosb, sinb],
            [(alpha * cosb - beta * sinb), (alpha * sinb + beta * cosb)]
        ])
        Y = np.array([y0 / x0**alpha, yprime0 / x0**(alpha-1)])
        C1, C2 = np.linalg.solve(A, Y)
        return lambda x: x**alpha * (C1 * np.cos(beta * np.log(x)) + C2 * np.sin(beta * np.log(x)))

    elif r1 == r2:  # repeated root
        r = r1
        A = np.array([
            [x0**r, x0**r * np.log(x0)],
            [r * x0**(r-1), r * x0**(r-1) * np.log(x0) + x0**(r-1)]
        ])
        Y = np.array([y0, yprime0])
        C1, C2 = np.linalg.solve(A, Y)
        return lambda x: C1 * x**r + C2 * x**r * np.log(x)

    else:  # distinct real roots
        C1 = (yprime0 - r2 * y0 / x0) / (r1 - r2)
        C2 = y0 - C1
        return lambda x: C1 * x**r1 + C2 * x**r2

def logistic_sol(k: float, x0: float, y0: float) -> Callable:
    """
    returns the analytic solution for the ODE
    dy/dx = k*y*(1-y)
    y(x0) = y0
    """
    def y_analytic(x: np.ndarray) -> np.ndarray:
            return 1 / (1 + ((1 - y0) / y0) * np.exp(-k * (x - x0)))
    return y_analytic

def linear_homogeneous_sol(b: float, c: float, x0: float, y0: float, yprime0: float) -> Callable:
    """
    returns the analytic solution for the ODE
    y'' + b y' + c y = 0
    y(x0) = y0
    y'(x0) = yprime0
    """
    discriminant = b**2 - 4*c
    if discriminant > 0:  # two distinct real roots
        r1 = (-b + np.sqrt(discriminant)) / 2
        r2 = (-b - np.sqrt(discriminant)) / 2
        A = np.array([[1, 1], [r1, r2]])
        Y = np.array([y0, yprime0])
        C1, C2 = np.linalg.solve(A, Y)
        return lambda x: C1 * np.exp(r1 * (x - x0)) + C2 * np.exp(r2 * (x - x0))
    elif discriminant == 0:  # one repeated real root
        r = -b / 2
        A = np.array([[1, 0], [r, 1]])
        Y = np.array([y0, yprime0])
        C1, C2 = np.linalg.solve(A, Y)
        return lambda x: (C1 + C2 * (x - x0)) * np.exp(r * (x - x0))
    else:  # complex roots
        alpha = -b / 2
        beta = np.sqrt(-discriminant) / 2
        A = np.array([[1, 0], [alpha, beta]])
        Y = np.array([y0, yprime0])
        C1, C2 = np.linalg.solve(A, Y)
        return lambda x: np.exp(alpha * (x - x0)) * (C1 * np.cos(beta * (x - x0)) + C2 * np.sin(beta * (x - x0)))