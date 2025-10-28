import torch
import math
from pinn_utils.pinn import compute_unique_derivatives
from torch.testing import assert_close

def test_simple_polynomial():
    # f(x) = x0^2 + 3x1
    def f(x):
        return (x[:, 0]**2 + 3 * x[:, 1]).unsqueeze(-1)

    x = torch.tensor([[2.0, 1.0]], requires_grad=True)
    derivs = compute_unique_derivatives(f, x, order=2)[0]

    # First derivatives
    assert_close(derivs["x0"], torch.tensor([4.0]))  # df/dx0 = 2x0 = 4
    assert_close(derivs["x1"], torch.tensor([3.0]))  # df/dx1 = 3

    # Second derivatives
    assert_close(derivs["x0_x0"], torch.tensor([2.0]))  # d²f/dx0² = 2
    assert_close(derivs["x1_x1"], torch.tensor([0.0]))  # d²f/dx1² = 0
    assert_close(derivs["x0_x1"], torch.tensor([0.0]))  # mixed = 0

def test_trig_exponential():
    # f(x) = exp(x0 * sin(x1))
    def f(x):
        return torch.exp(x[:,0] * torch.sin(x[:,1])).unsqueeze(-1)
    x = torch.tensor([[1, math.pi /6]])
    derivs = compute_unique_derivatives(f, x, order=2)
    # analytics checks
    y_expected = math.exp(math.sin(math.pi / 6))
    dx0 = y_expected * math.sin(math.pi / 6)
    dx1 = y_expected * math.cos(math.pi / 6)
    dx0_x0 = y_expected * math.sin(math.pi / 6) ** 2
    dx1_x1 = y_expected * math.cos(math.pi / 6) ** 2
    dx0_x1 = y_expected * math.cos(math.pi / 6) * math.sin(math.pi / 6)
    assert_close(derivs["y"], torch.tensor([y_expected]))
    assert_close(derivs["x0"], torch.tensor([dx0]))
    assert_close(derivs["x1"], torch.tensor([dx1]))
    assert_close(derivs["x0_x0"], torch.tensor([dx0_dx0]), rtol=1e-6, atol=1e-6)
    assert_close(derivs["x1_x1"], torch.tensor([dx1_dx1]), rtol=1e-6, atol=1e-6)
    assert_close(derivs["x0_x1"], torch.tensor([dx0_dx1]), rtol=1e-6, atol=1e-6)

def test_trig_function():
    # f(x) = sin(x0) * cos(x1)
    def f(x):
        return (torch.sin(x[:, 0]) * torch.cos(x[:, 1])).unsqueeze(-1)

    x = torch.tensor([[math.pi / 4, math.pi / 4]], requires_grad=True)
    derivs = compute_unique_derivatives(f, x, order=2)[0]

    # Analytic checks
    y_expected = 0.5 
    dx0 = math.cos(math.pi/4) * math.cos(math.pi/4)
    dx1 = -math.sin(math.pi/4) * math.sin(math.pi/4)
    dx0_dx0 = -math.sin(math.pi/4) * math.cos(math.pi/4)
    dx1_dx1 = -math.sin(math.pi/4) * math.cos(math.pi/4)
    dx0_dx1 = -math.cos(math.pi/4) * math.sin(math.pi/4)

    assert_close(derivs["y"], torch.tensor([y_expected]))
    assert_close(derivs["x0"], torch.tensor([dx0]))
    assert_close(derivs["x1"], torch.tensor([dx1]))
    assert_close(derivs["x0_x0"], torch.tensor([dx0_dx0]), rtol=1e-6, atol=1e-6)
    assert_close(derivs["x1_x1"], torch.tensor([dx1_dx1]), rtol=1e-6, atol=1e-6)
    assert_close(derivs["x0_x1"], torch.tensor([dx0_dx1]), rtol=1e-6, atol=1e-6)


def test_multiple_outputs():
    # f(x) = [x0^2, x1^3]
    def f(x):
        return torch.stack([x[:, 0]**2, x[:, 1]**3], dim=1)

    x = torch.tensor([[2.0, 3.0]], requires_grad=True)
    derivs = compute_unique_derivatives(f, x, order=2)

    # Output 0: x0^2
    d0 = derivs[0]
    assert_close(d0["x0"], torch.tensor([4.0]))
    assert_close(d0["x1"], torch.tensor([0.0]))
    assert_close(d0["x0_x0"], torch.tensor([2.0]))

    # Output 1: x1^3
    d1 = derivs[1]
    assert_close(d1["x1"], torch.tensor([27.0]))
    assert_close(d1["x1_x1"], torch.tensor([18.0]))
    assert_close(d1["x0"], torch.tensor([0.0]))


if __name__ == "__main__":
    test_simple_polynomial()
    test_trig_function()
    test_multiple_outputs()
    print("✅ All derivative tests passed.")
