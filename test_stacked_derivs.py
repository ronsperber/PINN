import torch
from pinn_utils.derivatives import compute_unique_derivatives
import pytest

X = torch.tensor([[-1.0,1.0], [-1.0,0.0], [1.0,1.0], [2.0,1.0], [-1.0,3.0]]).requires_grad_(True)
func_pairs = [
    (lambda X: X[:,0] ** 2 + X[:,1] ** 2, lambda X: X[:,0] ** 2 - X[:,1] ** 2),
    (lambda X: torch.sin(X[:,0] + X[:,1]), lambda X: torch.cos(X[:,0] + X[:,1])),
    (lambda X: torch.exp(X[:,0]) * torch.sin(X[:,1]), lambda X: torch.exp(X[:,0]) * torch.cos(X[:,1])),
    (lambda X: torch.exp(X[:,0] * X[:,1]), lambda X: torch.exp(X[:,0] + X[:,1]))
]
# Optional: human-readable IDs for pytest output
test_ids = ["quadratic_pair", "trig_pair", "exp_trig_pair", "exp_mult_pair"]
def stack(u1,u2):
    def stacked(X):
        return torch.stack([u1(X),u2(X)],axis=-1)
    return stacked

def tensor_dicts_equal(dict1, dict2, rtol=1e-5, atol=1e-8):
    # First, ensure the sets of keys match
    if dict1.keys() != dict2.keys():
        return False

    # Compare each corresponding tensor
    for k in dict1:
        v1 = dict1[k]
        v2 = dict2.get(k)
        if v2 is None:  # key missing in dict2
            return False
        if not torch.allclose(v1, v2, rtol=rtol, atol=atol):
            return False
    return True

@pytest.mark.parametrize("u1,u2", func_pairs)
def test_equal_derivs(u1, u2):
    stacked = stack(u1, u2)
    derivs = compute_unique_derivatives(stacked, X, 2)
    d1 = compute_unique_derivatives(u1, X, 2)
    d2 = compute_unique_derivatives(u2, X, 2)
    assert tensor_dicts_equal(derivs[0],d1[0])
    assert tensor_dicts_equal(derivs[1],d2[0])