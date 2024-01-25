"""Tests for the quantum_newton_raphson.my_module module."""
import numpy as np
from quantum_newton_raphson.newton_raphson import newton_raphson


def func(input: np.ndarray) -> np.ndarray:
    """Test function of a 3x3 system.

    Args:
        input (np.ndarray): input point
    """

    def f1(x, y, z):
        return 2 * x**3 - x * y + 4 * z - 12

    def f2(x, y, z):
        return -4 * x + z**5 + 8 * z * y

    def f3(x, y, z):
        return 3 * x**3 + z**5 + 8 * z * y

    x, y, z = input
    return np.array([f(x, y, z) for f in [f1, f2, f3]])


def grad(input: np.ndarray) -> np.ndarray:
    """Analytic gradients of the function.

    Args:
        input (np.ndarray): input point

    Returns:
        np.ndarray: jacobian matrix
    """

    def df1(x, y, z):
        return np.array([6 * x**2 - y, -x, 4])

    def df2(x, y, z):
        return np.array([-4, 8 * z, 5 * z**4 + 8 * y])

    def df3(x, y, z):
        return np.array([9 * x**2, 8 * z, 5 * z**4 + 8 * y])

    out = np.zeros((3, 3))
    x, y, z = input
    out[0, :] = df1(x, y, z)
    out[1, :] = df2(x, y, z)
    out[2, :] = df3(x, y, z)

    return out


def test_newton_raphson_analytic_gradients():
    """Compute NR solution using analytic gradients."""
    initial_point = np.random.rand(3)
    res = newton_raphson(func, initial_point, grad=grad)
    assert np.allclose(func(res.solution), 0)


def test_newton_raphson_fd_gradients():
    """Compute NR solution using fd gradients."""
    initial_point = np.random.rand(3)
    res = newton_raphson(func, initial_point)
    assert np.allclose(func(res.solution), 0)
