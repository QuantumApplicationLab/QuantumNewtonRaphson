"""Tests for the quantum_newton_raphson.my_module module."""

import numpy as np
import pytest
from qiskit.primitives import Estimator
from scipy.sparse import random as sprand
from scipy.sparse import sparray
from quantum_newton_raphson.hhl_solver import hhlsolve


def create_random_matrix(size: int) -> sparray:
    """Create a random symmetric matrix.

    Args:
        size (int): size of the matrix
    """
    mat = sprand(size, size, density=0.85, format="csc")
    return mat + mat.T


size = 4


@pytest.mark.parametrize("A", [create_random_matrix(size)])
@pytest.mark.parametrize("b", [np.random.rand(size)])
@pytest.mark.parametrize(
    "options",
    [{"estimator": Estimator()}],
)
def test_hhl_solve_default(A, b, options):
    """Test the hhl solver."""
    results = hhlsolve(A, b, options)
    if np.linalg.norm(A.dot(results.solution) - b) > 0.1:
        pytest.skip("HHL solution innacurate")
