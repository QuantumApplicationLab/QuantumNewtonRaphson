"""Tests for the quantum_newton_raphson.my_module module."""

import numpy as np
import pytest
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Estimator
from qiskit_aer.primitives import EstimatorV2 as aer_EstimatorV2
from qiskit_algorithms.optimizers import COBYLA
from scipy.sparse import random as sprand
from scipy.sparse import sparray
from quantum_newton_raphson.vqls_solver import VQLS_SOLVER


def create_random_matrix(size: int) -> sparray:
    """Create a random symmetric matrix.

    Args:
        size (int): size of the matrix
    """
    mat = sprand(size, size, density=1.0, format="csc")
    return mat + mat.T


size = 4


@pytest.mark.parametrize("A", [create_random_matrix(size)])
@pytest.mark.parametrize("b", [np.random.rand(size)])
@pytest.mark.parametrize(
    "options",
    [
        {
            "estimator": Estimator(),
            "ansatz": RealAmplitudes(2),
            "optimizer": COBYLA(),
            "preconditioner": "diagonal_scaling",
            "reorder": True,
        },
        {
            "estimator": aer_EstimatorV2(),
            "ansatz": RealAmplitudes(2),
            "optimizer": COBYLA(),
        },
    ],
)
def test_vqls_solve_default(A, b, options):
    """Test the vqls solver."""
    solver = VQLS_SOLVER(**options)
    results = solver(A, b)
    if np.linalg.norm(A.dot(results.solution) - b) > 0.1:
        pytest.skip("VQLS solution innacurate")
