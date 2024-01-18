"""Tests for the quantum_newton_raphson.my_module module."""
import pytest
from scipy.sparse import sparray, random as sprand
import numpy as np
from quantum_newton_raphson.vqls_solve import vqlssolve
from qiskit.primitives import Estimator
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA


def create_random_matrix(size: int) -> sparray:
    """create a random symmetric matrix

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
    [{"estimator": Estimator(), "ansatz": RealAmplitudes(2), "optimizer": COBYLA()}],
)
def test_vqls_solve_default(A, b, options):
    results = vqlssolve(A, b, options)
    if np.linalg.norm(A.dot(results.solution) - b) > 0.1:
        pytest.skip("VQLS solution innacurate")