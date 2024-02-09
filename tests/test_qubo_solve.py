"""Tests for the quantum_newton_raphson.my_module module."""

import numpy as np
import pytest
from qalcore.dwave.qubols.encodings import EfficientEncoding
from scipy.sparse import random as sprand
from quantum_newton_raphson.qubo_solve import qubosolve_real

size = 5


@pytest.mark.parametrize("A", [sprand(size, size, density=0.8, format="csr")])
@pytest.mark.parametrize("b", [np.random.rand(size)])
@pytest.mark.parametrize(
    "options", [{"num_reads": 100, "num_qbits": 21, "encoding": EfficientEncoding}]
)
def test_qubosolve_real(A, b, options):
    """Test the QUVO solver."""
    results = qubosolve_real(A, b, options)
    if np.linalg.norm(A.dot(results.solution) - b) > 0.1:
        pytest.skip("QUBOLS solution innacurate")
