"""Tests for the quantum_newton_raphson.my_module module."""

import numpy as np
import pytest
from scipy.sparse import random as sprand
from quantum_newton_raphson.splu_solve import SPLU_SOLVER
from quantum_newton_raphson.splu_solve import MaxEdgeSolver, NoReorderSolver
from qreorder.classical import MinimumChordalCompletion
from qreorder.quantum import QuantumSolver

size = 5


@pytest.mark.parametrize("A", [sprand(size, size, density=0.85, format="csc")])
@pytest.mark.parametrize("b", [np.random.rand(size)])
@pytest.mark.parametrize(
    "options",
    [{}, {"reorder_solver": MaxEdgeSolver()}, {"reorder_solver": NoReorderSolver()}],
)
def test_splu_solve_default(A, b, options):
    """Test the sparse LU solver."""
    solver = SPLU_SOLVER(**options)
    results = solver(A, b)
    assert np.allclose(A.dot(results.solution), b)


@pytest.mark.parametrize("A", [sprand(size, size, density=0.85, format="csr")])
@pytest.mark.parametrize("b", [np.random.rand(size)])
@pytest.mark.parametrize(
    "options",
    [
        {"reorder_solver": MinimumChordalCompletion()},
        {"reorder_solver": QuantumSolver()},
    ],
)
@pytest.mark.xfail
def test_splu_solve_quantum(A, b, options):
    """Test the sparse LU solver using quantum reordering."""
    solver = SPLU_SOLVER(**options)
    results = solver(A, b)
    assert np.allclose(A.dot(results.solution), b)
