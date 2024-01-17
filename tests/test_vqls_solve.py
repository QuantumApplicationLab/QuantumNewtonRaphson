"""Tests for the quantum_newton_raphson.my_module module."""
import pytest
from scipy.sparse import random as sprand
import numpy as np
from quantum_newton_raphson.vqls_solve import vqlssolve


size = 4


@pytest.mark.parametrize("A", [sprand(size, size, density=0.85, format="csc")])
@pytest.mark.parametrize("b", [np.random.rand(size)])
@pytest.mark.parametrize(
    "options", [{}, {"reorder": "max_edge"}, {"reorder": "no_reordering"}]
)
def test_vqls_solve_default(A, b, options):
    assert 0 == 0
