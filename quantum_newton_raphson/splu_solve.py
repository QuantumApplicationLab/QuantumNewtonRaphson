import numpy as np
from numpy.typing import ArrayLike

# from qreorder.classical_ordering import find_ordering as find_reordering_classical
# from qreorder.quantum_ordering import find_ordering as find_reordering_quantum
from qreorder.core import Solver
from scipy.sparse import triu
from scipy.sparse.linalg import splu
from .base_solver import BaseSolver
from .base_solver import ValidInputFormat
from .result import SPLUResult
from .utils import preprocess_data


class MaxEdgeSolver(Solver):
    """Solver for finding the reordering by max edge."""

    def get_ordering(self, matrix: ArrayLike) -> list[int]:
        """Get ordering of the matrix using the maximum number of edges.

        Args:
            matrix (sparray): input matrix

        Returns:
            np.ndarray: ordering indices
        """
        idx = np.argsort(triu(matrix, k=1).sum(1).flatten())
        return np.array(idx).ravel()


class NoReorderSolver(Solver):
    """Solver that returns the original ordering to use as default."""

    def get_ordering(self, matrix: ArrayLike) -> list[int]:
        """Return the original ordering.

        Args:
            matrix (sparray): input matrix

        Returns:
            np.ndarray: ordering indices
        """
        size = matrix.shape[0]
        return np.array(range(size))


class SPLU_SOLVER(BaseSolver):
    """Solve the linear sysem using SPLU.

    Args:
        BaseSolver (class): base class
    """

    def __init__(self, reorder_solver: Solver = NoReorderSolver()):
        """Solver to solve the linear system using a reordering approach.

        Args:
            reorder_solver (Solver, optional): Solver to obtain the reordering indices. Defaults NoReorderSolver().
        """
        self.reorder_solver = reorder_solver

    def __call__(self, A: ValidInputFormat, b: ValidInputFormat) -> SPLUResult:
        """Solve the linear system by reordering the system of eq.

        Args:
            A (ValidInputFormat): input matrix
            b (ValidInputFormat): input rhs
            options (dict, optional): options for the reordering. Defaults to {}.

        Returns:
            SPLUResult: object containing all the results of the solver
        """
        # convert the input data into a spsparse compatible format
        A, b = preprocess_data(A, b)

        # get the reordering of the matrix
        order = self.reorder_solver.get_ordering(A)

        # reorder matrix and rhs
        A = A[np.ix_(order, order)]
        b = b[order]

        # solve
        solver = splu(A, permc_spec="NATURAL")
        x = solver.solve(b)

        # reorder solution
        x = x[np.argsort(order)]
        return SPLUResult(x, solver)
