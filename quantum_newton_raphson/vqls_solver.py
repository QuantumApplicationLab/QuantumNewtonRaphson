import numpy as np
from qiskit.primitives import Estimator
from scipy.sparse import sparray
from vqls_prototype import VQLS
from .base_solver import BaseSolver
from .result import VQLSResult
from .utils import preprocess_data


class VQLS_SOLVER(BaseSolver):
    """Solver using VQLS.

    Args:
        BaseSolver (object): base solver class
    """

    def __init__(self, **quantum_solver_options):
        """Init the solver and process options."""
        # extract required options for the vqls solver
        self.estimator = quantum_solver_options.pop("estimator")
        self.ansatz = quantum_solver_options.pop("ansatz")
        self.optimizer = quantum_solver_options.pop("optimizer")

        # extract optional options for the vqls solver
        self.sampler = (
            quantum_solver_options.pop("sampler")
            if "sampler" in quantum_solver_options
            else None
        )
        self.initial_point = (
            quantum_solver_options.pop("initial_point")
            if "initial_point" in quantum_solver_options
            else None
        )
        self.gradient = (
            quantum_solver_options.pop("gradient")
            if "gradient" in quantum_solver_options
            else None
        )
        self.max_evals_grouped = (
            quantum_solver_options.pop("max_evals_grouped")
            if "max_evals_grouped" in quantum_solver_options
            else 1
        )

        self.quantum_solver_options = quantum_solver_options

        self._solver = VQLS(
            Estimator(),  # bugs when the estimator is not reset ...
            self.ansatz,
            self.optimizer,
            sampler=self.sampler,
            initial_point=self.initial_point,
            gradient=self.gradient,
            max_evals_grouped=self.max_evals_grouped,
            options=self.quantum_solver_options,
        )
        self.matrix_decomposition = None

    def __call__(self, A: sparray, b: np.ndarray) -> VQLSResult:
        """Solve the linear system using VQLS.

        Args:
            A (sparray): matrix of the linear syste
            b (np.ndarray): righ habd side vector
            quantum_solver_options (Dict): options for the solver
        """

        def post_process_vqls_solution(A, y, x):
            """Retreive the  norm and direction of the solution vector.

            VQLS provides a normalized form of the solution vector
            that can also have a -1 prefactor. This routine retrieves
            the un-normalized solution vector with the correct prefactor.

            Args:
                A (np.ndarray): matrix of the linear system
                y (np.ndarray): rhs of the linear system
                x (np.ndarray): proposed solution
            """
            Ax = A @ x
            normy = np.linalg.norm(y)
            normAx = np.linalg.norm(Ax)
            prefac = normy / normAx

            if np.dot(Ax * prefac, y) < 0:
                prefac *= -1
            sol = prefac * x
            return sol

        # convert the input data inot a spsparse compatible format
        A, b = preprocess_data(A, b)

        # preprocess the initial matrix
        A = A.todense()  # <= TO DO: allow for sparse matrix

        # use the input matrix of update the matrix decomposition
        if self.matrix_decomposition is None:
            self.matrix_decomposition = A
        else:
            self.matrix_decomposition.update_matrix(A)

        # solver
        res = self._solver.solve(self.matrix_decomposition, b)

        # extract the results
        x = post_process_vqls_solution(A, b, res.vector)
        ref = np.linalg.solve(A, b)  # <= of course we need to remove that at some point
        residue = np.linalg.norm(A @ x - b)
        return VQLSResult(x, residue, self._solver.logger, ref)
