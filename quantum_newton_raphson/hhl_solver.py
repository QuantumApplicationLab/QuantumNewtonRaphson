import numpy as np
from hhl_prototype.solver.hhl import HHL
from scipy.sparse import sparray
from .base_solver import BaseSolver
from .preconditioners import DiagonalScalingPreconditioner
from .result import HHLResult
from .utils import preprocess_data

SUPPORTED_PRECONDITIONERS = {
    "diagonal_scaling": DiagonalScalingPreconditioner,
}


class HHL_SOLVER(BaseSolver):
    """Solver using HHL.

    Args:
        BaseSolver (object): base solver class
    """

    def __init__(self, **quantum_solver_options):
        """Init the hhl solver."""
        # extract required options for the hhl solver
        self.estimator = quantum_solver_options.pop("estimator")

        # extract optional options for the hhl solver
        self.sampler = quantum_solver_options.pop("sampler", None)
        self.preconditioner = quantum_solver_options.pop("preconditioner", None)

        # Check if the provided preconditioner is supported
        if (
            self.preconditioner is not None
            and self.preconditioner not in SUPPORTED_PRECONDITIONERS
        ):
            raise ValueError(
                f"Unsupported preconditioner '{self.preconditioner}'. "
                f"Supported preconditioners are: {list(SUPPORTED_PRECONDITIONERS.keys())}"
            )

        self.quantum_solver_options = quantum_solver_options

        # solver
        self._solver = HHL(
            self.estimator,
            sampler=self.sampler,
            options=quantum_solver_options,
        )

    def __call__(self, A: sparray, b: np.ndarray) -> HHLResult:
        """Solve the linear system using HHL.

        Args:
            A (sparray): matrix of the linear syste
            b (np.ndarray): righ habd side vector
            quantum_solver_options (Dict): options for the solver
        """

        def post_process_hhl_solution(A, y, x):
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

        # convert the input data into a sparse compatible format
        A, b = preprocess_data(A, b)

        # preconditioning
        if self.preconditioner:
            preconditioner = SUPPORTED_PRECONDITIONERS[self.preconditioner](A, b)
            A, b = preconditioner.apply()
        else:
            # preprocess the initial matrix
            A = A.todense()  # <= TO DO: allow for sparse matrix

        # preprocess the b vector
        norm_b = np.linalg.norm(b)
        bnorm = np.copy(b)
        bnorm /= norm_b

        # solver
        res = self._solver.solve(A, b)

        # recover original problem
        if self.preconditioner:
            A, b, res.vector = preconditioner.reverse(A, b, res.solution)

        # extract the results
        return HHLResult(post_process_hhl_solution(A, b, res.solution))
