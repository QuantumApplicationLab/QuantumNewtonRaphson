import numpy as np
from hhl_prototype.solver.hhl import HHL
from scipy.sparse import sparray
from .base_solver import BaseSolver
from .preconditioners import DiagonalScalingPreconditioner
from .result import HHLResult
from .utils import pad_input
from .utils import post_process_solution
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

        # precision
        default_epsilon = 1e-2
        self.epsilon = quantum_solver_options.pop("epsilon", default_epsilon)

        # solver
        self._solver = HHL(self.estimator, sampler=self.sampler, epsilon=self.epsilon)

    def __call__(self, A: sparray, b: np.ndarray) -> HHLResult:
        """Solve the linear system using HHL.

        Args:
            A (sparray): matrix of the linear syste
            b (np.ndarray): righ habd side vector
        """
        # pad the input data if necessary
        A, b, original_input_size = pad_input(A, b)

        # convert the input data into a sparse compatible format
        A, b = preprocess_data(A, b)

        # preconditioning
        if self.preconditioner:
            preconditioner = SUPPORTED_PRECONDITIONERS[self.preconditioner](A, b)
            A, b = preconditioner.apply()
        else:
            # preprocess the initial matrix
            A = A.todense()  # <= TO DO: allow for sparse matrix

        # solver
        res = self._solver.solve(A, b)

        # recover original problem
        if self.preconditioner:
            A, b, res.vector = preconditioner.reverse(A, b, res.solution)

        # extract the results
        A, b, x = post_process_solution(A, b, res.vector, original_input_size)
        residue = np.linalg.norm(A @ x - b)

        # classical check
        ref = np.linalg.solve(A, b)  # <= of course we need to remove that at some point

        # extract the results
        return HHLResult(x, residue, ref)
