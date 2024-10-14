import numpy as np
from qiskit.primitives import Estimator
from scipy.sparse import csc_matrix
from scipy.sparse import linalg as spla
from scipy.sparse import sparray
from vqls_prototype import VQLS
from .base_solver import BaseSolver
from .preconditioners import DiagonalScalingPreconditioner
from .result import VQLSResult
from .utils import pad_input
from .utils import post_process_solution
from .utils import preprocess_data

SUPPORTED_PRECONDITIONERS = {
    "diagonal_scaling": DiagonalScalingPreconditioner,
}


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
        self.initial_point = quantum_solver_options.pop("initial_point", None)
        self.gradient = quantum_solver_options.pop("gradient", None)
        self.max_evals_grouped = quantum_solver_options.pop("max_evals_grouped", 1)
        self.reorder = quantum_solver_options.pop("reorder", False)
        self.preconditioner = quantum_solver_options.pop("preconditioner", None)

        # compute the classical solution
        self.compute_classical_solution = quantum_solver_options.pop(
            "compute_classical_solution", False
        )

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

        # register the decomposed matrix so that we can
        # update it instead of recalculating the decomposition
        self.decomposed_matrix = None

    def __call__(self, A: sparray, b: np.ndarray) -> VQLSResult:
        """Solve the linear system using VQLS.

        Args:
            A (sparray): matrix of the linear syste
            b (np.ndarray): righ habd side vector
        """
        # pad the input data if necessary
        A, b, original_input_size = pad_input(A, b)

        # convert the input data inot a spsparse compatible format
        A, b = preprocess_data(A, b)

        # preconditioning
        if self.preconditioner:
            preconditioner = SUPPORTED_PRECONDITIONERS[self.preconditioner](A, b)
            A, b = preconditioner.apply()
        else:
            # preprocess the initial matrix
            A = A.todense()  # <= TO DO: allow for sparse matrix

        # reorder the matrix elements to limit the number of Pauli gates
        if self.reorder:
            lu = spla.splu(csc_matrix(A), permc_spec="COLAMD")
            idx = lu.perm_c
            A = A[np.ix_(idx, idx)]
            b = b[idx]

        # use the input matrix of update the matrix decomposition
        if self.decomposed_matrix is None:
            # set it to the input matrix at the first call
            self.decomposed_matrix = A
        else:
            # update the matrix on the subsequent call
            self.decomposed_matrix.update_matrix(A)

        # solver
        res = self._solver.solve(self.decomposed_matrix, b)

        # recover the original order
        if self.reorder:
            idx_r = np.argsort(idx)
            A = A[np.ix_(idx_r, idx_r)]
            b = b[idx_r]
            res.vector = res.vector[idx_r]

        # recover original problem
        if self.preconditioner:
            A, b, res.vector = preconditioner.reverse(A, b, res.vector)

        # extract the results
        A, b, x = post_process_solution(A, b, res.vector, original_input_size)
        residue = np.linalg.norm(A @ x - b)

        # classical check
        if self.compute_classical_solution:
            ref = np.linalg.solve(A, b)
        else:
            ref = np.zeros(original_input_size)

        # register the matrix decomposition of the solver
        self.decomposed_matrix = self._solver.matrix_circuits

        return VQLSResult(x, residue, self._solver.logger, ref)
