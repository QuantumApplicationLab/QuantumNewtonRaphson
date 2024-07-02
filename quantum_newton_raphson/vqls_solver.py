import numpy as np
from qiskit.primitives import Estimator
from scipy.sparse import csc_matrix
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

        self.preconditioning = (
            quantum_solver_options.pop("preconditioning")
            if "preconditioning" in quantum_solver_options
            else None
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
            quantum_solver_options (Dict): options for the solver
        """

        def pad_input(A, y):
            """Process the input data to pad to power of two size.

            Args:
                A (sparray): matrix of the linear system
                y (np.ndarray): rhs of the linear system
            """
            input_size = A.shape[0]
            if np.log2(input_size).is_integer():
                return A, y, input_size
            else:
                # new size
                new_size = 2 ** int(np.ceil(np.log2(input_size)))

                # pad matrix
                Afull = np.eye(new_size)
                Afull[:input_size, :input_size] = A.todense()

                # convert pad matrix into a sparse matrix
                Afull = csc_matrix(Afull)

                # pad vector
                yfull = np.zeros(new_size)
                yfull[:input_size] = y
                return Afull, yfull, input_size

        def post_process_vqls_solution(A, y, x, original_input_size):
            """Retreive the  norm and direction of the solution vector.

            VQLS provides a normalized form of the solution vector
            that can also have a -1 prefactor. This routine retrieves
            the un-normalized solution vector with the correct prefactor.

            Args:
                A (np.ndarray): matrix of the linear system
                y (np.ndarray): rhs of the linear system
                x (np.ndarray): proposed solution
                original_input_size (int): size of the original vector
            """
            # unpad the data
            A = A[:original_input_size, :original_input_size]
            x = x[:original_input_size]
            y = y[:original_input_size]

            # compute the prefactor
            Ax = A @ x
            normy = np.linalg.norm(y)
            normAx = np.linalg.norm(Ax)
            prefac = normy / normAx

            # possible flip sign
            if np.dot(Ax * prefac, y) < 0:
                prefac *= -1
            sol = prefac * x
            return A, y, sol

        def preconditioner(A, b):
            """Precondition the linear system using a diagonal scaling preconditioner.

            This function takes a sparse matrix A and a right-hand side vector b,
            and returns the preconditioned matrix A_hat, the preconditioned vector b_hat,
            and the inverse of the diagonal scaling matrix D_inv used for preconditioning.

            Args:
                A (scipy.sparse.sparray): The matrix of the linear system.
                b (np.ndarray): The right-hand side vector of the linear system.

            Returns:
                A_hat (scipy.sparse.csc_matrix): The preconditioned matrix.
                b_hat (np.ndarray): The preconditioned right-hand side vector.
                D_inv (np.ndarray): The inverse of the diagonal scaling matrix used for preconditioning.
            """
            # Compute the diagonal preconditioner
            D = np.diag(np.sqrt(np.diag(A.todense())))
            D_inv = np.linalg.inv(D)

            # Preconditioned system
            A_hat = D_inv @ A @ D_inv
            b_hat = D_inv @ b

            return csc_matrix(A_hat), b_hat, D_inv

        # pad the input data if necessary
        A, b, original_input_size = pad_input(A, b)

        # convert the input data inot a spsparse compatible format
        A, b = preprocess_data(A, b)

        if self.preconditioning:
            A, b, D_inv = preconditioner(A, b)

        # preprocess the initial matrix
        A = A.todense()  # <= TO DO: allow for sparse matrix

        # use the input matrix of update the matrix decomposition
        if self.decomposed_matrix is None:
            # set it to the input matrix at the first call
            self.decomposed_matrix = A
        else:
            # update the matrix on the subsequent call
            self.decomposed_matrix.update_matrix(A)

        # solver
        res = self._solver.solve(self.decomposed_matrix, b)

        # extract the results
        A, b, x = post_process_vqls_solution(A, b, res.vector, original_input_size)
        residue = np.linalg.norm(A @ x - b)

        # classical check
        ref = np.linalg.solve(A, b)  # <= of course we need to remove that at some point

        # register the matrix decomposition of the solver
        self.decomposed_matrix = self._solver.matrix_circuits

        if self.preconditioning:
            # recover the solution of the original system
            x = D_inv @ x
            ref = D_inv @ x

        return VQLSResult(x, residue, self._solver.logger, ref)
