import numpy as np
from typing import Dict

from scipy.sparse import sparray
from qalcore.qiskit.vqls import VQLS
from .utils import preprocess_data
from .result import VQLSResult


def vqlssolve(A: sparray, b: np.ndarray, quantum_solver_options: Dict = {}) -> VQLSResult:
    """Solve the linear system using VQLS

    Args:
        A (sparray): matrix of the linear syste
        b (np.ndarray): righ habd side vector
        quantum_solver_options (Dict): options for the solver
    """

    def post_process_vqls_solution(A, y, x):
        """Retreive the  norm and direction of the solution vector
           VQLS provides a normalized form of the solution vector
           that can also have a -1 prefactor. This routine retrieves
           the un-normalized solution vector with the correct prefactor

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
    A = A.todense()

    # preprocess the b vector
    norm_b = np.linalg.norm(b)
    bnorm = np.copy(b)
    bnorm /= norm_b

    # extract required options for the vqls solver
    estimator = quantum_solver_options.pop("estimator")
    ansatz = quantum_solver_options.pop("ansatz")
    optimizer = quantum_solver_options.pop("optimizer")

    # extract optional options for the vqls solver
    sampler = quantum_solver_options.pop("sampler") if "sampler" in quantum_solver_options else None
    initial_point = quantum_solver_options.pop("initial_point") if "initial_point" in quantum_solver_options else None
    gradient = quantum_solver_options.pop("gradient") if "gradient" in quantum_solver_options else None
    max_evals_grouped = (
        quantum_solver_options.pop("max_evals_grouped") if "max_evals_grouped" in quantum_solver_options else 1
    )

    # solver
    vqls = VQLS(
        estimator,
        ansatz,
        optimizer,
        sampler=sampler,
        initial_point=initial_point,
        gradient=gradient,
        max_evals_grouped=max_evals_grouped,
        options=quantum_solver_options,
    )

    # solver
    res = vqls.solve(A, b)

    # extract the results
    return VQLSResult(post_process_vqls_solution(A, b, res.vector))
