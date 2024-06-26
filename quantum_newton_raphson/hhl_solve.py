from typing import Dict
import numpy as np
#fix this import
from hhl_prototype.solver.hhl import HHL
from scipy.sparse import sparray
from .result import HHLResult
from .utils import preprocess_data


def hhlsolve(
    A: sparray, b: np.ndarray, quantum_solver_options: Dict = {}
) -> HHLResult:
    """Solve the linear system using VQLS.

    Args:
        A (sparray): matrix of the linear syste
        b (np.ndarray): righ hand side vector
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

    # preprocess the initial matrix
    A = A.todense()

    # preprocess the b vector
    norm_b = np.linalg.norm(b)
    bnorm = np.copy(b)
    bnorm /= norm_b

    # extract required options for the hhl solver
    estimator = quantum_solver_options.pop("estimator")


    # extract optional options for the hhl solver
    sampler = (
        quantum_solver_options.pop("sampler")
        if "sampler" in quantum_solver_options
        else None
    )

    # solver
    hhl = HHL(
        estimator,
        sampler=sampler,
        options=quantum_solver_options,
    )

    # solver
    res = hhl.solve(A, b)

    # extract the results
    return HHLResult(post_process_hhl_solution(A, b, res.vector))
