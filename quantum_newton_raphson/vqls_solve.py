import numpy  as np 
from typing import List, Dict

from scipy.sparse import csr_matrix
from qalcore.qiskit.vqls import VQLS

from .result import VQLSResult

def vqlssolve(Agraph: List, blist: List, quantum_solver_options: Dict = {}):
    """Solve the linear system using VQLS

    Args:
        Agraph (tuple): graph representing the A matrix
        blist (List): b vector
        quantum_solver_options (dict): options for the solver
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

    # create a sparse matrix from the graph
    mat = csr_matrix(Agraph)

    # preprocess the b vector
    b = np.array(blist)
    norm_b = np.linalg.norm(b)
    b /= norm_b

    # extract required options for the vqls solver
    estimator = quantum_solver_options.pop('estimator')
    ansatz = quantum_solver_options.pop('ansatz')
    optimizer = quantum_solver_options.pop('optimizer')

    # extract optional options for the vqls solver
    sampler = quantum_solver_options.pop('sampler') if 'sampler' in quantum_solver_options else None
    initial_point = quantum_solver_options.pop('initial_point') if 'initial_point' in quantum_solver_options else None
    gradient = quantum_solver_options.pop('gradient') if 'gradient' in quantum_solver_options else None
    max_evals_grouped = quantum_solver_options.pop('max_evals_grouped') if 'max_evals_grouped' in quantum_solver_options else None

    # solver
    vqls = VQLS(
        estimator,
        ansatz,
        optimizer,
        sampler=sampler,
        initial_point=initial_point,
        gradient=gradient,
        max_evals_grouped=max_evals_grouped,
        options=quantum_solver_options
    )

    # solver
    res = vqls.solve(mat, b)

    # extract the results
    return VQLSResult(post_process_vqls_solution(mat, b, res.vector))

