import numpy as np
from typing import Dict
from scipy.sparse import sparray, triu
from scipy.sparse.linalg import splu
from .result import SPLUResult


def get_ordering(A: sparray, reorder_method: str, options: Dict) -> np.ndarray:
    """get the reordering

    Args:
        A (sparray): input matrix
        reorder_method (str): method to get the ordering
        options (Dict): options of the reodering method

    Returns:
        np.ndarray: ordering indices
    """

    reordering_functions = {
        "max_edge": get_max_edge_ordering,
        "quantum": get_quantum_ordering,
        "no_reordering": get_orginal_ordering,
    }

    if reorder_method not in reordering_functions:
        raise ValueError(
            "Ordering method not recognized, valid options are {}".format(
                list(reordering_functions.keys())
            )
        )

    return reordering_functions[reorder_method](A, **options)


def get_orginal_ordering(A: sparray) -> np.ndarray:
    """Return the original ordering

    Args:
        A (sparray): input matrix

    Returns:
        np.ndarray: ordering indices
    """
    size = A.shape[0]
    return np.array(range(size))


def get_max_edge_ordering(A: sparray) -> np.ndarray:
    """Get ordering of the matrix using the maximum number of edges

    Args:
        A (sparray): input matrix

    Returns:
        np.ndarray: ordering indices
    """
    idx = np.argsort(triu(A, k=1).sum(1).flatten())
    return np.array(idx).ravel()


def get_quantum_ordering(A: sparray, options: Dict = {}) -> np.ndarray:
    """Get the ordering of the matrix element following the quantum approach

    Args:
        A (sparray): inout matrix
        options (Dict, optional): options of the quantum routine Defaults to {}.

    Returns:
        np.ndarray: ordering indices
    """
    raise NotImplementedError(
        "Quantum routine for matrix reordeing not implemented yet"
    )


def splu_solve(A: sparray, b: np.ndarray, options: Dict = {}) -> SPLUResult:
    """Solve the linear system by reordering the system of eq.

    Args:
        A (sparray): input matrix
        b (np.ndarray): input rhs
        options (dict, optional): options for the reordering. Defaults to {}.

    Returns:
        SPLUResult: object containing all the results of the solver
    """

    # get order
    reorder_method = options.pop("reorder") if "reorder" in options else "max_edge"
    order = get_ordering(A, reorder_method, options)

    # reorder matrix and rhs
    A = A[np.ix_(order, order)]
    b = b[order]

    # solve
    solver = splu(A, permc_spec="NATURAL")
    x = solver.solve(b)

    # reorder solution
    x = x[np.argsort(order)]
    return SPLUResult(x, solver)
