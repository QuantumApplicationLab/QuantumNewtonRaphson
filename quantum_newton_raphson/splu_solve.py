from typing import Dict
from typing import Tuple
from typing import Union
import numpy as np
from qreorder.classical_ordering import find_ordering as find_reordering_classical
from qreorder.quantum_ordering import find_ordering as find_reordering_quantum
from scipy.sparse import sparray
from scipy.sparse import triu
from scipy.sparse.linalg import splu
from .result import SPLUResult
from .utils import preprocess_data

ValidInputFormat = Union[sparray, Tuple, np.ndarray]


def get_ordering(A: sparray, reorder_method, **options) -> np.ndarray:
    """Get the reordering.

    Args:
        A (sparray): input matrix
        reorder_method (str): method to get the ordering
        options (Dict): options of the reodering method

    Returns:
        np.ndarray: ordering indices
    """
    reordering_functions = {
        "no_reordering": get_orginal_ordering,
        "max_edge": get_max_edge_ordering,
        "classical": get_classical_minimal_fill_ordering,
        "quantum": get_quantum_minimal_fill_ordering,
    }

    if reorder_method not in reordering_functions:
        raise ValueError(
            "Ordering method not recognized, valid options are {}".format(
                list(reordering_functions.keys())
            )
        )

    return reordering_functions[reorder_method](A, **options)


def get_orginal_ordering(A: sparray) -> np.ndarray:
    """Return the original ordering.

    Args:
        A (sparray): input matrix

    Returns:
        np.ndarray: ordering indices
    """
    size = A.shape[0]
    return np.array(range(size))


def get_max_edge_ordering(A: sparray) -> np.ndarray:
    """Get ordering of the matrix using the maximum number of edges.

    Args:
        A (sparray): input matrix

    Returns:
        np.ndarray: ordering indices
    """
    idx = np.argsort(triu(A, k=1).sum(1).flatten())
    return np.array(idx).ravel()


def get_classical_minimal_fill_ordering(A: sparray, **kwargs) -> np.ndarray:
    """Get the classical minimal fill ordering classicaly.

    Args:
        A (sparray): Input of the matrix
        **kwargs(Dict): options for the ordering method

    Returns:
        np.ndarray: ordering indices
    """
    idx, _ = find_reordering_classical(A, **kwargs)
    return idx


def get_quantum_minimal_fill_ordering(A: sparray, **kwargs) -> np.ndarray:
    """Get the ordering of the matrix element following the quantum approach.

    Args:
        A (sparray): inout matrix
        **kwargs (Dict, optional): options of the quantum routine Defaults to {}.

    Returns:
        np.ndarray: ordering indices
    """
    idx, _ = find_reordering_quantum(A, **kwargs)
    return idx


def splu_solve(A: ValidInputFormat, b: ValidInputFormat, **options) -> SPLUResult:
    """Solve the linear system by reordering the system of eq.

    Args:
        A (ValidInputFormat): input matrix
        b (ValidInputFormat): input rhs
        options (dict, optional): options for the reordering. Defaults to {}.

    Returns:
        SPLUResult: object containing all the results of the solver
    """
    # convert the input data inot a spsparse compatible format
    A, b = preprocess_data(A, b)

    # get order
    reorder_method = options.pop("reorder") if "reorder" in options else "max_edge"
    order = get_ordering(A, reorder_method, **options)

    # reorder matrix and rhs
    A = A[np.ix_(order, order)]
    b = b[order]

    # solve
    solver = splu(A, permc_spec="NATURAL")
    x = solver.solve(b)

    # reorder solution
    x = x[np.argsort(order)]
    return SPLUResult(x, solver)
