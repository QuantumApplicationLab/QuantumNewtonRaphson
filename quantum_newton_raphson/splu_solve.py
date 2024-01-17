import numpy as np
from typing import Dict
from scipy.sparse import sparray, triu
from .result import SPLUResult


def get_ordering(A: sparray, reorder_method: str, options: Dict):
    """get the reordering

    Args:
        A (sparray): input matrix
        reorder_method (str): method to get the ordering
        options (Dict): options of the reodering method
    """

    if reorder_method == "max_edge":
        idx = get_max_edge_ordering(A, **options)
    elif reorder_method == "quantum":
        idx = get_quantum_ordering(A, **options)
    else:
        raise ValueError("Ordering method not recognized")
    return idx


def get_max_edge_ordering(A: sparray):
    """Get ordering of the matrix

    Args:
        A (sparray): input matrix
    """
    return np.argsort(triu(A, k=1).sum(1))


def get_quantum_ordering(A: sparray, options: Dict = {}):
    """Get the ordering of the matrix element following the quantum approach

    Args:
        A (sparray): inout matrix
        options (Dict, optional): options of the quantum routine Defaults to {}.
    """
    raise NotImplementedError(
        "Quantum routine for matrix reordeing not implemented yet"
    )


def splu_solve(A: sparray, b: sparray, options: Dict = {}):
    """Solve the linear system by reordering the system of eq.

    Args:
        A (sparray): input matrix
        b (sparray): input rhs
        options (dict, optional): _description_. Defaults to {}.
    """

    # get order
    reorder_method = options.pop("reorder") if "reoder" in options else "max_edge"
    order = get_ordering(A, reorder_method)

    # reorder matrix and rhs
    A = A[np.ix_(order, order)]
    b = b[order]

    # solve
    splu = splu(A, permc_spec="NATURAL")
    x = splu.solve(b)

    # reorder solution
    x = x[np.argsort(order)]
    return SPLUResult(x, splu)
