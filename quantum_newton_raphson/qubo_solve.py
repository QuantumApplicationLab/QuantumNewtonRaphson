from typing import Dict
import numpy as np
from qalcore.dwave.qubols.encodings import EfficientEncoding
from qalcore.dwave.qubols.qubols import QUBOLS
from scipy.sparse import sparray
from .result import QUBOResult
from .utils import preprocess_data


def qubosolve_real(A: sparray, b: np.ndarray, options: Dict = {}) -> QUBOResult:
    """Solve a real system of euqations using QUBO linear solver.

    Args:
        A (sparray): input matrix
        b (np.ndarray): right hand side
        options (Dict, optional): Options for the quantum solver. Defaults to {}.

    Returns:
        QUBOResult: solution of the system
    """
    # convert the input data inot a spsparse compatible format
    A, b = preprocess_data(A, b)

    # preprocess options
    if "encoding" not in options:
        options["encoding"] = EfficientEncoding

    # preprocess the b vector
    norm_b = np.linalg.norm(b)
    bnorm = np.copy(b)
    bnorm /= norm_b

    # solve
    update = QUBOLS(options).solve(A, bnorm)

    # postporcess solution
    update *= norm_b

    return QUBOResult(update)
